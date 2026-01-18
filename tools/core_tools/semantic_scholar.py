"""
Semantic Scholar 文献检索器

提供从 Semantic Scholar 数据库检索学术论文元数据并下载全文文档的功能。
遵循与 PubMedSearcher 和 OpenAlexSearcher 相同的设计模式，提供独立的 search 和 download 方法。

Semantic Scholar 是由 Allen Institute for AI 开发的免费学术搜索引擎，包含超过 2 亿篇学术论文，
提供丰富的元数据、引用分析和开放获取 PDF 链接。API 支持可选的 API Key 以获得更高的请求限制
（无 Key 时约 100 请求/5分钟，有 Key 时约 1 请求/秒）。

使用示例:

主要功能:
    - search: 根据关键词检索 Semantic Scholar 文献，返回 Paper 对象列表
    - download: 根据 Paper 对象中的 pdf_url 下载全文文档

Author: Semantic Scholar Searcher
Version: 1.0.0
"""

import os
import httpx
from typing import List, Union, Optional
from datetime import datetime
from dotenv import load_dotenv
from tools.core_tools.paper import Paper
from core.config import Config
import asyncio
load_dotenv()
class SemanticScholarSearcher:
    """
    Semantic Scholar 文献检索器类
    
    用于从 Semantic Scholar 数据库检索学术论文元数据并下载全文文档。
    
    Attributes:
        api_key: 可选的 API Key，用于获得更高的请求限制
        base_url: Semantic Scholar API 基础 URL
        headers: HTTP 请求头
        _fields: API 请求字段列表
    """
    
    def __init__(self, api_key: str = None, size: int = None):
        """
        初始化 SemanticScholarSearcher
        
        Args:
            api_key: 可选的 API Key，用于获得更高的请求限制。
                     默认从环境变量 SEMANTIC_API_KEY 读取。
                     无 Key 时约 100 请求/5分钟，有 Key 时约 1 请求/秒。
            size: 检索返回数量，默认使用 Config.SEARCH_SIZE
        """
        # 优先使用传入的 api_key，否则从环境变量读取
        self.api_key = api_key or os.getenv("SEMANTIC_API_KEY")
        self.size = size or Config.SEARCH_SIZE
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
        # 设置 HTTP 请求头
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "SemanticScholarSearcher/1.0"
        }
        
        # 如果提供了 API Key，添加到请求头
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
        
        # 定义请求字段列表，用于获取完整的论文信息
        # 参考设计文档中的 API 请求字段配置
        self._fields = [
            "paperId",
            "corpusId",
            "url",
            "title",
            "abstract",
            "venue",
            "publicationVenue",
            "year",
            "referenceCount",
            "citationCount",
            "influentialCitationCount",
            "isOpenAccess",
            "openAccessPdf",
            "fieldsOfStudy",
            "s2FieldsOfStudy",
            "publicationTypes",
            "publicationDate",
            "journal",
            "authors",
            "externalIds"
        ]

    async def _search_papers(
        self, 
        client: httpx.AsyncClient, 
        query: str, 
        limit: int
    ) -> List[dict]:
        """
        调用 Semantic Scholar API 搜索文献
        
        Args:
            client: HTTP 客户端
            query: 检索关键词
            limit: 最大返回数量
            
        Returns:
            List[dict]: Semantic Scholar Paper 对象列表
        """
        # 构建请求字段参数
        fields_param = ",".join(self._fields)
        
        # 构建 API 请求 URL
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": fields_param
        }
        
        try:
            response = await client.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            # API 返回格式: {"total": int, "offset": int, "next": int, "data": [...]}
            return data.get("data", [])
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"警告: API 限流，请稍后重试或使用 API Key")
            else:
                print(f"HTTP 错误: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"网络请求错误: {e}")
            return []
        except Exception as e:
            print(f"解析响应失败: {e}")
            return []

    def _extract_pdf_url(self, paper_data: dict) -> str:
        """
        从 Semantic Scholar Paper 对象中提取 PDF 下载链接
        
        Args:
            paper_data: Semantic Scholar Paper 对象
            
        Returns:
            str: PDF 下载链接，如果没有则返回空字符串
        """
        open_access_pdf = paper_data.get("openAccessPdf")
        if open_access_pdf and isinstance(open_access_pdf, dict):
            return open_access_pdf.get("url", "")
        return ""

    def _map_to_paper(self, paper_data: dict) -> Paper:
        """
        将 Semantic Scholar Paper 对象映射到 Paper 类
        
        Args:
            paper_data: Semantic Scholar Paper 对象（字典）
            
        Returns:
            Paper: 转换后的 Paper 对象
        """
        # 解析 paperId → paper_id
        paper_id = paper_data.get("paperId", "")
        
        # 解析 title → title
        title = paper_data.get("title", "")
        
        # 解析 authors[].name → authors
        authors_data = paper_data.get("authors", [])
        authors = []
        if authors_data:
            for author in authors_data:
                if isinstance(author, dict) and author.get("name"):
                    authors.append(author["name"])
        
        # 解析 abstract → abstract
        abstract = paper_data.get("abstract", "") or ""
        
        # 解析 externalIds.DOI → doi
        external_ids = paper_data.get("externalIds", {}) or {}
        doi = external_ids.get("DOI", "") or ""
        
        # 解析 publicationDate → published_date
        published_date = None
        pub_date_str = paper_data.get("publicationDate")
        if pub_date_str:
            try:
                published_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
            except ValueError:
                # 尝试只解析年份
                year = paper_data.get("year")
                if year:
                    try:
                        published_date = datetime(int(year), 1, 1)
                    except (ValueError, TypeError):
                        pass
        elif paper_data.get("year"):
            try:
                published_date = datetime(int(paper_data["year"]), 1, 1)
            except (ValueError, TypeError):
                pass
        
        # 调用 _extract_pdf_url 获取 pdf_url
        pdf_url = self._extract_pdf_url(paper_data)
        
        # 解析 url → url
        url = paper_data.get("url", "")
        
        # 解析 s2FieldsOfStudy 或 fieldsOfStudy → categories
        categories = []
        s2_fields = paper_data.get("s2FieldsOfStudy", [])
        if s2_fields:
            for field in s2_fields:
                if isinstance(field, dict) and field.get("category"):
                    categories.append(field["category"])
        else:
            # 回退到 fieldsOfStudy
            fields_of_study = paper_data.get("fieldsOfStudy", [])
            if fields_of_study:
                categories = [f for f in fields_of_study if f]
        
        # 去重
        categories = list(dict.fromkeys(categories))
        
        # 解析 citationCount → citations
        citations = paper_data.get("citationCount", 0) or 0
        
        # 解析 referenceCount → references（存为列表格式）
        reference_count = paper_data.get("referenceCount", 0) or 0
        references = [str(reference_count)]  # 存储为列表格式，包含数量
        
        # 设置 extra 包含额外元数据
        extra = {
            "venue": paper_data.get("venue", ""),
            "year": paper_data.get("year"),
            "publicationTypes": paper_data.get("publicationTypes", []),
            "isOpenAccess": paper_data.get("isOpenAccess", False),
            "influentialCitationCount": paper_data.get("influentialCitationCount", 0),
            "externalIds": external_ids,
            "corpusId": paper_data.get("corpusId"),
            "publicationVenue": paper_data.get("publicationVenue"),
            "journal": paper_data.get("journal"),
            "openAccessPdf": paper_data.get("openAccessPdf")
        }
        
        return Paper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            published_date=published_date,
            pdf_url=pdf_url,
            url=url,
            source="semanticscholar",
            updated_date=None,  # Semantic Scholar 不提供此字段
            categories=categories,
            keywords=[],  # Semantic Scholar 不提供关键词字段
            citations=citations,
            references=references,
            extra=extra
        )

    def _get_browser_headers(self) -> dict:
        """获取模拟浏览器的请求头"""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _is_valid_pdf(self, content: bytes) -> bool:
        """检查内容是否为有效 PDF"""
        return len(content) > 10000 and content[:4] == b'%PDF'

    def _find_existing_paper_by_doi(self, save_path: str, doi: str) -> Optional[str]:
        """
        在保存目录中查找是否已存在具有相同 DOI 的文件
        
        Args:
            save_path: 保存目录
            doi: 论文的 DOI
            
        Returns:
            Optional[str]: 如果找到已存在的文件，返回文件路径；否则返回 None
        """
        if not doi or not os.path.exists(save_path):
            return None
        
        # 将 DOI 转换为文件名格式（替换 / 为 _）
        safe_doi = doi.replace("/", "_")
        expected_filename = f"{safe_doi}.pdf"
        expected_path = os.path.join(save_path, expected_filename)
        
        # 检查标准文件名是否存在
        if os.path.exists(expected_path):
            return expected_path
        
        # 遍历目录中的所有 PDF 文件，检查文件名是否包含该 DOI
        try:
            for filename in os.listdir(save_path):
                if filename.endswith(".pdf") and safe_doi in filename:
                    return os.path.join(save_path, filename)
        except Exception:
            pass
        
        return None

    def _get_alternative_pdf_urls(self, paper: Paper) -> List[str]:
        """
        获取论文的所有可能的 PDF 下载链接
        
        Args:
            paper: Paper 对象
            
        Returns:
            List[str]: PDF 链接列表，按优先级排序
        """
        urls = []
        
        # 1. 原始 openAccessPdf URL
        if paper.pdf_url:
            urls.append(paper.pdf_url)
        
        # 2. 从 extra 中获取 externalIds
        external_ids = {}
        if paper.extra:
            external_ids = paper.extra.get("externalIds", {}) or {}
        
        # 3. ArXiv 直链（最可靠的开放获取源）
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id:
            urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        
        # 4. 从 DOI 尝试获取（通过 Unpaywall）
        doi = paper.doi or external_ids.get("DOI")
        if doi:
            # Unpaywall 会在下载时动态查询
            pass
        
        # 5. PubMed Central
        pmc_id = external_ids.get("PubMedCentral")
        if pmc_id:
            urls.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/")
        
        return urls

    async def _get_unpaywall_pdf_url(self, client: httpx.AsyncClient, doi: str) -> Optional[str]:
        """通过 Unpaywall API 获取 PDF 链接"""
        if not doi:
            return None
        
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": Config.EMAIL}
            
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                best_oa = data.get("best_oa_location")
                if best_oa and best_oa.get("url_for_pdf"):
                    return best_oa["url_for_pdf"]
                
                for loc in data.get("oa_locations", []):
                    if loc.get("url_for_pdf"):
                        return loc["url_for_pdf"]
        except Exception:
            pass
        
        return None

    async def _try_download_url(self, client: httpx.AsyncClient, url: str, file_path: str) -> bool:
        """尝试从单个 URL 下载文件"""
        try:
            headers = self._get_browser_headers()
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=15.0)
            
            if response.status_code != 200:
                return False
            
            content = response.content
            if not self._is_valid_pdf(content):
                return False
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            return True
        except Exception:
            return False

    async def _download_file(
        self, 
        client: httpx.AsyncClient, 
        paper: Paper, 
        save_path: str,
        max_retries: int = 2
    ) -> str:
        """
        下载单个文件，带多源重试机制
        
        Args:
            client: HTTP 客户端
            paper: Paper 对象
            save_path: 保存目录
            max_retries: 每个 URL 的最大重试次数
            
        Returns:
            str: 文件保存路径或 "No fulltext available"
        """

        
        # 检查 pdf_url 是否为空
        if not paper.pdf_url:
            return "No fulltext available"
        
        # 首先检查目录中是否已存在该 DOI 的文件
        doi = paper.doi
        if not doi and paper.extra:
            doi = paper.extra.get("externalIds", {}).get("DOI")
        
        if doi:
            existing_file = self._find_existing_paper_by_doi(save_path, doi)
            if existing_file:
                print(f"文件已存在（基于 DOI）: {existing_file}")
                return existing_file
        
        # 生成文件名（优先使用 DOI）
        if doi:
            safe_doi = doi.replace("/", "_")
            filename = f"{safe_doi}.pdf"
        elif paper.paper_id:
            filename = f"{paper.paper_id}.pdf"
        else:
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title[:50])
            filename = f"{safe_title.strip()}.pdf"
        
        file_path = os.path.join(save_path, filename)
        
        # 如果文件已存在，直接返回
        if os.path.exists(file_path):
            return file_path
        
        # 收集所有可能的下载链接
        urls_to_try = self._get_alternative_pdf_urls(paper)
        
        # 尝试从 Unpaywall 获取备用链接
        unpaywall_url = await self._get_unpaywall_pdf_url(client, doi)
        if unpaywall_url and unpaywall_url not in urls_to_try:
            urls_to_try.insert(1, unpaywall_url)
        
        tried_urls = set()
        
        for url in urls_to_try:
            if url in tried_urls:
                continue
            tried_urls.add(url)
            
            for attempt in range(max_retries):
                if attempt > 0:
                    await asyncio.sleep(1)
                
                if await self._try_download_url(client, url, file_path):
                    return file_path
        
        return "No fulltext available"

    async def search(self, query: str, limit: int = None) -> List[Paper]:
        """
        根据查询关键词检索 Semantic Scholar 文献
        
        Args:
            query: 检索关键词
            limit: 最大返回数量，默认使用 self.size（API 最大支持 100）
            
        Returns:
            List[Paper]: 符合条件的 Paper 对象列表，包含完整的元信息
        """
        if limit is None:
            limit = self.size
        
        papers = []
        
        # 创建 httpx.AsyncClient
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 调用 _search_papers 获取 Paper 数据列表
            paper_data_list = await self._search_papers(client, query, limit)
            
            # 遍历调用 _map_to_paper 转换为 Paper 列表
            for paper_data in paper_data_list:
                try:
                    paper = self._map_to_paper(paper_data)
                    papers.append(paper)
                except Exception as e:
                    print(f"解析论文数据失败: {e}")
                    continue
        
        return papers


    async def download(
        self, 
        papers: Union[Paper, List[Paper]], 
        save_path: str = None
    ) -> List[Paper]:
        """
        根据 Paper 对象中的 pdf_url 下载文档
        
        Args:
            papers: 单个 Paper 对象或 Paper 列表
            save_path: 保存路径，默认使用 Config.DOC_SAVE_PATH
            
        Returns:
            List[Paper]: 更新后的 Paper 列表，包含下载路径信息
        """
        # 处理单个 Paper 或 Paper 列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)
        
        # 使用默认保存路径
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH
        
        # 确保保存目录存在（不存在则创建）
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 统计有 PDF 链接的论文数量
        papers_with_pdf = [p for p in paper_list if p.pdf_url]
        print(f"共 {len(papers_with_pdf)} 篇论文待下载")
        
        success_count = 0
        # 创建 httpx.AsyncClient
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            # 遍历 Paper 列表调用 _download_file
            for paper in paper_list:
                saved_path = await self._download_file(client, paper, save_path)
                # 更新 Paper.extra["saved_path"]
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
                
                if saved_path and saved_path != "No fulltext available":
                    print(f"已保存: {saved_path}")
                    success_count += 1
        
        print(f"下载完成: {success_count}/{len(papers_with_pdf)}")
        
        # 返回更新后的 Paper 列表
        return paper_list


async def main():
    """
    使用示例：展示 search 和 download 方法
    """
    # 创建检索器实例（自动使用环境变量中的 SEMANTIC_API_KEY）
    searcher = SemanticScholarSearcher()
    
    print("=" * 60)
    print("Semantic Scholar 文献检索器")
    print("=" * 60)
    
    # 搜索文献
    query = "互联网+医疗 健康大数据"
    print(f"\n检索关键词: {query}")
    
    papers = await searcher.search(query)
    
    if papers:
        print(f"检索到 {len(papers)} 篇论文:\n")
        for i, paper in enumerate(papers, 1):
            print(f"[{i}] {paper.title}")
            print(f"    作者: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"    DOI: {paper.doi or '无'}")
            print(f"    PDF链接: {'有' if paper.pdf_url else '无'}")
            print()
        
        # 下载有 PDF 链接的论文
        papers_with_pdf = [p for p in papers if p.pdf_url]
        if papers_with_pdf:
            await searcher.download(papers_with_pdf)
    else:
        print("未检索到相关论文")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

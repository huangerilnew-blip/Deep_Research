"""
Semantic Scholar 文献检索器

提供从 Semantic Scholar 数据库检索学术论文元数据并下载全文文档的功能。
遵循与 PubMedSearcher 和 OpenAlexSearcher 相同的设计模式，提供独立的 search 和 download 方法。

Semantic Scholar 是由 Allen Institute for AI 开发的免费学术搜索引擎，包含超过 2 亿篇学术论文，
提供丰富的元数据、引用分析和开放获取 PDF 链接。API 支持可选的 API Key 以获得更高的请求限制
（无 Key 时约 100 请求/5分钟，有 Key 时约 1 请求/秒）。

使用示例:
    >>> import asyncio
    >>> from semantic_scholar import SemanticScholarSearcher
    >>> 
    >>> async def main():
    ...     searcher = SemanticScholarSearcher()
    ...     # 检索论文
    ...     papers = await searcher.search("machine learning", limit=5)
    ...     for paper in papers:
    ...         print(f"{paper.title} - {paper.doi}")
    ...     # 下载全文
    ...     papers = await searcher.download(papers)
    ...     for paper in papers:
    ...         print(f"Saved to: {paper.extra.get('saved_path')}")
    >>> 
    >>> asyncio.run(main())

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

from paper import Paper
from config import Config


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
    
    def __init__(self, api_key: str = None):
        """
        初始化 SemanticScholarSearcher
        
        Args:
            api_key: 可选的 API Key，用于获得更高的请求限制。
                     无 Key 时约 100 请求/5分钟，有 Key 时约 1 请求/秒。
        """
        self.api_key = api_key
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

    async def _download_file(
        self, 
        client: httpx.AsyncClient, 
        paper: Paper, 
        save_path: str
    ) -> str:
        """
        下载单个文件
        
        Args:
            client: HTTP 客户端
            paper: Paper 对象
            save_path: 保存目录
            
        Returns:
            str: 文件保存路径或 "No fulltext available"
        """
        # 检查 pdf_url 是否为空
        if not paper.pdf_url:
            return "No fulltext available"
        
        try:
            # 发送 GET 请求下载文件
            response = await client.get(paper.pdf_url, follow_redirects=True)
            response.raise_for_status()
            
            # 生成文件名：使用 paper_id 或 doi 作为文件名
            if paper.paper_id:
                filename = f"{paper.paper_id}.pdf"
            elif paper.doi:
                # DOI 中的 / 替换为 _
                safe_doi = paper.doi.replace("/", "_")
                filename = f"{safe_doi}.pdf"
            else:
                # 使用标题的前 50 个字符作为文件名
                safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title[:50])
                filename = f"{safe_title.strip()}.pdf"
            
            # 构建完整的文件路径
            file_path = os.path.join(save_path, filename)
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return file_path
            
        except httpx.HTTPStatusError as e:
            print(f"下载失败 (HTTP {e.response.status_code}): {paper.pdf_url}")
            return "No fulltext available"
        except httpx.RequestError as e:
            print(f"网络请求错误: {e}")
            return "No fulltext available"
        except Exception as e:
            print(f"下载文件失败: {e}")
            return "No fulltext available"

    async def search(self, query: str, limit: int = 5) -> List[Paper]:
        """
        根据查询关键词检索 Semantic Scholar 文献
        
        Args:
            query: 检索关键词
            limit: 最大返回数量，默认 5（API 最大支持 100）
            
        Returns:
            List[Paper]: 符合条件的 Paper 对象列表，包含完整的元信息
        """
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
        
        # 创建 httpx.AsyncClient
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            # 遍历 Paper 列表调用 _download_file
            for paper in paper_list:
                saved_path = await self._download_file(client, paper, save_path)
                # 更新 Paper.extra["saved_path"]
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
        
        # 返回更新后的 Paper 列表
        return paper_list


async def main():
    """
    使用示例：展示 search 和 download 方法的独立使用方式
    
    本示例演示了 SemanticScholarSearcher 的两个核心功能：
    1. search: 根据关键词检索 Semantic Scholar 文献
    2. download: 根据 Paper 对象中的 pdf_url 下载全文文档
    
    两个方法可以独立使用，也可以组合使用。
    """
    import asyncio
    
    # 创建检索器实例
    # 可选：传入 API Key 以获得更高的请求限制
    # searcher = SemanticScholarSearcher(api_key="your-api-key")
    searcher = SemanticScholarSearcher()
    
    print("=" * 60)
    print("Semantic Scholar 文献检索器使用示例")
    print("=" * 60)
    
    # ========== 示例 1: 单独使用 search 方法 ==========
    print("\n【示例 1】单独使用 search 方法检索文献")
    print("-" * 40)
    
    query = "deep learning natural language processing"
    limit = 3
    
    print(f"检索关键词: {query}")
    print(f"最大返回数量: {limit}")
    print()
    
    papers = await searcher.search(query, limit=limit)
    
    if papers:
        print(f"共检索到 {len(papers)} 篇论文:\n")
        for i, paper in enumerate(papers, 1):
            print(f"[{i}] {paper.title}")
            print(f"    作者: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"    DOI: {paper.doi or '无'}")
            print(f"    发布日期: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else '未知'}")
            print(f"    引用数: {paper.citations}")
            print(f"    PDF链接: {'有' if paper.pdf_url else '无'}")
            print(f"    来源: {paper.source}")
            print()
    else:
        print("未检索到相关论文")
    
    # ========== 示例 2: 单独使用 download 方法 ==========
    print("\n【示例 2】单独使用 download 方法下载文档")
    print("-" * 40)
    
    # 筛选有 PDF 链接的论文
    papers_with_pdf = [p for p in papers if p.pdf_url]
    
    if papers_with_pdf:
        print(f"有 {len(papers_with_pdf)} 篇论文提供 PDF 下载")
        print(f"保存路径: {Config.DOC_SAVE_PATH}")
        print()
        
        # 下载文档（这里只下载第一篇作为示例）
        downloaded_papers = await searcher.download(papers_with_pdf[:1])
        
        for paper in downloaded_papers:
            saved_path = paper.extra.get("saved_path", "")
            if saved_path and saved_path != "No fulltext available":
                print(f"✓ 下载成功: {paper.title[:50]}...")
                print(f"  保存位置: {saved_path}")
            else:
                print(f"✗ 下载失败: {paper.title[:50]}...")
                print(f"  状态: {saved_path}")
    else:
        print("没有可下载的 PDF 文档")
    
    # ========== 示例 3: 组合使用 search 和 download ==========
    print("\n【示例 3】组合使用 search 和 download")
    print("-" * 40)
    
    # 检索并下载一步完成
    query2 = "transformer attention mechanism"
    print(f"检索关键词: {query2}")
    
    papers2 = await searcher.search(query2, limit=2)
    
    if papers2:
        print(f"检索到 {len(papers2)} 篇论文，开始下载...")
        
        # 下载所有论文
        downloaded_papers2 = await searcher.download(papers2)
        
        print("\n下载结果:")
        for paper in downloaded_papers2:
            saved_path = paper.extra.get("saved_path", "")
            status = "✓ 成功" if saved_path and saved_path != "No fulltext available" else "✗ 无全文"
            print(f"  {status}: {paper.title[:40]}...")
    
    # ========== 示例 4: 查看论文详细信息 ==========
    print("\n【示例 4】查看论文详细元数据")
    print("-" * 40)
    
    if papers:
        paper = papers[0]
        print(f"论文标题: {paper.title}")
        print(f"Paper ID: {paper.paper_id}")
        print(f"DOI: {paper.doi or '无'}")
        print(f"作者: {', '.join(paper.authors)}")
        print(f"摘要: {paper.abstract[:200] + '...' if paper.abstract and len(paper.abstract) > 200 else paper.abstract or '无'}")
        print(f"发布日期: {paper.published_date}")
        print(f"学科分类: {', '.join(paper.categories) if paper.categories else '无'}")
        print(f"引用数: {paper.citations}")
        print(f"参考文献数: {paper.references[0] if paper.references else '0'}")
        print(f"PDF链接: {paper.pdf_url or '无'}")
        print(f"页面链接: {paper.url}")
        
        # 显示额外元数据
        if paper.extra:
            print("\n额外元数据:")
            print(f"  期刊/会议: {paper.extra.get('venue', '无')}")
            print(f"  年份: {paper.extra.get('year', '未知')}")
            print(f"  开放获取: {'是' if paper.extra.get('isOpenAccess') else '否'}")
            print(f"  影响力引用数: {paper.extra.get('influentialCitationCount', 0)}")
            print(f"  Corpus ID: {paper.extra.get('corpusId', '无')}")
    
    print("\n" + "=" * 60)
    print("示例执行完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

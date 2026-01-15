"""
OpenAlex 文献检索器

提供从 OpenAlex 数据库检索学术论文元数据并下载全文文档的功能。
遵循与 PubMedSearcher 相同的设计模式，提供独立的 search 和 download 方法。

OpenAlex 是一个完全开放的学术文献目录，包含超过 2.4 亿篇文章、书籍和论文，
提供丰富的元数据和开放获取信息。API 无需 API Key，但建议提供 email 以进入
polite pool 获得更高的请求限制。

使用示例:
    >>> import asyncio
    >>> from openalex import OpenAlexSearcher
    >>> 
    >>> async def main():
    ...     searcher = OpenAlexSearcher()
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
    - search: 根据关键词检索 OpenAlex 文献，返回 Paper 对象列表
    - download: 根据 Paper 对象中的 pdf_url 下载全文文档

Author: OpenAlex Searcher
Version: 1.0.0
"""

import os
import httpx
from typing import List, Union, Optional
from datetime import datetime

from paper import Paper
from config import Config


class OpenAlexSearcher:
    """
    OpenAlex 文献检索器类
    
    用于从 OpenAlex 数据库检索学术论文元数据并下载全文文档。
    
    Attributes:
        email: 用于 OpenAlex polite pool 的邮箱地址
        base_url: OpenAlex API 基础 URL
        headers: HTTP 请求头
    """
    
    def __init__(self, email: str = None):
        """
        初始化 OpenAlexSearcher
        
        Args:
            email: 邮箱地址，用于进入 OpenAlex polite pool 获得更高的请求限制。
                   默认使用 Config.EMAIL
        """
        self.email = email or Config.EMAIL
        self.base_url = "https://api.openalex.org"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": f"OpenAlexSearcher/1.0 (mailto:{self.email})"
        }
    
    def _convert_abstract(self, inverted_index: dict) -> str:
        """
        将 OpenAlex 的 abstract_inverted_index 转换为纯文本
        
        OpenAlex 使用倒排索引存储摘要，格式如：
        {"Despite": [0], "growing": [1], "interest": [2], "in": [3, 57, 73]}
        
        Args:
            inverted_index: 倒排索引格式的摘要
            
        Returns:
            str: 纯文本摘要，如果输入为空则返回空字符串
        """
        if not inverted_index:
            return ""
        
        # 找到最大位置索引
        max_pos = max(pos for positions in inverted_index.values() for pos in positions)
        
        # 创建词列表，初始化为空字符串
        words = [""] * (max_pos + 1)
        
        # 将每个词放到对应位置
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        
        return " ".join(words)

    async def _search_works(
        self, 
        client: httpx.AsyncClient, 
        query: str, 
        limit: int
    ) -> List[dict]:
        """
        调用 OpenAlex API 搜索文献
        
        Args:
            client: HTTP 客户端
            query: 检索关键词
            limit: 最大返回数量
            
        Returns:
            List[dict]: OpenAlex Work 对象列表
        """
        url = f"{self.base_url}/works"
        params = {
            "search": query,
            "per_page": limit,
            "mailto": self.email
        }
        
        try:
            response = await client.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return []
        except Exception as e:
            print(f"Error searching OpenAlex: {e}")
            return []


    def _extract_pdf_url(self, work: dict) -> str:
        """
        从 OpenAlex Work 对象中提取最佳 PDF 下载链接
        
        优先级：
        1. best_oa_location.pdf_url
        2. locations 中第一个有 pdf_url 的位置
        
        Args:
            work: OpenAlex Work 对象
            
        Returns:
            str: PDF 下载链接，如果没有则返回空字符串
        """
        # 优先从 best_oa_location 获取
        best_oa = work.get("best_oa_location")
        if best_oa and best_oa.get("pdf_url"):
            return best_oa["pdf_url"]
        
        # 其次遍历 locations 查找可用的 pdf_url
        locations = work.get("locations", [])
        for location in locations:
            if location and location.get("pdf_url"):
                return location["pdf_url"]
        
        return ""


    def _map_to_paper(self, work: dict) -> Paper:
        """
        将 OpenAlex Work 对象映射到 Paper 类
        
        Args:
            work: OpenAlex Work 对象（字典）
            
        Returns:
            Paper: 转换后的 Paper 对象
        """
        # 解析 paper_id
        paper_id = work.get("id", "")
        
        # 解析 title
        title = work.get("display_name", "")
        
        # 解析 authors（从 authorships 提取）
        authorships = work.get("authorships", [])
        authors = []
        for authorship in authorships:
            author = authorship.get("author", {})
            if author and author.get("display_name"):
                authors.append(author["display_name"])
        
        # 解析 abstract（调用 _convert_abstract）
        abstract_inverted_index = work.get("abstract_inverted_index")
        abstract = self._convert_abstract(abstract_inverted_index)
        
        # 解析 doi（去除 https://doi.org/ 前缀）
        doi_raw = work.get("doi", "") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""
        
        # 解析 published_date
        pub_date_str = work.get("publication_date", "")
        published_date = None
        if pub_date_str:
            try:
                published_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
            except ValueError:
                try:
                    # 尝试只有年份的情况
                    published_date = datetime.strptime(pub_date_str[:4], "%Y")
                except ValueError:
                    published_date = None
        
        # 解析 pdf_url（调用 _extract_pdf_url）
        pdf_url = self._extract_pdf_url(work)
        
        # 解析 url（primary_location.landing_page_url 或 OpenAlex 页面）
        primary_location = work.get("primary_location", {}) or {}
        url = primary_location.get("landing_page_url", "") or work.get("id", "")
        
        # 解析 updated_date
        updated_date_str = work.get("updated_date", "")
        updated_date = None
        if updated_date_str:
            try:
                # OpenAlex 的 updated_date 格式为 ISO 8601
                updated_date = datetime.fromisoformat(updated_date_str.replace("Z", "+00:00"))
            except ValueError:
                updated_date = None
        
        # 解析 categories（从 topics 或 concepts 提取）
        categories = []
        topics = work.get("topics", [])
        if topics:
            for topic in topics:
                if topic and topic.get("display_name"):
                    categories.append(topic["display_name"])
        else:
            # 如果没有 topics，尝试从 concepts 获取
            concepts = work.get("concepts", [])
            for concept in concepts:
                if concept and concept.get("display_name"):
                    categories.append(concept["display_name"])
        
        # 解析 keywords
        keywords = []
        keywords_list = work.get("keywords", [])
        for kw in keywords_list:
            if kw and kw.get("display_name"):
                keywords.append(kw["display_name"])
        
        # 解析 citations
        citations = work.get("cited_by_count", 0) or 0
        
        # 解析 references
        references = work.get("referenced_works", []) or []
        
        # 设置 extra（包含 type、language、is_oa 等）
        extra = {
            "type": work.get("type", ""),
            "language": work.get("language", ""),
            "is_oa": work.get("open_access", {}).get("is_oa", False) if work.get("open_access") else False,
            "oa_status": work.get("open_access", {}).get("oa_status", "") if work.get("open_access") else "",
            "cited_by_api_url": work.get("cited_by_api_url", ""),
            "publication_year": work.get("publication_year"),
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
            source="openalex",
            updated_date=updated_date,
            categories=categories,
            keywords=keywords,
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
        # 如果 pdf_url 为空，返回无全文标记
        if not paper.pdf_url:
            return "No fulltext available"
        
        try:
            # 发送 GET 请求下载文件
            response = await client.get(
                paper.pdf_url, 
                headers=self.headers,
                follow_redirects=True
            )
            response.raise_for_status()
            
            # 生成文件名：使用 paper_id 或 DOI 作为文件名
            if paper.paper_id:
                # OpenAlex ID 格式为 https://openalex.org/W2741809807，提取最后部分
                file_id = paper.paper_id.split("/")[-1] if "/" in paper.paper_id else paper.paper_id
            elif paper.doi:
                # DOI 中的 / 替换为 _
                file_id = paper.doi.replace("/", "_")
            else:
                # 使用时间戳作为备选
                file_id = f"paper_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            file_name = f"{file_id}.pdf"
            file_path = os.path.join(save_path, file_name)
            
            # 保存文件
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return file_path
            
        except httpx.HTTPError as e:
            print(f"HTTP error downloading {paper.pdf_url}: {e}")
            return "No fulltext available"
        except Exception as e:
            print(f"Error downloading file: {e}")
            return "No fulltext available"

    async def search(self, query: str, limit: int = 5) -> List[Paper]:
        """
        根据查询关键词检索 OpenAlex 文献
        
        Args:
            query: 检索关键词，支持 OpenAlex 搜索语法
            limit: 最大返回数量，默认 5
            
        Returns:
            List[Paper]: 符合条件的 Paper 对象列表，包含完整的元信息
        """
        papers = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 调用 _search_works 获取 Work 列表
            works = await self._search_works(client, query, limit)
            
            # 遍历调用 _map_to_paper 转换为 Paper 列表
            for work in works:
                try:
                    paper = self._map_to_paper(work)
                    papers.append(paper)
                except Exception as e:
                    print(f"Error mapping work to paper: {e}")
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
        
        # 创建 httpx.AsyncClient 并下载文件
        async with httpx.AsyncClient(timeout=60.0) as client:
            for paper in paper_list:
                # 调用 _download_file 下载文件
                saved_path = await self._download_file(client, paper, save_path)
                
                # 更新 Paper.extra["saved_path"]
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
        
        # 返回更新后的 Paper 列表
        return paper_list


async def main():
    """
    OpenAlexSearcher 使用示例
    
    展示 search 和 download 方法的独立使用方式：
    1. 单独使用 search 方法检索论文元信息
    2. 单独使用 download 方法下载已有 Paper 对象的全文
    3. 组合使用 search + download 完成完整的检索下载流程
    """
    # 创建检索器实例
    # 可以传入自定义 email 以进入 OpenAlex polite pool
    searcher = OpenAlexSearcher()
    
    print("=" * 60)
    print("OpenAlex 文献检索器使用示例")
    print("=" * 60)
    
    # ========== 示例 1: 单独使用 search 方法 ==========
    print("\n【示例 1】单独使用 search 方法检索论文")
    print("-" * 40)
    
    query = "deep learning neural network"
    papers = await searcher.search(query, limit=3)
    
    print(f"检索关键词: {query}")
    print(f"检索到 {len(papers)} 篇论文:\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"[{i}] {paper.title}")
        print(f"    作者: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"    DOI: {paper.doi or 'N/A'}")
        print(f"    发布日期: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
        print(f"    引用数: {paper.citations}")
        print(f"    PDF链接: {'有' if paper.pdf_url else '无'}")
        print(f"    摘要: {paper.abstract[:100]}..." if paper.abstract else "    摘要: N/A")
        print()
    
    # ========== 示例 2: 单独使用 download 方法 ==========
    print("\n【示例 2】单独使用 download 方法下载论文")
    print("-" * 40)
    
    # 筛选有 PDF 链接的论文进行下载
    papers_with_pdf = [p for p in papers if p.pdf_url]
    
    if papers_with_pdf:
        print(f"找到 {len(papers_with_pdf)} 篇有 PDF 链接的论文，开始下载...")
        
        # 下载论文（使用默认保存路径 Config.DOC_SAVE_PATH）
        downloaded_papers = await searcher.download(papers_with_pdf)
        
        for paper in downloaded_papers:
            saved_path = paper.extra.get("saved_path", "")
            if saved_path and saved_path != "No fulltext available":
                print(f"✓ 下载成功: {paper.title[:50]}...")
                print(f"  保存路径: {saved_path}")
            else:
                print(f"✗ 下载失败: {paper.title[:50]}...")
    else:
        print("没有找到有 PDF 链接的论文")
    
    # ========== 示例 3: 组合使用 search + download ==========
    print("\n【示例 3】组合使用 search + download 完成完整流程")
    print("-" * 40)
    
    # 检索特定主题的论文
    query2 = "transformer attention mechanism"
    print(f"检索关键词: {query2}")
    
    papers2 = await searcher.search(query2, limit=2)
    print(f"检索到 {len(papers2)} 篇论文")
    
    # 直接下载所有检索到的论文
    if papers2:
        # 可以指定自定义保存路径
        custom_save_path = "./downloads"
        downloaded = await searcher.download(papers2, save_path=custom_save_path)
        
        print(f"\n下载结果 (保存到 {custom_save_path}):")
        for paper in downloaded:
            saved_path = paper.extra.get("saved_path", "")
            status = "✓ 成功" if saved_path and saved_path != "No fulltext available" else "✗ 无全文"
            print(f"  {status}: {paper.title[:40]}...")
    
    # ========== 示例 4: 访问 Paper 对象的详细信息 ==========
    print("\n【示例 4】访问 Paper 对象的详细信息")
    print("-" * 40)
    
    if papers:
        paper = papers[0]
        print(f"论文详细信息:")
        print(f"  paper_id: {paper.paper_id}")
        print(f"  title: {paper.title}")
        print(f"  authors: {paper.authors}")
        print(f"  doi: {paper.doi}")
        print(f"  published_date: {paper.published_date}")
        print(f"  url: {paper.url}")
        print(f"  source: {paper.source}")
        print(f"  categories: {paper.categories[:3] if paper.categories else []}")
        print(f"  keywords: {paper.keywords[:3] if paper.keywords else []}")
        print(f"  citations: {paper.citations}")
        print(f"  references count: {len(paper.references) if paper.references else 0}")
        print(f"  extra: {paper.extra}")
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

"""
OpenAlex 文献检索器 (修改版 v3)

提供从 OpenAlex 数据库检索学术论文元数据并下载全文文档的功能。
遵循与 PubMedSearcher 相同的设计模式，提供独立的 search 和 download 方法。

OpenAlex 是一个完全开放的学术文献目录，包含超过 2.4 亿篇文章、书籍和论文，
提供丰富的元数据和开放获取信息。API 无需 API Key，但建议提供 email 以进入
polite pool 获得更高的请求限制。

【修改说明 v4】
- 仅使用 Unpaywall 进行 PDF 下载
- 修复下载过滤逻辑：从 pdf_url 改为 doi（Unpaywall 只需 DOI）
- 移除了 arXiv、PubMed Central 等其他下载源
- 移除了 OpenAlex pdf_url 作为保底下载选项
- 增加了并发下载功能，最大并发数默认为 3

"""

import os
import httpx
from typing import List, Union, Optional
from datetime import datetime
import asyncio
from .paper import Paper
from core.config import Config


class OpenAlexSearcher:
    """
    OpenAlex 文献检索器类

    用于从 OpenAlex 数据库检索学术论文元数据并下载全文文档。

    Attributes:
        email: 用于 OpenAlex polite pool 的邮箱地址
        base_url: OpenAlex API 基础 URL
        headers: HTTP 请求头
    """

    def __init__(self, email: str = None, size: int = None):
        """
        初始化 OpenAlexSearcher

        Args:
            email: 邮箱地址，用于进入 OpenAlex polite pool 获得更高的请求限制。
                   默认使用 Config.EMAIL
            size: 检索返回数量，默认使用 Config.SEARCH_SIZE
        """
        self.email = email or Config.EMAIL
        self.size = size or Config.SEARCH_SIZE
        self.base_url = "https://api.openalex.org"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": f"OpenAlexSearcher/2.0 (mailto:{self.email})"
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
        limit: int,
        sort_by: str = "relevance_score:desc",
        filter_params: dict = None
    ) -> List[dict]:
        """
        调用 OpenAlex API 搜索文献

        Args:
            client: HTTP 客户端
            query: 检索关键词
            limit: 最大返回数量
            sort_by: 排序方式，默认按相关性降序
            filter_params: 额外的过滤参数

        Returns:
            List[dict]: OpenAlex Work 对象列表
        """
        url = f"{self.base_url}/works"
        params = {
            "search": query,
            "per_page": limit,
            "mailto": self.email,
            "sort": sort_by,
        }

        # 添加额外过滤参数
        if filter_params:
            params.update(filter_params)

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


    def _get_browser_headers(self, referer: str = None) -> dict:
        """
        获取模拟浏览器的请求头，降低被反爬检测的概率

        Args:
            referer: 可选的 Referer 头，某些网站需要
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _is_valid_pdf(self, content: bytes) -> bool:
        """
        检查下载的内容是否为有效的 PDF 文件

        Args:
            content: 下载的文件内容

        Returns:
            bool: 是否为有效 PDF
        """
        if len(content) < 4:
            return False
        return content[:4] == b'%PDF'

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

    def _get_all_pdf_urls(self, paper: Paper) -> List[str]:
        """
        获取论文的所有可能的 PDF 下载链接（不包含 OpenAlex pdf_url）

        【注意 v3】此方法已不再被 _download_file 使用，仅保留用于向后兼容
        当前下载流程仅使用 Unpaywall

        Args:
            paper: Paper 对象

        Returns:
            List[str]: PDF 链接列表（稳定的第三方源）
        """
        urls = []

        # 如果有 DOI，构建常见的直接下载链接
        if paper.doi:
            # arXiv（如果是 arXiv 论文）
            if "arxiv" in paper.doi.lower():
                arxiv_id = paper.doi.split("/")[-1]
                urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

            # PubMed Central（需要 PMC ID，不是 DOI，暂时注释）
            # urls.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{paper.doi}/pdf/")

        # 从 extra 中获取其他可能的链接
        if paper.extra:
            # OpenAlex 可能在 extra 中存储了其他位置信息
            oa_status = paper.extra.get("oa_status", "")
            if oa_status == "gold" or oa_status == "green":
                # 开放获取论文，原链接应该可用
                pass

        return urls

    async def _try_download_url(
        self,
        client: httpx.AsyncClient,
        url: str,
        file_path: str,
        referer: str = None
    ) -> bool:
        """
        尝试从单个 URL 下载文件

        Args:
            client: HTTP 客户端
            url: 下载链接
            file_path: 保存路径
            referer: Referer 头

        Returns:
            bool: 是否下载成功
        """
        try:
            headers = self._get_browser_headers(referer)

            response = await client.get(
                url,
                headers=headers,
                follow_redirects=True,
                timeout=30.0
            )

            if response.status_code != 200:
                return False

            content = response.content

            # 检查文件大小（PDF 通常大于 10KB）
            if len(content) < 10000:
                return False

            # 验证是否为有效 PDF
            if not self._is_valid_pdf(content):
                return False

            # 保存文件
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
        下载单个文件，仅使用 Unpaywall

        【修改 v3】仅使用 Unpaywall 获取合法 OA 版本的 PDF

        Args:
            client: HTTP 客户端
            paper: Paper 对象
            save_path: 保存目录
            max_retries: 最大重试次数（默认减少到 2 次以加快失败响应）

        Returns:
            str: 文件保存路径或 "No fulltext available"
        """
        import random
        from urllib.parse import urlparse

        # 首先检查目录中是否已存在该 DOI 的文件
        if paper.doi:
            existing_file = self._find_existing_paper_by_doi(save_path, paper.doi)
            if existing_file:
                print(f"文件已存在（基于 DOI）: {existing_file}")
                return existing_file

        # 生成文件名（优先使用 DOI）
        if paper.doi:
            file_id = paper.doi.replace("/", "_")
        elif paper.paper_id:
            file_id = paper.paper_id.split("/")[-1] if "/" in paper.paper_id else paper.paper_id
        else:
            file_id = f"paper_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        file_name = f"{file_id}.pdf"
        file_path = os.path.join(save_path, file_name)

        # 如果文件已存在，直接返回
        if os.path.exists(file_path):
            return file_path

        # 如果没有 DOI，无法使用 Unpaywall，直接返回
        if not paper.doi:
            return "No fulltext available"

        # 仅从 Unpaywall 获取 PDF 链接（有超时控制）
        try:
            pdf_url = await self._get_alternative_pdf_url(client, paper.doi)
        except Exception as e:
            print(f"获取 Unpaywall 链接异常: {e}")
            return "No fulltext available"

        # 如果没有可用的下载链接，返回无全文标记
        if not pdf_url:
            return "No fulltext available"

        # 从 URL 提取 referer
        parsed = urlparse(pdf_url)
        referer = f"{parsed.scheme}://{parsed.netloc}/"

        for attempt in range(max_retries):
            # 添加随机延迟（减少延迟时间）
            if attempt > 0:
                delay = random.uniform(0.5, 1.5) * attempt
                await asyncio.sleep(delay)

            if await self._try_download_url(client, pdf_url, file_path, referer):
                return file_path

        return "No fulltext available"

    async def _get_alternative_pdf_url(self, client: httpx.AsyncClient, doi: str) -> Optional[str]:
        """
        通过 Unpaywall API 获取备用 PDF 链接

        Unpaywall 是一个免费的开放获取查找服务，可以找到论文的合法免费版本

        Args:
            client: HTTP 客户端
            doi: 论文的 DOI

        Returns:
            Optional[str]: 备用 PDF 链接，如果没有则返回 None
        """
        if not doi:
            return None

        try:
            # Unpaywall API
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": self.email}

            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code != 200:
                return None

            data = response.json()

            # 优先获取 best_oa_location
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("url_for_pdf"):
                return best_oa["url_for_pdf"]

            # 遍历所有 oa_locations
            oa_locations = data.get("oa_locations", [])
            for loc in oa_locations:
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]

            return None

        except Exception as e:
            print(f"获取备用链接失败: {e}")
            return None

    async def search(
        self,
        query: str,
        limit: int = None,
        sort_by: str = "relevance_score:desc",
        from_year: int = None,
        to_year: int = None,
        open_access_only: bool = False,
        has_pdf: bool = False
    ) -> List[Paper]:
        """
        根据查询关键词检索 OpenAlex 文献

        Args:
            query: 检索关键词，支持 OpenAlex 搜索语法
            limit: 最大返回数量，默认使用 self.size
            sort_by: 排序方式，可选值：
                - "relevance_score:desc" (默认，按相关性降序)
                - "cited_by_count:desc" (按引用数降序)
                - "publication_date:desc" (按发布日期降序)
            from_year: 起始年份，如 2020
            to_year: 结束年份，如 2024
            open_access_only: 是否只返回开放获取的文献
            has_pdf: 是否只返回有 PDF 链接的文献

        Returns:
            List[Paper]: 符合条件的 Paper 对象列表，包含完整的元信息
        """
        if limit is None:
            limit = self.size

        # 构建过滤参数
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if open_access_only:
            filters.append("is_oa:true")
        if has_pdf:
            filters.append("has_pdf_url:true")

        filter_params = {}
        if filters:
            filter_params["filter"] = ",".join(filters)

        papers = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 调用 _search_works 获取 Work 列表
            works = await self._search_works(client, query, limit, sort_by, filter_params)

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
        save_path: str = None,
        max_concurrent: int = 3
    ) -> List[Paper]:
        """
        根据 Paper 对象中的 DOI 使用 Unpaywall 并发下载文档

        Args:
            papers: 单个 Paper 对象或 Paper 列表
            save_path: 保存路径，默认使用 Config.DOC_SAVE_PATH
            max_concurrent: 最大并发下载数，默认 3

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

        # 统计有 DOI 的论文数量（Unpaywall 只需 DOI 即可尝试下载）
        papers_with_doi = [p for p in paper_list if p.doi]
        print(f"共 {len(papers_with_doi)} 篇论文待下载（最大并发数: {max_concurrent}）")

        # 创建并发限制信号量
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(paper: Paper):
            """带并发限制的下载函数"""
            async with semaphore:
                saved_path = await self._download_file(client, paper, save_path)
                # 更新 Paper.extra["saved_path"]
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
                return saved_path

        # 创建 httpx.AsyncClient 并并发下载文件
        success_count = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 创建并发任务
            tasks = [download_with_limit(paper) for paper in paper_list]

            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"下载出错: {result}")
                    continue

                saved_path = result

                # 打印成功下载的文件路径
                if saved_path and saved_path != "No fulltext available":
                    print(f"已保存: {saved_path}")
                    success_count += 1

        print(f"下载完成: {success_count}/{len(papers_with_doi)}")

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
    print("OpenAlex 文献检索器使用示例 (v3 - 使用 DOI 过滤)")
    print("=" * 60)

    # ========== 示例 1: 单独使用 search 方法 ==========
    print("\n【示例 1】单独使用 search 方法检索论文")
    print("-" * 40)

    query = "deep learning neural network"
    # query = "强化学习"
    # query = "神经网络 图像识别"
    papers = await searcher.search(query)

    print(f"检索关键词: {query}")
    print(f"检索到 {len(papers)} 篇论文:\n")

    for i, paper in enumerate(papers, 1):
        print(f"[{i}] {paper.title}")
        print(f"    作者: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"    DOI: {paper.doi or 'N/A'}")
        print(f"    发布日期: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
        print(f"    引用数: {paper.citations}")
        print(f"    PDF链接: {'有' if paper.pdf_url else '无'}")
        print(f"    摘要: {paper.abstract}..." if paper.abstract else "    摘要: N/A")
        print()

    # ========== 示例 2: 单独使用 download 方法 ==========
    print("\n【示例 2】单独使用 download 方法下载论文")
    print("-" * 40)

    # 筛选有 DOI 的论文进行下载（Unpaywall 只需 DOI 即可）
    papers_with_doi = [p for p in papers if p.doi]

    if papers_with_doi:
        # 下载论文（使用默认保存路径 Config.DOC_SAVE_PATH）
        downloaded_papers = await searcher.download(papers_with_doi)
    else:
        print("没有找到有 DOI 的论文")



if __name__ == "__main__":

    asyncio.run(main())

"""
arXiv 和 PubMed Central 论文检索器

提供从 arXiv 和 PubMed Central 检索学术论文元数据并下载全文文档的功能。
遵循与 OpenAlexSearcher 相同的设计模式，提供独立的 search 和 download 方法。

【类说明】
- ArxivSearcher: arXiv 预印本论文检索器
- PMCSearcher: PubMed Central 生物医学文献检索器
"""

import os
import httpx
import re
import xml.etree.ElementTree as ET
from typing import List, Union, Optional
from datetime import datetime
import asyncio
from .paper import Paper
from core.config import Config


class ArxivSearcher:
    """
    arXiv 预印本论文检索器类

    用于从 arXiv 数据库检索学术论文元数据并下载全文文档。

    Attributes:
        base_url: arXiv API 基础 URL
        max_results: 默认检索返回数量
    """

    def __init__(self, max_results: int = None):
        """
        初始化 ArxivSearcher

        Args:
            max_results: 检索返回数量，默认使用 Config.SEARCH_SIZE
        """
        self.base_url = "https://export.arxiv.org/api/query"
        self.max_results = max_results or Config.SEARCH_SIZE

    def _parse_arxiv_date(self, date_str: str) -> Optional[datetime]:
        """
        解析 arXiv 日期格式

        arXiv 日期格式: 2023-01-15T10:30:00Z

        Args:
            date_str: arXiv 日期字符串

        Returns:
            Optional[datetime]: 解析后的日期
        """
        if not date_str:
            return None
        try:
            # 去除时缀并解析
            date_str = date_str.replace("Z", "").replace("T", " ")
            return datetime.strptime(date_str.split()[0], "%Y-%m-%d")
        except ValueError:
            return None

    def _extract_arxiv_id(self, id_url: str) -> str:
        """
        从 arXiv ID URL 中提取 arXiv ID

        例如: http://arxiv.org/abs/2301.12345 -> 2301.12345

        Args:
            id_url: arXiv ID URL

        Returns:
            str: arXiv ID
        """
        # 从 URL 中提取 ID
        match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', id_url)
        if match:
            return match.group(1)

        # 或者直接返回最后部分
        parts = id_url.split("/")[-1]
        # 处理新旧 arXiv ID 格式
        # 旧格式: cs/0001001 -> 新格式: 0001.0001
        return parts

    async def _search_papers(
        self,
        client: httpx.AsyncClient,
        query: str,
        limit: int
    ) -> List[dict]:
        """
        调用 arXiv API 搜索文献

        Args:
            client: HTTP 客户端
            query: 检索关键词
            limit: 最大返回数量

        Returns:
            List[dict]: arXiv Entry 对象列表
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
        }

        try:
            response = await client.get(self.base_url, params=params, timeout=30.0)
            response.raise_for_status()

            # 解析 XML 响应
            root = ET.fromstring(response.content)

            # arXiv API 使用 Atom 命名空间
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }

            entries = []
            for entry in root.findall('atom:entry', namespaces):
                entry_data = {}

                # 基本信息
                entry_data['id'] = entry.find('atom:id', namespaces).text
                entry_data['title'] = entry.find('atom:title', namespaces).text.strip()
                entry_data['summary'] = entry.find('atom:summary', namespaces).text.strip()
                entry_data['published'] = entry.find('atom:published', namespaces).text
                entry_data['updated'] = entry.find('atom:updated', namespaces).text

                # 作者
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name = author.find('atom:name', namespaces)
                    if name is not None:
                        authors.append(name.text)
                entry_data['authors'] = authors

                # 分类 (categories)
                categories = []
                for category in entry.findall('atom:category', namespaces):
                    term = category.get('term')
                    if term:
                        categories.append(term)
                entry_data['categories'] = categories

                # arXiv ID
                arxiv_id = self._extract_arxiv_id(entry_data['id'])
                entry_data['arxiv_id'] = arxiv_id

                # PDF URL
                entry_data['pdf_url'] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                # 主页 URL
                entry_data['url'] = entry_data['id']

                # DOI (如果有)
                primary_category = entry.find('arxiv:primary_category', namespaces)
                if primary_category is not None:
                    entry_data['primary_category'] = primary_category.get('term', '')

                entries.append(entry_data)

            return entries

        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return []
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []

    def _map_to_paper(self, entry: dict) -> Paper:
        """
        将 arXiv Entry 对象映射到 Paper 类

        Args:
            entry: arXiv Entry 对象（字典）

        Returns:
            Paper: 转换后的 Paper 对象
        """
        # 解析 paper_id (使用 arXiv ID)
        paper_id = entry.get('arxiv_id', entry.get('id', ''))

        # 解析 title
        title = entry.get('title', '')

        # 解析 authors
        authors = entry.get('authors', [])

        # 解析 abstract
        abstract = entry.get('summary', '')

        # arXiv 论文通常没有 DOI
        doi = ''

        # 解析 published_date
        published_date = self._parse_arxiv_date(entry.get('published', ''))

        # 解析 updated_date
        updated_date = self._parse_arxiv_date(entry.get('updated', ''))

        # 解析 pdf_url
        pdf_url = entry.get('pdf_url', '')

        # 解析 url
        url = entry.get('url', '')

        # 解析 categories
        categories = entry.get('categories', [])

        # keywords (arXiv 使用 categories 作为关键词)
        keywords = categories[:5] if categories else []

        # 设置 extra
        extra = {
            'primary_category': entry.get('primary_category', ''),
            'arxiv_id': entry.get('arxiv_id', ''),
            'type': 'preprint',
            'language': 'en',
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
            source='arxiv',
            updated_date=updated_date,
            categories=categories,
            keywords=keywords,
            citations=0,
            references=[],
            extra=extra
        )

    async def search(
        self,
        query: str,
        limit: int = None
    ) -> List[Paper]:
        """
        根据查询关键词检索 arXiv 文献

        Args:
            query: 检索关键词
            limit: 最大返回数量，默认使用 self.max_results

        Returns:
            List[Paper]: 符合条件的 Paper 对象列表
        """
        if limit is None:
            limit = self.max_results

        papers = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            entries = await self._search_papers(client, query, limit)

            for entry in entries:
                try:
                    paper = self._map_to_paper(entry)
                    papers.append(paper)
                except Exception as e:
                    print(f"Error mapping entry to paper: {e}")
                    continue

        return papers

    async def download(
        self,
        papers: Union[Paper, List[Paper]],
        save_path: str = None,
        max_concurrent: int = 3
    ) -> List[Paper]:
        """
        根据 Paper 对象中的 pdf_url 并发下载文档

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

        # 确保保存目录存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 统计有 PDF 链接的论文数量
        papers_with_pdf = [p for p in paper_list if p.pdf_url]
        print(f"共 {len(papers_with_pdf)} 篇论文待下载（最大并发数: {max_concurrent}）")

        # 创建并发限制信号量
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(paper: Paper):
            """带并发限制的下载函数"""
            async with semaphore:
                saved_path = await self._download_file(client, paper, save_path)
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
                return saved_path

        # 创建 httpx.AsyncClient 并并发下载文件
        success_count = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [download_with_limit(paper) for paper in paper_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"下载出错: {result}")
                    continue

                saved_path = result
                if saved_path and saved_path != "No fulltext available":
                    print(f"已保存: {saved_path}")
                    success_count += 1

        print(f"下载完成: {success_count}/{len(papers_with_pdf)}")
        return paper_list

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        paper: Paper,
        save_path: str
    ) -> str:
        """下载单个文件"""
        if not paper.pdf_url:
            return "No fulltext available"

        # 使用 arXiv ID 作为文件名
        if paper.extra and paper.extra.get('arxiv_id'):
            file_id = paper.extra['arxiv_id']
        else:
            file_id = f"arxiv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        file_name = f"{file_id}.pdf"
        file_path = os.path.join(save_path, file_name)

        # 如果文件已存在，直接返回
        if os.path.exists(file_path):
            return file_path

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = await client.get(
                paper.pdf_url,
                headers=headers,
                follow_redirects=True,
                timeout=60.0
            )

            if response.status_code == 200:
                content = response.content

                # 检查是否为有效 PDF
                if len(content) > 10000 and content[:4] == b'%PDF':
                    with open(file_path, "wb") as f:
                        f.write(content)
                    print(f"[arXiv] 完成了 (arXiv ID: {paper.extra.get('arxiv_id', 'N/A')}) 的下载")
                    return file_path

        except Exception as e:
            print(f"下载失败: {e}")

        return "No fulltext available"


class PMCSearcher:
    """
    PubMed Central 生物医学文献检索器类

    用于从 PubMed Central 数据库检索学术论文元数据并下载全文文档。

    Attributes:
        base_url: PMC API 基础 URL
        max_results: 默认检索返回数量
        email: 用于 NCBI 的邮箱（可选）
    """

    def __init__(self, max_results: int = None, email: str = None):
        """
        初始化 PMCSearcher

        Args:
            max_results: 检索返回数量，默认使用 Config.SEARCH_SIZE
            email: 邮箱地址，用于 NCBI API（推荐）
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.max_results = max_results or Config.SEARCH_SIZE
        self.email = email or Config.EMAIL
        self.tool = "searcher.py"

    def _parse_pmc_date(self, date_str: str) -> Optional[datetime]:
        """
        解析 PMC 日期格式

        PMC 日期格式可能为: 2023 Jan 15, 2023-01-15, 或其他格式

        Args:
            date_str: PMC 日期字符串

        Returns:
            Optional[datetime]: 解析后的日期
        """
        if not date_str:
            return None

        # 尝试多种日期格式
        date_formats = [
            "%Y %b %d",  # 2023 Jan 15
            "%Y %b",     # 2023 Jan
            "%Y-%m-%d",  # 2023-01-15
            "%Y",        # 2023
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

    async def _search_ids(
        self,
        client: httpx.AsyncClient,
        query: str,
        limit: int
    ) -> List[str]:
        """
        第一步：搜索获取 PMCID 列表

        使用 PMC ESearch API 获取符合条件的 PMCID

        Args:
            client: HTTP 客户端
            query: 检索关键词
            limit: 最大返回数量

        Returns:
            List[str]: PMCID 列表
        """
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": query,
            "retmax": limit,
            "retmode": "xml",
            "tool": self.tool,
            "email": self.email,
        }

        try:
            response = await cl
            ient.get(url, params=params, timeout=30.0)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            # 提取 PMCID 列表
            pmcids = []
            for id_elem in root.findall('.//Id'):
                if id_elem.text:
                    pmcids.append(f"PMC{id_elem.text}")

            return pmcids

        except Exception as e:
            print(f"Error searching PMCID: {e}")
            return []

    async def _fetch_summaries(
        self,
        client: httpx.AsyncClient,
        pmcids: List[str]
    ) -> List[dict]:
        """
        第二步：获取论文详细信息

        使用 PMC ESummary API 获取论文的详细元信息

        Args:
            client: HTTP 客户端
            pmcids: PMCID 列表

        Returns:
            List[dict]: 论文详细信息列表
        """
        if not pmcids:
            return []

        url = f"{self.base_url}/esummary.fcgi"
        params = {
            "db": "pmc",
            "id": ",".join(pmcids),
            "retmode": "xml",
            "tool": self.tool,
            "email": self.email,
        }

        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()

            root = ET.fromstring(response.content)

            papers = []
            for doc in root.findall('.//DocSum'):
                paper_data = {}

                # 提取 PMCID
                pmcid_elem = doc.find('Id')
                if pmcid_elem is not None and pmcid_elem.text:
                    paper_data['pmcid'] = f"PMC{pmcid_elem.text}"

                # 提取标题
                title_elem = doc.find('.//Item[@Name="Title"]')
                if title_elem is not None:
                    paper_data['title'] = title_elem.text

                # 提取作者
                authors = []
                for author in doc.findall('.//Item[@Name="AuthorList"]/Item[@Name="Author"]'):
                    author_name = author.find('Item[@Name="Name"]')
                    if author_name is not None:
                        authors.append(author_name.text)
                paper_data['authors'] = authors

                # 提取发布日期
                pub_date_elem = doc.find('.//Item[@Name="PubDate"]')
                if pub_date_elem is not None:
                    paper_data['pub_date'] = pub_date_elem.text

                # 提取期刊信息
                journal_elem = doc.find('.//Item[@Name="Source"]')
                if journal_elem is not None:
                    paper_data['journal'] = journal_elem.text

                # 提取 DOI
                doi_elem = doc.find('.//Item[@Name="DOI"]')
                if doi_elem is not None and doi_elem.text:
                    paper_data['doi'] = doi_elem.text
                else:
                    paper_data['doi'] = ''

                # 构建 URL 和 PDF URL
                pmcid = paper_data.get('pmcid', '')
                if pmcid:
                    # 提取数字部分
                    pmcid_num = pmcid.replace('PMC', '')
                    paper_data['url'] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    paper_data['pdf_url'] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

                papers.append(paper_data)

            return papers

        except Exception as e:
            print(f"Error fetching summaries: {e}")
            return []

    def _map_to_paper(self, entry: dict) -> Paper:
        """
        将 PMC 论文信息映射到 Paper 类

        Args:
            entry: PMC 论文信息（字典）

        Returns:
            Paper: 转换后的 Paper 对象
        """
        # 解析 paper_id (使用 PMCID)
        paper_id = entry.get('pmcid', '')

        # 解析 title
        title = entry.get('title', '')

        # 解析 authors
        authors = entry.get('authors', [])

        # PMC ESummary 不提供摘要，需要额外获取（这里留空）
        abstract = ''

        # 解析 doi
        doi = entry.get('doi', '')

        # 解析 published_date
        published_date = self._parse_pmc_date(entry.get('pub_date', ''))

        # 解析 pdf_url
        pdf_url = entry.get('pdf_url', '')

        # 解析 url
        url = entry.get('url', '')

        # 解析 categories (使用期刊名)
        journal = entry.get('journal', '')
        categories = [journal] if journal else []

        # keywords
        keywords = []

        # 设置 extra
        extra = {
            'pmcid': entry.get('pmcid', ''),
            'journal': journal,
            'type': 'article',
            'language': 'en',
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
            source='pmc',
            updated_date=None,
            categories=categories,
            keywords=keywords,
            citations=0,
            references=[],
            extra=extra
        )

    async def search(
        self,
        query: str,
        limit: int = None
    ) -> List[Paper]:
        """
        根据查询关键词检索 PMC 文献

        Args:
            query: 检索关键词
            limit: 最大返回数量，默认使用 self.max_results

        Returns:
            List[Paper]: 符合条件的 Paper 对象列表
        """
        if limit is None:
            limit = self.max_results

        papers = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 第一步：搜索获取 PMCID 列表
            pmcids = await self._search_ids(client, query, limit)

            # 第二步：获取详细信息
            entries = await self._fetch_summaries(client, pmcids)

            for entry in entries:
                try:
                    paper = self._map_to_paper(entry)
                    papers.append(paper)
                except Exception as e:
                    print(f"Error mapping entry to paper: {e}")
                    continue

        return papers

    async def download(
        self,
        papers: Union[Paper, List[Paper]],
        save_path: str = None,
        max_concurrent: int = 3
    ) -> List[Paper]:
        """
        根据 Paper 对象中的 pdf_url 并发下载文档

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

        # 确保保存目录存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 统计有 PDF 链接的论文数量
        papers_with_pdf = [p for p in paper_list if p.pdf_url]
        print(f"共 {len(papers_with_pdf)} 篇论文待下载（最大并发数: {max_concurrent}）")

        # 创建并发限制信号量
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(paper: Paper):
            """带并发限制的下载函数"""
            async with semaphore:
                saved_path = await self._download_file(client, paper, save_path)
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
                return saved_path

        # 创建 httpx.AsyncClient 并并发下载文件
        success_count = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [download_with_limit(paper) for paper in paper_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"下载出错: {result}")
                    continue

                saved_path = result
                if saved_path and saved_path != "No fulltext available":
                    print(f"已保存: {saved_path}")
                    success_count += 1

        print(f"下载完成: {success_count}/{len(papers_with_pdf)}")
        return paper_list

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        paper: Paper,
        save_path: str
    ) -> str:
        """下载单个文件"""
        if not paper.pdf_url:
            return "No fulltext available"

        # 使用 PMCID 作为文件名
        if paper.extra and paper.extra.get('pmcid'):
            file_id = paper.extra['pmcid']
        else:
            file_id = f"pmc_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        file_name = f"{file_id}.pdf"
        file_path = os.path.join(save_path, file_name)

        # 如果文件已存在，直接返回
        if os.path.exists(file_path):
            return file_path

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = await client.get(
                paper.pdf_url,
                headers=headers,
                follow_redirects=True,
                timeout=60.0
            )

            if response.status_code == 200:
                content = response.content

                # 检查是否为有效 PDF
                if len(content) > 10000 and content[:4] == b'%PDF':
                    with open(file_path, "wb") as f:
                        f.write(content)
                    print(f"[PMC] 完成了 (PMCID: {paper.extra.get('pmcid', 'N/A')}) 的下载")
                    return file_path

        except Exception as e:
            print(f"下载失败: {e}")

        return "No fulltext available"


# 测试代码
async def main():
    """测试 ArxivSearcher 和 PMCSearcher"""

    print("=" * 60)
    print("arXiv 和 PMC 检索器测试")
    print("=" * 60)

    # 测试 ArxivSearcher
    print("\n【测试 ArxivSearcher】")
    print("-" * 40)
    arxiv_searcher = ArxivSearcher()
    query = "deep learning"
    print(f"检索关键词: {query}")

    arxiv_papers = await arxiv_searcher.search(query, limit=3)
    print(f"检索到 {len(arxiv_papers)} 篇 arXiv 论文:\n")

    for i, paper in enumerate(arxiv_papers, 1):
        print(f"[{i}] {paper.title}")
        print(f"    arXiv ID: {paper.extra.get('arxiv_id', 'N/A')}")
        print(f"    作者: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"    发布日期: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
        print(f"    PDF链接: {paper.pdf_url}")
        print()

    # 测试 PMCSearcher
    print("\n【测试 PMCSearcher】")
    print("-" * 40)
    pmc_searcher = PMCSearcher()
    query2 = "insulin"
    print(f"检索关键词: {query2}")

    pmc_papers = await pmc_searcher.search(query2, limit=3)
    print(f"检索到 {len(pmc_papers)} 篇 PMC 论文:\n")

    for i, paper in enumerate(pmc_papers, 1):
        print(f"[{i}] {paper.title}")
        print(f"    PMCID: {paper.extra.get('pmcid', 'N/A')}")
        print(f"    作者: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"    期刊: {paper.extra.get('journal', 'N/A')}")
        print(f"    发布日期: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
        print(f"    PDF链接: {paper.pdf_url}")
        print()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

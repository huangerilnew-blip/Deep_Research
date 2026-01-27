# paper_search_mcp/sources/arxiv.py
from typing import List, Union
from datetime import datetime
import httpx
import asyncio
import feedparser
import random
from .paper import Paper
import os
from core.config.config import Config

class PaperSource:
    """Abstract base class for paper sources"""
    def search(self, query: str, **kwargs) -> List[Paper]:
        raise NotImplementedError

    def download_pdf(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

    def read_paper(self, paper_id: str, save_path: str) -> str:
        raise NotImplementedError

class ArxivSearcher(PaperSource):
    """Searcher for arXiv papers"""

    def __init__(self):
        self.BASE_URL = "https://export.arxiv.org/api/query"

    async def search(self, client: httpx.AsyncClient, query: str, max_results: int = 10) -> List[Paper]:
        """
        搜索 arXiv 论文

        Args:
            client: httpx.AsyncClient 实例（由调用方管理生命周期）
            query: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            List[Paper]: 论文列表
        """
        params = {
            'search_query': query,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        response = await client.get(self.BASE_URL, params=params)
        print(f"Response status: {response.status_code}")
        feed = feedparser.parse(response.content)
        print(f"Feed entries count: {len(feed.entries)}")
        papers = []
        for entry in feed.entries:
            try:
                authors = [author.name for author in entry.authors]
                published = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
                updated = datetime.strptime(entry.updated, '%Y-%m-%dT%H:%M:%SZ')
                pdf_url = next((link.href for link in entry.links if link.type == 'application/pdf'), '')
                paper_id = entry.id.split('/')[-1]
                # 自行构造 arXiv DOI
                doi = f"10.48550/arXiv.{paper_id}"
                papers.append(Paper(
                    paper_id=paper_id,
                    title=entry.title,
                    authors=authors,
                    abstract=entry.summary,
                    url=entry.id,
                    pdf_url=pdf_url,
                    published_date=published,
                    updated_date=updated,
                    source='arxiv',
                    categories=[tag.term for tag in entry.tags],
                    keywords=[],
                    doi=doi
                ))
            except Exception as e:
                print(f"Error parsing arXiv entry: {e}")
        return papers

    async def download(
        self,
        client: httpx.AsyncClient,
        papers: Union[Paper, List[Paper]],
        save_path: str = None
    ) -> List[Paper]:
        """
        根据 Paper 对象下载 arXiv HTML 文档

        Args:
            client: httpx.AsyncClient 实例（由调用方管理生命周期）
            papers: 单个 Paper 对象或 Paper 列表
            save_path: 保存路径，默认为 ./downloads

        Returns:
            List[Paper]: 更新后的 Paper 列表，包含下载路径信息
        """
        print(f"[DEBUG] download 方法被调用")
        print(f"[DEBUG] client 类型: {type(client)}")
        print(f"[DEBUG] client: {client}")

        # 处理默认 save_path
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH
            print(f"[DEBUG] 使用默认 save_path: {save_path}")
        else:
            print(f"[DEBUG] 使用传入 save_path: {save_path}")

        # 处理单个 Paper 或 Paper 列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)

        print(f"[DEBUG] paper_list 长度: {len(paper_list)}")

        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

        # 统计有链接的论文数量
        papers_with_pdf = [p for p in paper_list if p.pdf_url]
        print(f"共 {len(papers_with_pdf)} 篇论文待下载")

        success_count = 0
        for paper in paper_list:
            print(f"[DEBUG] 开始处理论文: {paper.paper_id}")
            if not paper.pdf_url:
                print(f"[DEBUG] 论文 {paper.paper_id} 没有 URL")
                saved_path = "No fulltext available"
            else:
                try:
                    # 1. 使用随机延迟 (0.5-1 秒)
                    wait_time = random.uniform(0.5, 1.0)
                    print(f"[DEBUG] 等待 {wait_time:.1f} 秒...")
                    await asyncio.sleep(wait_time)
                    print(f"[DEBUG] 开始下载...")

                    # 2. 规范化 arXiv ID
                    clean_id = paper.paper_id.replace('abs/', '').split('v')[0]
                    print(f"[DEBUG] clean_id: {clean_id}")

                    # 3. 构造 arXiv HTML URL
                    download_url = f"https://arxiv.org/html/{clean_id}"
                    print(f"[DEBUG] 标准化 URL: {download_url}")

                    # 4. 使用 DOI 命名，将 / 替换为 _ 避免路径问题
                    if paper.doi:
                        filename = paper.doi.replace('/', '_')
                    else:
                        filename = paper.paper_id
                    output_file = os.path.join(save_path, f"{filename}.html")
                    print(f"[DEBUG] 输出文件: {output_file}")

                    # 5. 添加请求头，模拟浏览器行为
                    download_headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    }

                    # 6. 使用流式下载
                    print(f"[DEBUG] 发起请求...")
                    async with client.stream("GET", download_url, headers=download_headers, timeout=120.0, follow_redirects=True) as response:
                        print(f"[DEBUG] 响应状态码: {response.status_code}")

                        # 7. 检查 Content-Type，确保是 HTML
                        content_type = response.headers.get("Content-Type", "")
                        print(f"[DEBUG] Content-Type: {content_type}")

                        if response.status_code == 200:
                            if "text/html" not in content_type:
                                print(f"[WARNING] 返回的不是 HTML，可能是错误页")
                                print(f"[WARNING] URL: {download_url}")
                                print(f"[WARNING] Content-Type: {content_type}")
                                saved_path = "No fulltext available"
                            else:
                                with open(output_file, 'wb') as f:
                                    async for chunk in response.aiter_bytes(chunk_size=65536):
                                        f.write(chunk)
                                saved_path = output_file
                                print(f"已保存: {saved_path}")
                                print(f"[DEBUG] 流式下载完成")
                                success_count += 1
                        else:
                            print(f"[DEBUG] 下载失败，状态码: {response.status_code}")
                            saved_path = "No fulltext available"
                except Exception as e:
                    import traceback
                    print(f"下载失败 {paper.paper_id}: {e}")
                    traceback.print_exc()
                    saved_path = "No fulltext available"

            # 更新 Paper.extra["saved_path"]
            if paper.extra is None:
                paper.extra = {}
            paper.extra["saved_path"] = saved_path

        print(f"下载完成: {success_count}/{len(papers_with_pdf)}")

        # 返回更新后的 Paper 列表
        return paper_list

async def main():
    # 测试 ArxivSearcher 的功能
    searcher = ArxivSearcher()

    # 创建 client 时添加 User-Agent 和 follow_redirects
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    # 测试搜索功能
    print("Testing search functionality...")
    query = "machine learning"
    max_results = 5
    papers = None

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as search_client:
        print(f"[TEST] search_client 创建: {search_client}")
        try:
            papers = await searcher.search(search_client, query, max_results=max_results)
            print(f"Found {len(papers)} papers for query '{query}':")
            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.title} (ID: {paper.paper_id})")
        except Exception as e:
            print(f"Error during search: {e}")
            return

    if papers:
        print("\nTesting HTML download functionality...")
        save_path = "./testdownloads"

        # 使用另一个独立的 client 用于下载
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as download_client:
            print(f"[TEST] download_client 创建: {download_client}")
            print(f"[TEST] search_client 和 download_client 是不同对象: {search_client is not download_client}")
            try:
                downloaded_papers = await searcher.download(download_client, papers[:2], save_path)
                # 检查 saved_path
                for paper in downloaded_papers:
                    print(f"{paper.paper_id}: {paper.extra.get('saved_path', 'N/A')}")
            except Exception as e:
                print(f"Error during HTML download: {e}")

if __name__ == "__main__":
    asyncio.run(main())

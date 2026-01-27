# paper_search_mcp/sources/arxiv.py
from typing import List, Union
from datetime import datetime
import httpx
import asyncio
import feedparser
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
        self.session = httpx.AsyncClient(timeout=30.0)
    async  def search(self, query: str, max_results: int = 10) -> List[Paper]:
        params = {
            'search_query': query,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        response = await self.session.get(self.BASE_URL, params=params)
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
        papers: Union[Paper, List[Paper]],
        save_path: str = Config.DOC_SAVE_PATH
    ) -> List[Paper]:
        """
        根据 Paper 对象中的 pdf_url 下载 arXiv 文档

        Args:
            papers: 单个 Paper 对象或 Paper 列表
            save_path: 保存路径，默认为 ./downloads

        Returns:
            List[Paper]: 更新后的 Paper 列表，包含下载路径信息
        """
        # 处理单个 Paper 或 Paper 列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)

        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

        # 统计有 PDF 链接的论文数量
        papers_with_pdf = [p for p in paper_list if p.pdf_url]
        print(f"共 {len(papers_with_pdf)} 篇论文待下载")

        success_count = 0
        for paper in paper_list:
            if not paper.pdf_url:
                saved_path = "No fulltext available"
            else:
                try:
                    # 添加延迟避免频繁请求
                    await asyncio.sleep(0.5)

                    # 下载 PDF
                    response = await self.session.get(paper.pdf_url, timeout=30.0)
                    response.raise_for_status()

                    # 使用 DOI 命名，将 / 替换为 _ 避免路径问题
                    if paper.doi:
                        filename = paper.doi.replace('/', '_')
                    else:
                        filename = paper.paper_id
                    output_file = os.path.join(save_path, f"{filename}.pdf")

                    # 保存文件
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    saved_path = output_file
                    print(f"已保存: {saved_path}")
                    success_count += 1
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

    # 测试搜索功能
    print("Testing search functionality...")
    query = "machine learning"
    max_results = 5
    try:
        papers = await searcher.search(query, max_results=max_results)
        print(f"Found {len(papers)} papers for query '{query}':")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.title} (ID: {paper.paper_id})")

        if papers:
            print("\nTesting PDF download functionality...")
            save_path = "./downloads"
            try:
                downloaded_papers = await searcher.download(papers[:2], save_path)
                # 检查 saved_path
                for paper in downloaded_papers:
                    print(f"{paper.paper_id}: {paper.extra.get('saved_path', 'N/A')}")
            except Exception as e:
                print(f"Error during PDF download: {e}")
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    asyncio.run(main())



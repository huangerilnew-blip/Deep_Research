import asyncio
import httpx
import xmltodict
import os
from config import Config
from datetime import datetime
from typing import List, Optional
from .paper import Paper  # 假设你的类定义在次


class PubMedSearcher:
    def __init__(self, email: str=Config.EMAIL):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    async def run(self, query: str, limit: int = 5, download_dir: str = "downloads"):
        """主入口：检索、解析、筛选、下载"""
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        async with httpx.AsyncClient(timeout=30.0, headers=self.headers) as client:
            # Step 1: 检索 ID 列表
            print(f"正在检索关键词: {query}...")
            pmids = await self._search_ids(client, query, limit)
            if not pmids:
                print("未找到相关文献。")
                return []

            # Step 2: 获取元数据并解析为 Paper 对象
            print(f"正在获取 {len(pmids)} 篇文献的元数据...")
            papers = await self._fetch_and_parse(client, pmids)

            final_results = []
            for paper in papers:
                # Step 3: 摘要匹配 (这里可以替换为更复杂的 LLM 逻辑)
                if self._check_relevance(query, paper.abstract):
                    print(f"匹配成功: {paper.title[:50]}...")

                    # Step 4: 下载/获取全文
                    content_path = await self._acquire_content(client, paper, download_dir)
                    paper.extra["saved_path"] = content_path
                    final_results.append(paper)
                else:
                    print(f"跳过不相关文献: {paper.title[:50]}")

            return final_results

    async def _search_ids(self, client: httpx.AsyncClient, query: str, limit: int) -> List[str]:
        params = {
            "db": "pubmed", "term": query, "retmax": limit,
            "retmode": "json", "email": self.email
        }
        resp = await client.get(f"{self.base_url}/esearch.fcgi", params=params)
        return resp.json().get("esearchresult", {}).get("idlist", [])

    async def _fetch_and_parse(self, client: httpx.AsyncClient, ids: List[str]) -> List[Paper]:
        params = {
            "db": "pubmed", "id": ",".join(ids),
            "retmode": "xml", "email": self.email
        }
        resp = await client.get(f"{self.base_url}/efetch.fcgi", params=params)
        data = xmltodict.parse(resp.text)
        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(articles, dict): articles = [articles]

        return [self._map_to_paper(art) for art in articles]

    def _map_to_paper(self, art: dict) -> Paper:
        """核心解析逻辑：将 XML 字典映射到 Paper 类"""
        medline = art.get("MedlineCitation", {})
        article = medline.get("Article", {})
        pubmed_data = art.get("PubmedData", {})

        # 提取 ID 和 DOI
        pmid = medline.get("PMID", {}).get("#text", "")
        doi = ""
        pmc_id = ""
        for id_obj in pubmed_data.get("ArticleIdList", {}).get("ArticleId", []):
            if id_obj.get("@IdType") == "doi": doi = id_obj.get("#text")
            if id_obj.get("@IdType") == "pmc": pmc_id = id_obj.get("#text")

        # 提取摘要
        abs_node = article.get("Abstract", {}).get("AbstractText", "")
        abstract = " ".join(abs_node) if isinstance(abs_node, list) else str(abs_node)

        # 提取日期
        p_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        try:
            pub_date = datetime(int(p_date.get("Year", 2024)), 1, 1)
        except:
            pub_date = datetime.now()

        return Paper(
            paper_id=pmid,
            title=article.get("ArticleTitle", "No Title"),
            authors=[f"{a.get('LastName')} {a.get('ForeName')}" for a in article.get("AuthorList", {}).get("Author", [])
                     if isinstance(a, dict)],
            abstract=abstract,
            doi=doi,
            published_date=pub_date,
            pdf_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/" if pmc_id else "",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="pubmed",
            extra={"pmcid": pmc_id}
        )

    def _check_relevance(self, query: str, abstract: str) -> bool:
        """简单的关键词匹配筛选，建议此处对接 LLM"""
        keywords = query.lower().split()
        return any(word in abstract.lower() for word in keywords)

    async def _acquire_content(self, client: httpx.AsyncClient, paper: Paper, download_dir: str) -> str:
        """下载 PDF 或 获取 XML 全文"""
        # 尝试下载 PDF (如果有 PMC ID)
        if paper.pdf_url:
            file_path = os.path.join(download_dir, f"{paper.paper_id}.pdf")
            try:
                resp = await client.get(paper.pdf_url, follow_redirects=True)
                if resp.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(resp.content)
                    return file_path
            except Exception as e:
                print(f"PDF 下载失败: {e}")

        # 如果没有 PDF 或下载失败，获取 XML 全文内容作为 fallback
        pmcid = paper.extra.get("pmcid")
        if pmcid:
            params = {"db": "pmc", "id": pmcid, "retmode": "xml", "email": self.email}
            resp = await client.get(f"{self.base_url}/efetch.fcgi", params=params)
            file_path = os.path.join(download_dir, f"{paper.paper_id}.xml")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            return file_path

        return "No fulltext available"


# --- 使用示例 ---
async def main():
    searcher = PubMedSearcher(email="your_email@example.com")
    results = await searcher.run("Deep Learning Cancer Detection", limit=3)

    for p in results:
        print(f"\n最终保存: {p.title}")
        print(f"文件位置: {p.extra.get('saved_path')}")


if __name__ == "__main__":
    asyncio.run(main())
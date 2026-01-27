import asyncio
import httpx
import xmltodict
import os
from core.config import Config
from datetime import datetime
from typing import List, Optional, Union
from tools.core_tools.paper import Paper


class PubMedSearcher:
    def __init__(self, email: str=Config.EMAIL):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    async def search(self, query: str, limit: int = 5) -> List[Paper]:
        """
        根据查询关键词检索 PubMed 文献
        
        该方法执行以下步骤：
        1. 调用 PubMed API 搜索匹配的文献 ID (PMID)
        2. 获取并解析文献的完整元数据
        3. 通过相关性筛选过滤结果
        
        Args:
            query: 检索关键词，支持 PubMed 查询语法
            limit: 最大返回数量，默认为 5
            
        Returns:
            List[Paper]: 符合条件的 Paper 对象列表，包含完整的元信息：
                - paper_id: PMID 唯一标识符
                - title: 论文标题
                - authors: 作者列表
                - abstract: 摘要文本
                - doi: Digital Object Identifier
                - published_date: 发布日期
                - pdf_url: PDF/XML 下载地址（如有）
                - url: 论文页面链接
                - source: 来源平台（"pubmed"）
                - updated_date: 更新日期（如有）
                - categories: 学科分类（如有）
                - keywords: 关键词（如有）
                - references: 参考文献列表（如有）
                - extra: 额外元数据（如 pmcid）

        """
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

            # Step 3: 相关性筛选
            final_results = []
            for paper in papers:
                if self._check_relevance(query, paper.abstract):
                    print(f"匹配成功: {paper.title[:50]}...")
                    final_results.append(paper)
                else:
                    print(f"跳过不相关文献: {paper.title[:50]}")

            return final_results

    async def download(
        self, 
        papers: Union[Paper, List[Paper]], 
        save_path: str = Config.DOC_SAVE_PATH
    ) -> List[Paper]:
        """
        根据 Paper 对象中的 pdf_url 下载文档
        
        该方法执行以下步骤：
        1. 处理单个 Paper 或 Paper 列表输入
        2. 确保保存目录存在（不存在则自动创建）
        3. 遍历 Paper 列表，根据 pdf_url 下载文件
        4. 更新每个 Paper 的 extra["saved_path"] 字段
        
        Args:
            papers: 单个 Paper 对象或 Paper 列表
            save_path: 保存路径，默认使用 Config.DOC_SAVE_PATH
            
        Returns:
            List[Paper]: 更新后的 Paper 列表，包含下载路径信息：
                - 下载成功时：extra["saved_path"] 为文件保存路径
                - 下载失败或 pdf_url 为空时：extra["saved_path"] 为 "No fulltext available"
                

        """
        # 处理单个 Paper 或 Paper 列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)
        
        # 确保保存目录存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 创建 HTTP 客户端并下载文件
        async with httpx.AsyncClient(timeout=30.0, headers=self.headers) as client:
            for paper in paper_list:
                saved_path = await self._download_file(client, paper, save_path)
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = saved_path
                print(f"下载完成: {paper.title[:50]}... -> {saved_path}")
        
        return paper_list

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
            if isinstance(id_obj, dict):
                if id_obj.get("@IdType") == "doi": doi = id_obj.get("#text", "")
                if id_obj.get("@IdType") == "pmc": pmc_id = id_obj.get("#text", "")

        # 提取摘要
        abs_node = article.get("Abstract", {}).get("AbstractText", "")
        if isinstance(abs_node, list):
            # 处理结构化摘要（多个 AbstractText 节点）
            abstract_parts = []
            for part in abs_node:
                if isinstance(part, dict):
                    abstract_parts.append(part.get("#text", ""))
                else:
                    abstract_parts.append(str(part))
            abstract = " ".join(abstract_parts)
        elif isinstance(abs_node, dict):
            abstract = abs_node.get("#text", "")
        else:
            abstract = str(abs_node) if abs_node else ""

        # 提取发布日期
        p_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        try:
            year = int(p_date.get("Year", 2024))
            month = self._parse_month(p_date.get("Month", "1"))
            day = int(p_date.get("Day", 1))
            pub_date = datetime(year, month, day)
        except:
            pub_date = datetime.now()

        # 提取更新日期 (DateRevised)
        date_revised = medline.get("DateRevised", {})
        updated_date = None
        if date_revised:
            try:
                year = int(date_revised.get("Year", 0))
                month = int(date_revised.get("Month", 1))
                day = int(date_revised.get("Day", 1))
                if year > 0:
                    updated_date = datetime(year, month, day)
            except:
                pass

        # 提取学科分类 (MeshHeadingList)
        categories = []
        mesh_list = medline.get("MeshHeadingList", {}).get("MeshHeading", [])
        if isinstance(mesh_list, dict):
            mesh_list = [mesh_list]
        for mesh in mesh_list:
            if isinstance(mesh, dict):
                descriptor = mesh.get("DescriptorName", {})
                if isinstance(descriptor, dict):
                    cat_name = descriptor.get("#text", "")
                    if cat_name:
                        categories.append(cat_name)

        # 提取关键词 (KeywordList)
        keywords = []
        keyword_list = medline.get("KeywordList", {})
        if keyword_list:
            kw_items = keyword_list.get("Keyword", [])
            if isinstance(kw_items, dict):
                kw_items = [kw_items]
            elif isinstance(kw_items, str):
                kw_items = [kw_items]
            for kw in kw_items:
                if isinstance(kw, dict):
                    kw_text = kw.get("#text", "")
                    if kw_text:
                        keywords.append(kw_text)
                elif isinstance(kw, str) and kw:
                    keywords.append(kw)

        # 提取参考文献 (ReferenceList)
        references = []
        ref_list = pubmed_data.get("ReferenceList", {})
        if ref_list:
            refs = ref_list.get("Reference", [])
            if isinstance(refs, dict):
                refs = [refs]
            for ref in refs:
                if isinstance(ref, dict):
                    # 尝试获取引用的 ArticleId
                    article_id_list = ref.get("ArticleIdList", {}).get("ArticleId", [])
                    if isinstance(article_id_list, dict):
                        article_id_list = [article_id_list]
                    for aid in article_id_list:
                        if isinstance(aid, dict):
                            ref_id = aid.get("#text", "")
                            if ref_id:
                                references.append(ref_id)
                                break
                    else:
                        # 如果没有 ArticleId，使用 Citation 文本
                        citation = ref.get("Citation", "")
                        if citation:
                            references.append(citation[:100])  # 截断过长的引用

        # 提取作者列表
        authors = []
        author_list = article.get("AuthorList", {}).get("Author", [])
        if isinstance(author_list, dict):
            author_list = [author_list]
        for a in author_list:
            if isinstance(a, dict):
                last_name = a.get('LastName', '')
                fore_name = a.get('ForeName', '')
                if last_name or fore_name:
                    authors.append(f"{last_name} {fore_name}".strip())

        return Paper(
            paper_id=pmid,
            title=article.get("ArticleTitle", "No Title"),
            authors=authors,
            abstract=abstract,
            doi=doi,
            published_date=pub_date,
            pdf_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/" if pmc_id else "",
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="pubmed",
            updated_date=updated_date,
            categories=categories,
            keywords=keywords,
            references=references,
            extra={"pmcid": pmc_id}
        )

    def _parse_month(self, month_str: str) -> int:
        """解析月份字符串，支持数字和英文缩写"""
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        try:
            return int(month_str)
        except ValueError:
            return month_map.get(month_str.lower()[:3], 1)

    def _check_relevance(self, query: str, abstract: str) -> bool:
        """简单的关键词匹配筛选，建议此处对接 LLM"""
        keywords = query.lower().split()
        return any(word in abstract.lower() for word in keywords)

    async def _download_file(
        self, 
        client: httpx.AsyncClient, 
        paper: Paper, 
        save_path: str
    ) -> str:
        """
        下载单个文件
        
        根据 Paper.pdf_url 下载文件，如果 pdf_url 为空或下载失败，
        则尝试通过 PMC ID 获取 XML 全文作为备选方案。
        
        Args:
            client: HTTP 客户端
            paper: Paper 对象
            save_path: 保存目录
            
        Returns:
            str: 文件保存路径（下载成功时）或 "No fulltext available"（下载失败或无链接时）
        """
        # 尝试下载 PDF (如果有 pdf_url)
        if paper.pdf_url:
            file_path = os.path.join(save_path, f"{paper.paper_id}.pdf")
            try:
                resp = await client.get(paper.pdf_url, follow_redirects=True)
                if resp.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(resp.content)
                    return file_path
            except Exception as e:
                print(f"PDF 下载失败: {e}")

        # 如果没有 PDF 或下载失败，尝试获取 XML 全文作为 fallback
        pmcid = paper.extra.get("pmcid") if paper.extra else None
        if pmcid:
            try:
                params = {"db": "pmc", "id": pmcid, "retmode": "xml", "email": self.email}
                resp = await client.get(f"{self.base_url}/efetch.fcgi", params=params)
                if resp.status_code == 200:
                    file_path = os.path.join(save_path, f"{paper.paper_id}.xml")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(resp.text)
                    return file_path
            except Exception as e:
                print(f"XML 下载失败: {e}")

        return "No fulltext available"


# --- 使用示例 ---
async def main():
    searcher = PubMedSearcher(email=Config.EMAIL)
    
    # Step 1: 检索文献
    print("=== 检索文献 ===")
    papers = await searcher.search("Deep Learning Cancer Detection", limit=3)
    
    if not papers:
        print("未找到相关文献")
        return
    
    print(f"\n找到 {len(papers)} 篇相关文献:")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p.title[:60]}...")
        print(f"   作者: {', '.join(p.authors[:3])}{'...' if len(p.authors) > 3 else ''}")
        print(f"   DOI: {p.doi or 'N/A'}")
        print(f"   PDF链接: {'有' if p.pdf_url else '无'}")
    
    # Step 2: 下载文献（可选）
    print("\n=== 下载文献 ===")
    downloaded = await searcher.download(papers)
    
    for p in downloaded:
        print(f"\n文献: {p.title[:50]}...")
        print(f"保存位置: {p.extra.get('saved_path')}")


if __name__ == "__main__":
    asyncio.run(main())
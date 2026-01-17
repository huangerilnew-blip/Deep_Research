
import os
import asyncio
import httpx
from typing import List, Union
from datetime import datetime

from paper import Paper
from config import Config


class WikipediaSearcher:
    """
    Wikipedia 百科内容检索器类
    
    使用 httpx 异步并发请求获取 Wikipedia 词条内容，
    支持多语言（英文、中文等）。
    """
    
    def __init__(self, language: str = "en", timeout: int = 30):
        """
        初始化 WikipediaSearcher
        
        Args:
            language: 语言代码，默认"en"（英文），支持"zh"（中文）等
            timeout: HTTP请求超时时间，默认30秒
        """
        self.language = language
        self.timeout = timeout
        # Wikipedia API 要求特定格式的 User-Agent
        self.headers = {
            "User-Agent": "PaperSearchBot/1.0 (https://github.com/paper-search; contact@example.com)"
        }
    
    def _get_api_url(self, language: str = None) -> str:
        """获取 Wikipedia API URL"""
        lang = language or self.language
        return f"https://{lang}.wikipedia.org/w/api.php"

    async def _search_articles(self, query: str, limit: int, language: str) -> List[dict]:
        """搜索 Wikipedia 词条"""
        api_url = self._get_api_url(language)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json"
        }
        
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
                response = await client.get(api_url, params=params)
                if response.status_code != 200:
                    print(f"Wikipedia API 请求失败: HTTP {response.status_code}")
                    return []
                
                data = response.json()
                if "query" not in data or "search" not in data["query"]:
                    return []
                
                return [{"pageid": item.get("pageid"), "title": item.get("title", "")} 
                        for item in data["query"]["search"]]
        except Exception as e:
            print(f"Wikipedia API 请求异常: {e}")
            return []

    async def _get_article_content(self, client: httpx.AsyncClient, pageid: int, language: str) -> dict:
        """获取单个词条内容（异步）"""
        api_url = self._get_api_url(language)
        params = {
            "action": "query",
            "pageids": pageid,
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "format": "json"
        }
        
        try:
            response = await client.get(api_url, params=params)
            if response.status_code != 200:
                return {}
            
            data = response.json()
            if "query" not in data or "pages" not in data["query"]:
                return {}
            
            pages = data["query"]["pages"]
            page_data = pages.get(str(pageid), {})
            
            return {
                "pageid": page_data.get("pageid"),
                "title": page_data.get("title", ""),
                "extract": page_data.get("extract", ""),
                "fullurl": page_data.get("fullurl", "")
            }
        except Exception as e:
            print(f"获取词条内容失败 (pageid={pageid}): {e}")
            return {}

    def _map_to_paper(self, article_data: dict, language: str) -> Paper:
        """将 Wikipedia 数据映射到 Paper 对象"""
        pageid = article_data.get("pageid", 0)
        title = article_data.get("title", "")
        extract = article_data.get("extract", "")
        fullurl = article_data.get("fullurl", "")
        
        # abstract 保存完整内容，不截取
        abstract = extract
        
        extra = {
            "full_extract": extract,
            "pageid": pageid,
            "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": language
        }
        
        return Paper(
            paper_id=str(pageid),
            title=title,
            authors=["Wikipedia"],
            abstract=abstract,
            doi="",
            published_date=datetime.now(),
            pdf_url=None,
            url=fullurl,
            source="wikipedia",
            categories=[],
            keywords=[],
            citations=0,
            references=[],
            extra=extra
        )

    async def search(self, query: str, limit: int =Config.WIKI_NUM, language: str = Config.WIKI_LANGUAGE) -> List[Paper]:
        """
        根据关键词搜索 Wikipedia 词条（异步并发获取内容）
        
        Args:
            query: 搜索关键词
            limit: 最大返回数量，默认3
            language: 语言代码，覆盖初始化时的设置
            
        Returns:
            List[Paper]: Paper 对象列表
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []
        
        query = query.strip()
        lang = language or self.language
        
        print(f"正在搜索 Wikipedia ({lang}): {query}")
        search_results = await self._search_articles(query, limit, lang)
        
        if not search_results:
            print(f"未找到相关词条: {query}")
            return []
        
        print(f"找到 {len(search_results)} 个词条，正在并发获取内容...")
        
        # 异步并发获取词条内容
        papers = []
        async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout) as client:
            tasks = [
                self._get_article_content(client, item["pageid"], lang)
                for item in search_results if item.get("pageid")
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for article_data in results:
                if isinstance(article_data, Exception):
                    continue
                if article_data and article_data.get("extract"):
                    paper = self._map_to_paper(article_data, lang)
                    papers.append(paper)
                    print(f"  ✓ {paper.title}")
        
        print(f"成功获取 {len(papers)} 个词条")
        return papers

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        if not filename:
            return "unnamed"
        forbidden_chars = '<>:"/\\|?*\x00\n\r\t'
        result = filename
        for char in forbidden_chars:
            result = result.replace(char, '_')
        result = result.strip()
        return result[:200] if len(result) > 200 else result if result else "unnamed"

    def _generate_markdown(self, paper: Paper) -> str:
        """生成 Markdown 格式内容"""
        extra = paper.extra or {}
        lines = [
            f"# {paper.title}",
            "",
            "## 基本信息",
            f"- 词条ID: {paper.paper_id}",
            f"- 来源: Wikipedia ({extra.get('language', 'en')})",
            f"- URL: {paper.url}",
            f"- 获取时间: {extra.get('fetch_time', '')}",
            "",
            "## 完整内容",
            paper.abstract,
            "",
            "---",
            f"*数据来源: Wikipedia | 获取时间: {extra.get('fetch_time', '')}*"
        ]
        return "\n".join(lines)

    def _save_file(self, content: str, save_path: str, filename: str) -> str:
        """保存文件"""
        try:
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return file_path
        except Exception as e:
            return f"Error: {e}"

    async def download(self, papers: Union[Paper, List[Paper]], save_path: str =Config.DOC_SAVE_PATH) -> List[Paper]:
        """
        将 Paper 对象导出为 Markdown 文件
        
        Args:
            papers: Paper 对象或 Paper 列表
            save_path: 保存路径，默认 Config.DOC_SAVE_PATH
            
        Returns:
            List[Paper]: 更新了 saved_path 的 Paper 列表
        """
        paper_list = [papers] if isinstance(papers, Paper) else list(papers)
        if not paper_list:
            print("没有 Paper 需要导出。")
            return paper_list

        
        for paper in paper_list:
            filename = f"wiki_{paper.paper_id}.md"
            file_path = os.path.join(save_path, filename)
            
            # 检查文件是否已存在
            if os.path.exists(file_path):
                print(f"文件已存在，跳过下载: {paper.title} -> {file_path}")
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["saved_path"] = file_path
                continue
            
            markdown_content = self._generate_markdown(paper)
            saved_path = self._save_file(markdown_content, save_path, filename)
            
            if paper.extra is None:
                paper.extra = {}
            paper.extra["saved_path"] = saved_path
            
            if not saved_path.startswith("Error:"):
                print(f"导出成功: {paper.title} -> {saved_path}")
            else:
                print(f"导出失败: {paper.title}")
        
        return paper_list


# --- 使用示例 ---
async def main():
    """WikipediaSearcher 使用示例"""
    searcher = WikipediaSearcher()
    
    print("=" * 60)
    print("Wikipedia 百科内容检索器使用示例 (httpx异步并发)")
    print("=" * 60)
    
    # 示例 1: 英文 Wikipedia 搜索
    print("\n【示例 1】英文 Wikipedia 搜索")
    print("-" * 40)
    papers = await searcher.search("人工智能中美差距情况", limit=2)
    
    if papers:
        for paper in papers:
            print(f"\n词条: {paper.title}")
            print(f"URL: {paper.url}")
            print(f"摘要: {paper.abstract[:150]}...")
    
    # 示例 2: 中文 Wikipedia 搜索
    print("\n【示例 2】中文 Wikipedia 搜索")
    print("-" * 40)
    zh_papers = await searcher.search("人工智能", limit=2, language="zh")
    
    if zh_papers:
        for paper in zh_papers:
            print(f"\n词条: {paper.title}")
            print(f"URL: {paper.url}")
            print(f"摘要: {paper.abstract[:150]}...")
    
    # 示例 3: 导出为 Markdown
    print("\n【示例 3】导出为 Markdown")
    print("-" * 40)
    all_papers = papers + zh_papers
    if all_papers:
        await searcher.download(all_papers)
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

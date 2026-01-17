import asyncio
import hashlib
import httpx
import json
import os
from config import Config
from datetime import datetime
from typing import List, Optional, Union
from paper import Paper
from dotenv import load_dotenv
load_dotenv()
class TavilySearcher:
    """Tavily网络搜索器，提供search和download功能"""
    
    def __init__(self):
        """
        初始化TavilySearcher
        - 从Config读取TAVILY_API_KEY
        - 从Config读取TAVILY_NUM
        - 验证API Key是否配置
        """
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.max_results = Config.TAVILY_NUM
        self.base_url = "https://api.tavily.com/search"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # 验证API Key是否配置
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY未配置，请在环境变量中设置TAVILY_API_KEY")
    
    async def search(self, query: str, limit: int = None) -> List[Paper]:
        """
        执行Tavily搜索并返回Paper列表
        
        Args:
            query: 搜索关键词
            limit: 结果数量限制，默认使用Config.TAVILY_NUM
            
        Returns:
            List[Paper]: 解析后的Paper对象列表（仅包含score > 0.8的结果）
        """
        max_results = limit if limit is not None else self.max_results
        
        print(f"正在使用Tavily搜索: {query}...")
        response_data = await self._call_api(query, max_results)
        
        if not response_data:
            print("未获取到搜索结果。")
            return []
        
        results = response_data.get("results", [])
        if not results:
            print("搜索结果为空。")
            return []
        
        print(f"获取到 {len(results)} 条搜索结果，正在过滤和解析...")
        papers = []
        filtered_count = 0
        
        for result in results:
            try:
                score = result.get("score", 0.0)
                
                # 只解析 score > 0.8 的结果
                if score <= 0.8:
                    filtered_count += 1
                    print(f"过滤低分结果 (score={score:.4f}): {result.get('title', 'N/A')[:50]}...")
                    continue
                
                paper = self._parse_result(result)
                papers.append(paper)
                print(f"解析成功 (score={score:.4f}): {paper.title[:50]}...")
            except Exception as e:
                print(f"解析结果失败，跳过: {e}")
                continue
        
        print(f"过滤了 {filtered_count} 条低分结果，保留 {len(papers)} 条高质量结果")
        return papers
    
    async def download(
        self, 
        papers: Union[Paper, List[Paper]], 
        save_path: str = None,
        filename: str = None
    ) -> List[Paper]:
        """
        将Paper列表导出为Markdown文件
        
        Args:
            papers: 单个Paper或Paper列表
            save_path: 保存目录，默认Config.DOC_SAVE_PATH
            filename: 文件名，默认使用时间戳生成
            
        Returns:
            List[Paper]: 更新了saved_path的Paper列表
        """
        # 处理单个Paper或Paper列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)
        
        if not paper_list:
            print("没有Paper需要下载。")
            return paper_list
        
        # 使用默认保存路径
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tavily_search_{timestamp}.md"
        
        # 生成Markdown内容
        markdown_lines = [
            f"# Tavily 搜索结果",
            f"",
            f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**结果数量:** {len(paper_list)}",
            f"",
            "---",
            ""
        ]
        
        for i, paper in enumerate(paper_list, 1):
            markdown_lines.append(self._paper_to_markdown(paper, index=i))
        
        markdown_content = "\n".join(markdown_lines)
        
        # 保存文件
        file_path = self._save_markdown(markdown_content, save_path, filename)
        
        # 更新每个Paper的saved_path
        for paper in paper_list:
            if paper.extra is None:
                paper.extra = {}
            paper.extra["saved_path"] = file_path
        
        print(f"Markdown文件已保存: {file_path}")
        return paper_list


    async def _call_api(self, query: str, max_results: int) -> dict:
        """
        调用Tavily API
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            dict: API响应JSON，失败时返回空dict
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_raw_content": True
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            print(f"API请求失败: {e.response.status_code} - {e.response.text}")
            return {}
        except httpx.RequestError as e:
            print(f"网络错误: {e}")
            return {}
        except Exception as e:
            print(f"未知错误: {e}")
            return {}

    def _parse_result(self, result: dict) -> Paper:
        """
        将单个Tavily搜索结果解析为Paper对象
        
        Args:
            result: Tavily API返回的单个结果
            
        Returns:
            Paper: 解析后的Paper对象
        """
        # 提取字段
        title = result.get("title", "No Title")
        content = result.get("content", "")
        raw_content = result.get("raw_content", "")
        url = result.get("url", "")
        score = result.get("score", 0.0)
        
        # 使用URL的MD5 hash生成paper_id
        paper_id = hashlib.md5(url.encode()).hexdigest()[:16] if url else f"tavily_{datetime.now().timestamp()}"
        
        # 处理发布日期
        published_date_raw = result.get("publishedDate")
        if published_date_raw:
            try:
                # Tavily返回的是毫秒时间戳
                if isinstance(published_date_raw, (int, float)):
                    published_date = datetime.fromtimestamp(published_date_raw / 1000)
                else:
                    published_date = datetime.fromisoformat(str(published_date_raw).replace('Z', '+00:00'))
            except Exception:
                published_date = datetime.now()
        else:
            published_date = datetime.now()
        
        return Paper(
            paper_id=paper_id,
            title=title,
            authors=[],
            abstract=content,
            doi="",
            published_date=published_date,
            pdf_url=None,
            url=url,
            source="tavily",
            extra={
                "score": score,
                "raw_content": raw_content
            }
        )

    def _paper_to_dict(self, paper: Paper) -> dict:
        """
        将单个Paper转换为字典格式
        
        Args:
            paper: Paper对象
            
        Returns:
            dict: Paper的字典表示
        """
        result = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "url": paper.url,
            "published_date": paper.published_date.strftime('%Y-%m-%d') if paper.published_date else None,
            "abstract": paper.abstract if paper.abstract else "",
            "score": paper.extra.get("score") if paper.extra else None,
            "source": paper.source
        }
        
        # 添加 raw_content（如果存在）
        if paper.extra and "raw_content" in paper.extra:
            result["raw_content"] = paper.extra["raw_content"]
        
        return result

    def _save_json(self, data: dict, save_path: str, filename: str) -> str:
        """
        保存JSON内容到文件
        
        Args:
            data: JSON数据
            save_path: 保存目录
            filename: 文件名
            
        Returns:
            str: 完整文件路径，失败时返回错误信息
        """
        try:
            # 确保目录存在
            os.makedirs(save_path, exist_ok=True)
            
            file_path = os.path.join(save_path, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return file_path
        except IOError as e:
            error_msg = f"文件保存失败: {e}"
            print(error_msg)
            return error_msg

    def _paper_to_markdown(self, paper: Paper, index: int = None) -> str:
        """
        将单个Paper转换为Markdown格式字符串
        
        Args:
            paper: Paper对象
            index: 序号（可选）
            
        Returns:
            str: Markdown格式字符串
        """
        lines = []
        
        # 标题
        if index:
            lines.append(f"## {index}. {paper.title}\n")
        else:
            lines.append(f"## {paper.title}\n")
        
        # 基本信息
        lines.append("### 基本信息")
        lines.append(f"- **来源:** [{paper.url}]({paper.url})")
        lines.append(f"- **发布日期:** {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
        
        # 添加相关性评分（如果有）
        if paper.extra and "score" in paper.extra:
            lines.append(f"- **相关性评分:** {paper.extra['score']:.4f}")
        
        # 摘要
        lines.append("\n### 摘要")
        lines.append(paper.abstract if paper.abstract else "无摘要内容")
        
        # 完整内容（如果有 raw_content）
        if paper.extra and paper.extra.get("raw_content"):
            lines.append("\n### 完整内容")
            lines.append(paper.extra["raw_content"])
        
        lines.append("\n---\n")
        
        return "\n".join(lines)

    def _save_markdown(self, content: str, save_path: str, filename: str) -> str:
        """
        保存Markdown内容到文件
        
        Args:
            content: Markdown内容
            save_path: 保存目录
            filename: 文件名
            
        Returns:
            str: 完整文件路径，失败时返回错误信息
        """
        try:
            # 确保目录存在
            os.makedirs(save_path, exist_ok=True)
            
            file_path = os.path.join(save_path, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return file_path
        except IOError as e:
            error_msg = f"文件保存失败: {e}"
            print(error_msg)
            return error_msg


# --- 使用示例 ---
async def main():
    try:
        searcher = TavilySearcher()
    except ValueError as e:
        print(f"初始化失败: {e}")
        return
    
    # Step 1: 搜索
    print("=== Tavily搜索 ===")
    query="Amid the US-China competition, what will become of Manus?"
    papers = await searcher.search(query)
    
    if not papers:
        print("未找到相关结果")
        return
    
    print(f"\n找到 {len(papers)} 条结果:")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p.title[:60]}...")
        print(f"   URL: {p.url}")
        print(f"   评分: {p.extra.get('score', 'N/A')}")
    
    # Step 2: 下载为Markdown
    print("\n=== 导出为Markdown ===")
    downloaded = await searcher.download(papers)
    
    for p in downloaded:
        print(f"\n文献: {p.title[:50]}...")
        print(f"保存位置: {p.extra.get('saved_path')}")


if __name__ == "__main__":
    asyncio.run(main())

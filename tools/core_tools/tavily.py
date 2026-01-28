"""
Tavily 搜索工具
提供 search 和 download 两个方法
"""
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

from tavily import TavilyClient
from core.config.config import Config
from .paper import Paper
from dotenv import load_dotenv
load_dotenv()  # 从 .env 文件加载环境变量

class TavilySearch:
    """Tavily 搜索类，提供 search 和 download 两个公共接口"""

    def __init__(self):
        """
        初始化 TavilySearch
        - 验证 API Key 是否配置
        - 初始化 TavilyClient
        """
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY 环境变量未设置")

        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = Config.TAVILY_NUM) -> List[Paper]:
        """
        使用 Tavily 搜索并返回 Paper 对象列表

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            List[Paper]: 过滤后的 Paper 列表（score >= 0.8）
        """
        # 执行搜索（参数固定）
        response = self.client.search(
            query=query,
            search_depth="advanced",
            topic="general",
            max_results=max_results,
            include_answer=False,
            include_raw_content="markdown",
            include_images=False,
            include_favicon=False,
            include_usage=False,
            timeout=30
        )

        # 解析结果并过滤
        papers = []
        for result in response.get('results', []):
            # 检查 score 并过滤
            score = result.get('score')
            if score is not None and score < 0.8:
                continue  # 过滤低质量结果

            # 转换为 Paper 对象
            paper = self._result_to_paper(result)
            papers.append(paper)

        return papers

    def download(self, papers: List[Paper], save_dir: str = None) -> List[Paper]:
        """
        将 Paper 列表保存为 Markdown 文件

        Args:
            papers: Paper 对象列表
            save_dir: 保存目录，默认为 Config.DOC_SAVE_PATH

        Returns:
            List[Paper]: 成功保存的 Paper 列表（paper.extra["save_path"] 已填充）
        """
        # 使用默认保存路径
        if save_dir is None:
            save_dir = Config.DOC_SAVE_PATH

        # 确保目录存在
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        saved_papers = []

        for paper in papers:
            try:
                # 生成文件名
                if paper.title and paper.title != "无标题":
                    # 有标题：tavily_{title}.md
                    safe_title = self._sanitize_filename(paper.title)
                    filename = f"tavily_{safe_title}.md"
                else:
                    # 无标题：tavily_无标题_{timestamp}.md
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"tavily_无标题_{timestamp}.md"

                # 构造完整路径
                file_path = os.path.join(save_dir, filename)

                # abstract 中保存的是 raw_content
                markdown_content = paper.abstract

                # 写入文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                # 更新 Paper 对象的 save_path
                paper.extra['save_path'] = file_path
                saved_papers.append(paper)

            except Exception as e:
                print(f"保存文件失败: {paper.title}, 错误: {e}")
                continue

        return saved_papers

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除非法字符"""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename[:100]  # 限制长度
        return filename

    def _result_to_paper(self, tavily_result: dict) -> Paper:
        """
        将 Tavily 搜索结果转换为 Paper 对象

        Args:
            tavily_result: Tavily API 返回的单个搜索结果

        Returns:
            Paper: 转换后的 Paper 对象
        """
        # 提取标题（如果为空，使用 "无标题"）
        title = tavily_result.get('title') or "无标题"

        # 构造 Paper 对象
        paper = Paper(
            # 核心字段
            paper_id="",  # 直接为空
            title=title,
            authors=[],  # Tavily 不提供作者信息
            abstract=tavily_result.get('raw_content', ''),  # 保存 raw_content 到 abstract
            doi='',  # Tavily 结果无 DOI
            published_date=None,  # 直接为空
            pdf_url='',  # Tavily 搜索结果无 PDF
            url=tavily_result['url'],
            source='tavily',
            # 可选字段
            updated_date=None,
            categories=['web_search'],
            keywords=[],
            citations=0,
            references=[],
            extra={
                'score': tavily_result.get('score'),  # 只保留 score
                'save_path': None  # 将在 download 时填充
            }
        )

        return paper


if __name__ == "__main__":
    # 测试代码
    print("=== 测试 Tavily 搜索和下载功能 ===\n")

    try:
        # 初始化 TavilySearch 实例
        tavily_search = TavilySearch()

        # 测试 1: 搜索
        print("测试 1: 执行搜索")
        print("-" * 50)
        query = "artificial intelligence latest developments"
        print(f"查询: {query}")
        print(f"最大结果数: 5")

        papers = tavily_search.search(query, max_results=5)

        print(f"\n搜索结果: 找到 {len(papers)} 篇高质量文档 (score >= 0.8)\n")

        if papers:
            for i, paper in enumerate(papers, 1):
                print(f"{i}. 标题: {paper.title}")
                print(f"   URL: {paper.url}")
                print(f"   Score: {paper.extra.get('score')}")
                print(f"   Paper ID: '{paper.paper_id}'")
                print(f"   Published Date: {paper.published_date}")
                print(f"   Abstract 长度: {len(paper.abstract)} 字符")
                print()

            # 测试 2: 下载
            print("\n测试 2: 下载为 Markdown 文件")
            print("-" * 50)
            print(f"保存目录: {Config.DOC_SAVE_PATH}")

            saved_papers = tavily_search.download(papers)

            print(f"\n成功保存 {len(saved_papers)} 篇文档:\n")

            for paper in saved_papers:
                print(f"- 标题: {paper.title}")
                print(f"  保存路径: {paper.extra.get('save_path')}")
                print()

            # 测试 3: 验证文件内容
            print("\n测试 3: 验证文件内容")
            print("-" * 50)
            for paper in saved_papers:
                save_path = paper.extra.get('save_path')
                if save_path and os.path.exists(save_path):
                    with open(save_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"文件: {os.path.basename(save_path)}")
                    print(f"大小: {len(content)} 字符")
                    print(f"前200字符预览: {content[:200]}...")
                    print()
                else:
                    print(f"文件不存在: {save_path}")
        else:
            print("未找到符合条件的搜索结果")

    except ValueError as e:
        print(f"\n错误: {e}")
        print("请确保已设置 TAVILY_API_KEY 环境变量")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

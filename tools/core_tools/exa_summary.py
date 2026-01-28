"""
Exa搜索器 - 使用exa_py库进行网络搜索和内容提取

功能:
- 使用search_and_contents方法搜索并获取网页内容
- 将每个搜索结果保存为独立的markdown文件

安装要求:
    pip install exa_py

环境变量:
    EXA_API_KEY: Exa API密钥 (需要在.env文件中配置)
"""

import os
from datetime import datetime
from typing import List, Optional
from exa_py import Exa
from dotenv import load_dotenv

from core.config import Config
from .paper import Paper

load_dotenv()


class ExaSearcherSummary:
    """Exa搜索器类，用于网络搜索和内容提取"""

    def __init__(self, api_key: str = None):
        """
        初始化ExaSearcher

        Args:
            api_key: Exa API密钥，如果未提供则从环境变量EXA_API_KEY读取
        """
        if api_key is None:
            api_key = os.getenv("EXA_API_KEY")

        if not api_key:
            raise ValueError("EXA_API_KEY未配置，请在环境变量中设置EXA_API_KEY")

        self.api_key = api_key
        self.client = Exa(api_key=api_key)
        self.max_results = Config.EXA_NUM

    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除非法字符

        Args:
            filename: 原始文件名

        Returns:
            str: 清理后的文件名
        """
        if not filename:
            return "unnamed"

        forbidden_chars = '<>:"/\\|?*\x00\n\r\t'
        result = filename
        for char in forbidden_chars:
            result = result.replace(char, '_')

        result = result.strip()
        return result[:200] if len(result) > 200 else result

    def _should_save_result(self, result: dict) -> bool:
        """
        判断是否应该保存该结果

        判断条件：
        - url 必须存在且不为空
        - score 为空（不存在或None）或 score > 0.8
        - 并且 summary 或 text 至少有一个不为空

        Args:
            result: 搜索结果字典

        Returns:
            bool: True表示应该保存
        """
        url = result.get('url', '')
        score = result.get('score')
        summary = result.get('summary', '')
        text = result.get('text', '')

        # 检查 url 条件：必须存在且不为空
        url_condition = bool(url.strip()) if url else False

        # 检查 score 条件：为空或 > 0.8
        score_condition = score is None or score > 0.8

        # 检查内容条件：summary 或 text 至少有一个不为空
        summary_condition = bool(summary.strip()) if summary else False
        text_condition = bool(text.strip()) if text else False
        content_condition = summary_condition or text_condition

        return url_condition and score_condition and content_condition

    def _generate_markdown(self, result: dict) -> str:
        """
        将搜索结果转换为Markdown格式（包含title, url, summary, data/text）

        Args:
            result: 搜索结果字典

        Returns:
            str: Markdown格式内容
        """
        title = result.get('title', '无标题')
        url = result.get('url', '')
        summary = result.get('summary', '')
        text = result.get('text', '')
        published_date = result.get('published_date', '未知')
        author = result.get('author', '')
        score = result.get('score')

        lines = [
            f"# {title}",
            ""
        ]

        # 添加URL（关键信息）
        if url:
            lines.extend([
                "**URL**:",
                f"{url}",
                ""
            ])

        # 添加发布日期和作者信息（如果有）
        if published_date != '未知' or author:
            lines.append("## 元数据")
            if published_date != '未知':
                lines.append(f"- 发布日期: {published_date}")
            if author:
                lines.append(f"- 作者: {author}")
            lines.append("")

        # 添加摘要（关键信息）
        if summary:
            lines.extend([
                "## 摘要",
                "---",
                summary,
                ""
            ])

        # 添加完整文本内容（如果有）
        if text:
            lines.extend([
                "## 完整内容",
                "---",
                text,
                ""
            ])

        # 添加相似度分数（如果有）
        if score is not None:
            lines.append(f"**相似度分数**: {score:.3f}")

        # 添加数据来源信息
        lines.extend([
            "",
            "---",
            f"*数据来源: Exa搜索 | 获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])

        return "\n".join(lines)

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """解析日期字符串"""
        if not date_str or not date_str.strip():
            return datetime.now()

        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            try:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                return datetime.now()

    def _parse_to_paper(self, result_dict: dict) -> Optional[Paper]:
        """将 Exa 结果解析为 Paper 对象"""
        url = result_dict.get('url', '')
        if not url:
            return None

        published_date = self._parse_date(result_dict.get('published_date'))

        return Paper(
            paper_id='',
            title=result_dict.get('title', '无标题'),
            authors=[],
            abstract=result_dict.get('summary', ''),
            doi='',
            published_date=published_date,
            pdf_url='',
            url=url,
            source='exa'
        )

    def search(
        self,
        query: str,
        num_results: int = None,
        type: str = "auto"
    ) -> List[Paper]:
        """
        搜索Exa API并返回Paper列表

        Args:
            query: 搜索关键词
            num_results: 结果数量，默认使用Config.EXA_NUM
            type: 搜索类型，默认'auto'

        Returns:
            List[Paper]: 解析后的 Paper 对象列表（已过滤 score < 0.8）
        """
        if num_results is None:
            num_results = self.max_results

        print(f"正在使用Exa搜索: {query}...")
        print(f"参数: num_results={num_results}, type={type}, summary=True")

        try:
            # 执行搜索 - 使用Summary模式
            results = self.client.search_and_contents(
                query,
                num_results=num_results,
                type=type,
                summary=True
            )

            # 打印结果数量
            result_count = len(results.results)
            print(f"\n{'='*60}")
            print(f"搜索完成，共找到 {result_count} 条结果")
            print(f"{'='*60}\n")

            # 解析并过滤结果
            papers = []
            skipped_count = 0
            for i, result in enumerate(results.results, 1):
                try:
                    # 转换为字典格式，提取所有关键信息
                    # 注意：exa_py API返回的属性是published_date而不是publishedDate
                    result_dict = {
                        'title': result.title if hasattr(result, 'title') else '',
                        'url': result.url if hasattr(result, 'url') else '',
                        'summary': result.summary if hasattr(result, 'summary') else '',
                        'text': result.text if hasattr(result, 'text') else '',
                        'score': result.score if hasattr(result, 'score') else None,
                        'published_date': result.published_date if hasattr(result, 'published_date') else '',
                        'author': result.author if hasattr(result, 'author') else ''
                    }

                    # 检查是否应该保存该结果（score过滤）
                    if not self._should_save_result(result_dict):
                        skipped_count += 1
                        score = result_dict.get('score')
                        print(f"[{i}/{result_count}] 跳过 (score={score}, url={result_dict.get('url', 'N/A')})")
                        continue

                    # 解析为 Paper 对象（注意：summary存储在abstract中）
                    paper = self._parse_to_paper(result_dict)
                    if not paper:
                        continue

                    papers.append(paper)

                except Exception as e:
                    print(f"[{i}/{result_count}] 解析失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\n{'='*60}")
            print(f"成功解析 {len(papers)} 个结果, 跳过 {skipped_count} 个结果")
            print(f"{'='*60}")

            return papers

        except Exception as e:
            print(f"\n搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def download(
        self,
        papers: List[Paper],
        save_path: str = None
    ) -> List[Paper]:
        """
        将Paper列表保存为Markdown文件

        Args:
            papers: Paper对象列表
            save_path: 保存路径，默认使用Config.DOC_SAVE_PATH

        Returns:
            List[Paper]: 成功下载的 Paper 对象列表
        """
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH

        # 确保保存目录存在
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        print(f"\n开始下载 {len(papers)} 个Paper到 {save_path}")

        downloaded = []
        for i, paper in enumerate(papers, 1):
            try:
                # 将Paper对象转换为字典格式（用于复用_generate_markdown）
                # 注意：paper.abstract 存储的是 summary
                result_dict = {
                    'title': paper.title,
                    'url': paper.url,
                    'summary': paper.abstract,  # abstract存储的是summary
                    'text': '',
                    'published_date': paper.published_date.isoformat() if paper.published_date else '',
                    'author': ''
                }

                # 生成文件名（与原search_and_save逻辑一致）
                title = paper.title.strip()
                if not title:
                    # 没有标题，使用"无标题" + 当前时间
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    title = f"无标题_{timestamp}"

                filename = f"exa_{self._sanitize_filename(title)}.md"
                file_path = os.path.join(save_path, filename)

                # 保存文件
                markdown_content = self._generate_markdown(result_dict)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                # 设置 extra["save_path"]
                if paper.extra is None:
                    paper.extra = {}
                paper.extra["save_path"] = file_path

                downloaded.append(paper)
                print(f"[{i}/{len(papers)}] 已保存: {file_path}")

            except Exception as e:
                print(f"[{i}/{len(papers)}] 保存失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*60}")
        print(f"成功保存 {len(downloaded)} 个文件")
        print(f"{'='*60}")

        return downloaded


# 使用示例
async def main():
    """测试ExaSearcher"""

    print("=" * 60)
    print("Exa搜索器测试")
    print("=" * 60)

    try:
        # 初始化搜索器
        searcher = ExaSearcherSummary()

        # 测试: 先搜索，再下载
        print("\n【测试】搜索并下载为markdown文件")
        print("-" * 40)
        query = "大模型的能力边界"

        # 第一步：搜索
        papers = searcher.search(
            query=query,
            num_results=4,
            type="auto",
        )
        print(f"\n返回的 Paper 对象数量: {len(papers)}")
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}] 标题: {paper.title}")
            print(f"    URL: {paper.url}")
            print(f"    摘要长度: {len(paper.abstract)} 字符")

        # 第二步：下载
        papers = searcher.download(papers)

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

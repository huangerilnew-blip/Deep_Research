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
from typing import List
from exa_py import Exa
from dotenv import load_dotenv

from core.config import Config

load_dotenv()


class ExaSearcher:
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
        - score 为空（不存在或None）或 score > 0.8
        - 并且 summary 不为空

        Args:
            result: 搜索结果字典

        Returns:
            bool: True表示应该保存
        """
        score = result.get('score')
        summary = result.get('summary', '')

        # 检查 score 条件：为空或 > 0.8
        score_condition = score is None or score > 0.8

        # 检查 summary 条件：不为空
        summary_condition = bool(summary.strip()) if summary else False

        return score_condition and summary_condition

    def _generate_markdown(self, result: dict) -> str:
        """
        将搜索结果转换为Markdown格式（简化版，只保存summary内容）

        Args:
            result: 搜索结果字典

        Returns:
            str: Markdown格式内容
        """
        summary = result.get('summary', '')

        lines = [
            f"# {result.get('title', '无标题')}",
            "",
            "## 摘要",
            "---",
            summary,
            "",
            "---",
            f"*数据来源: Exa搜索 | 获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ]

        return "\n".join(lines)

    def search_and_save(
        self,
        query: str,
        num_results: int =None,
        type: str = "auto",
        save_path: str = None
    ) -> List[dict]:
        """
        搜索并保存结果为markdown文件

        Args:
            query: 搜索关键词
            num_results: 结果数量，默认使用Config.EXA_NUM
            type: 搜索类型，默认'auto'
            save_path: 保存路径，默认使用Config.DOC_SAVE_PATH

        Returns:
            List[dict]: 搜索结果列表
        """
        if num_results is None:
            num_results = self.max_results

        if save_path is None:
            save_path = Config.DOC_SAVE_PATH

        # 确保保存目录存在
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        print(f"正在使用Exa搜索: {query}...")
        print(f"参数: num_results={num_results}, type={type}, summary=True")

        try:
            # 执行搜索
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

            # 保存每个结果为markdown文件
            saved_files = []
            skipped_count = 0
            for i, result in enumerate(results.results, 1):
                try:
                    # 转换为字典格式，只保留 title 和 summary
                    result_dict = {
                        'title': result.title if hasattr(result, 'title') else '',
                        'summary': result.summary if hasattr(result, 'summary') else '',
                        'score': result.score if hasattr(result, 'score') else None
                    }

                    # 检查是否应该保存该结果
                    if not self._should_save_result(result_dict):
                        skipped_count += 1
                        score = result_dict.get('score')
                        print(f"[{i}/{result_count}] 跳过 (score={score}, summary长度={len(result_dict.get('summary', ''))})")
                        continue

                    # 生成markdown内容
                    markdown_content = self._generate_markdown(result_dict)

                    # 生成文件名：exa_ + title
                    title = result_dict.get('title', '无标题')
                    filename = f"exa_{self._sanitize_filename(title)}.md"
                    file_path = os.path.join(save_path, filename)

                    # 保存文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)

                    saved_files.append(file_path)
                    print(f"[{i}/{result_count}] 已保存: {file_path}")

                except Exception as e:
                    print(f"[{i}/{result_count}] 保存失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\n{'='*60}")
            print(f"成功保存 {len(saved_files)} 个文件, 跳过 {skipped_count} 个结果")
            print(f"{'='*60}")

            return results.results

        except Exception as e:
            print(f"\n搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []


# 使用示例
async def main():
    """测试ExaSearcher"""

    print("=" * 60)
    print("Exa搜索器测试")
    print("=" * 60)

    try:
        # 初始化搜索器
        searcher = ExaSearcher()

        # 测试: 搜索并保存为markdown
        print("\n【测试】搜索并保存为markdown文件")
        print("-" * 40)
        query = "大模型的能力边界"
        results = searcher.search_and_save(
            query=query,
            num_results=4,
            type="auto",
        )

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

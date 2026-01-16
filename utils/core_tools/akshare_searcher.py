"""
AkShare 金融数据检索器

提供从 AkShare 获取金融数据并解析为 Paper 对象的功能。
支持个股新闻、财经资讯等数据类型的检索和导出。

AkShare 是一个开源的 Python 财经数据接口库，提供股票、期货、期权、
基金、债券、外汇、数字货币等金融数据。

使用示例::

    import asyncio
    from akshare_searcher import AkShareSearcher
    
    async def main():
        searcher = AkShareSearcher()
        papers = await searcher.search("603777", data_type="news", limit=10)
        papers = await searcher.download(papers)
    
    asyncio.run(main())

Author: AkShare Searcher
Version: 1.0.0
"""

import asyncio
import hashlib
import os
from datetime import datetime
from typing import List, Union, Optional

import akshare as ak

from config import Config
from paper import Paper


class AkShareSearcher:
    """
    AkShare 金融数据检索器
    
    用于从 AkShare 获取金融数据并解析为 Paper 对象。
    
    Attributes:
        max_results: 默认返回结果数量
    """
    
    def __init__(self, max_results: int = None):
        """
        初始化 AkShareSearcher
        
        Args:
            max_results: 最大返回数量，默认 20
        """
        self.max_results = max_results or 20

    async def search(
        self,
        symbol: str,
        data_type: str = "news",
        limit: int = None
    ) -> List[Paper]:
        """
        搜索金融数据并解析为 Paper 列表
        
        Args:
            symbol: 股票代码或关键词，如 "603777"
            data_type: 数据类型
                - "news": 个股新闻 (stock_news_em)
            limit: 结果数量限制
            
        Returns:
            List[Paper]: 解析后的 Paper 对象列表
        """
        max_results = limit or self.max_results
        
        print(f"正在使用 AkShare 获取数据: {symbol} (类型: {data_type})...")
        
        # 在线程池中运行同步的 akshare 调用
        loop = asyncio.get_event_loop()
        
        try:
            if data_type == "news":
                df = await loop.run_in_executor(
                    None,
                    lambda: ak.stock_news_em(symbol=symbol)
                )
                papers = self._parse_news(df, symbol, max_results)
            else:
                print(f"不支持的数据类型: {data_type}")
                papers = []
        except Exception as e:
            print(f"获取数据失败: {e}")
            return []
        
        print(f"解析完成，共 {len(papers)} 条记录")
        return papers

    def _parse_news(self, df, symbol: str, limit: int) -> List[Paper]:
        """
        解析个股新闻数据为 Paper 列表
        
        Args:
            df: AkShare 返回的 DataFrame
            symbol: 股票代码
            limit: 最大返回数量
            
        Returns:
            List[Paper]: Paper 对象列表
        """
        papers = []
        
        for _, row in df.head(limit).iterrows():
            try:
                paper = self._map_news_to_paper(row, symbol)
                papers.append(paper)
            except Exception as e:
                print(f"解析记录失败: {e}")
                continue
        
        return papers

    def _map_news_to_paper(self, row, symbol: str) -> Paper:
        """
        将单条新闻记录映射到 Paper 对象
        
        AkShare stock_news_em 返回字段:
        - 关键词: 搜索关键词
        - 新闻标题: 新闻标题
        - 新闻内容: 新闻摘要内容
        - 发布时间: 发布时间字符串
        - 文章来源: 来源网站
        - 新闻链接: 原文链接
        
        Args:
            row: DataFrame 的一行数据
            symbol: 股票代码
            
        Returns:
            Paper: 映射后的 Paper 对象
        """
        # 提取字段
        title = row.get("新闻标题", "")
        content = row.get("新闻内容", "")
        url = row.get("新闻链接", "")
        source = row.get("文章来源", "")
        keyword = row.get("关键词", symbol)
        pub_time_str = row.get("发布时间", "")
        
        # 生成 paper_id (URL 的 MD5 哈希)
        if url:
            paper_id = hashlib.md5(url.encode()).hexdigest()[:16]
        else:
            paper_id = f"akshare_{datetime.now().timestamp()}"
        
        # 解析发布时间
        published_date = self._parse_datetime(pub_time_str)
        
        return Paper(
            paper_id=paper_id,
            title=title,
            authors=[source] if source else [],
            abstract=content,
            doi="",
            published_date=published_date,
            pdf_url=None,
            url=url,
            source="akshare",
            keywords=[keyword, symbol] if keyword != symbol else [symbol],
            extra={
                "keyword": keyword,
                "news_source": source,
                "symbol": symbol
            }
        )

    def _parse_datetime(self, time_str: str) -> Optional[datetime]:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串，支持多种格式
            
        Returns:
            datetime: 解析后的时间对象，失败返回当前时间
        """
        if not time_str:
            return datetime.now()
        
        # 尝试多种时间格式
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(time_str), fmt)
            except ValueError:
                continue
        
        return datetime.now()

    async def download(
        self,
        papers: Union[Paper, List[Paper]],
        save_path: str = None,
        filename: str = None
    ) -> List[Paper]:
        """
        将 Paper 列表导出为 Markdown 文件
        
        Args:
            papers: 单个 Paper 或 Paper 列表
            save_path: 保存目录，默认 Config.DOC_SAVE_PATH
            filename: 文件名，默认使用时间戳生成
            
        Returns:
            List[Paper]: 更新了 saved_path 的 Paper 列表
        """
        # 处理单个 Paper 或 Paper 列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)
        
        if not paper_list:
            print("没有 Paper 需要下载。")
            return paper_list
        
        # 使用默认保存路径
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 尝试从第一个 paper 获取 symbol
            symbol = ""
            if paper_list[0].extra:
                symbol = paper_list[0].extra.get("symbol", "")
            if symbol:
                filename = f"akshare_{symbol}_{timestamp}.md"
            else:
                filename = f"akshare_search_{timestamp}.md"
        
        # 生成 Markdown 内容
        content = self._generate_markdown(paper_list)
        
        # 保存文件
        file_path = self._save_markdown(content, save_path, filename)
        
        # 更新每个 Paper 的 saved_path
        for paper in paper_list:
            if paper.extra is None:
                paper.extra = {}
            paper.extra["saved_path"] = file_path
        
        print(f"Markdown 文件已保存: {file_path}")
        return paper_list

    def _generate_markdown(self, papers: List[Paper]) -> str:
        """
        生成 Markdown 内容
        
        Args:
            papers: Paper 列表
            
        Returns:
            str: Markdown 格式字符串
        """
        lines = []
        
        # 标题和元信息
        lines.append("# AkShare 金融数据搜索结果\n")
        lines.append(f"**搜索时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**结果数量:** {len(papers)}\n")
        lines.append("---\n")
        
        # 每条记录
        for paper in papers:
            lines.append(self._paper_to_markdown(paper))
        
        return "\n".join(lines)

    def _paper_to_markdown(self, paper: Paper) -> str:
        """
        将单个 Paper 转换为 Markdown 格式
        
        Args:
            paper: Paper 对象
            
        Returns:
            str: Markdown 格式字符串
        """
        lines = []
        
        lines.append(f"## {paper.title}\n")
        
        # 来源信息
        if paper.authors:
            lines.append(f"**来源:** {paper.authors[0]}")
        
        # 发布时间
        if paper.published_date:
            lines.append(f"**发布时间:** {paper.published_date.strftime('%Y-%m-%d %H:%M')}")
        
        # 链接
        if paper.url:
            lines.append(f"**链接:** [{paper.url}]({paper.url})")
        
        # 关键词
        if paper.keywords:
            lines.append(f"**关键词:** {', '.join(paper.keywords)}")
        
        # 内容
        lines.append("\n### 内容")
        lines.append(paper.abstract if paper.abstract else "无内容")
        lines.append("\n---\n")
        
        return "\n".join(lines)

    def _save_markdown(self, content: str, save_path: str, filename: str) -> str:
        """
        保存 Markdown 内容到文件
        
        Args:
            content: Markdown 内容
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
    """
    AkShareSearcher 使用示例
    """
    searcher = AkShareSearcher()
    
    print("=" * 60)
    print("AkShare 金融数据检索器使用示例")
    print("=" * 60)
    
    # Step 1: 搜索个股新闻
    print("\n【示例】搜索个股新闻")
    print("-" * 40)
    
    symbol = "603777"
    papers = await searcher.search(symbol, data_type="news", limit=5)
    
    if not papers:
        print("未找到相关结果")
        return
    
    print(f"\n找到 {len(papers)} 条结果:")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p.title[:50]}...")
        print(f"   来源: {p.authors[0] if p.authors else 'N/A'}")
        print(f"   时间: {p.published_date.strftime('%Y-%m-%d %H:%M') if p.published_date else 'N/A'}")
    
    # Step 2: 下载为 Markdown（使用默认 Config.DOC_SAVE_PATH）
    print("\n【示例】导出为 Markdown")
    print("-" * 40)
    
    downloaded = await searcher.download(papers)
    
    for p in downloaded:
        print(f"文献: {p.title[:40]}...")
        print(f"保存位置: {p.extra.get('saved_path')}")
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

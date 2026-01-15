"""
SEC EDGAR 公司财务报告检索器

从美国证券交易委员会 EDGAR 数据库提取上市公司关键财务信息：
- 公司名称
- 风险因素 (Item 1A)
- 管理层讨论与分析 (Item 7)
- 财务快照 (收入、净利润、总资产等)

使用 SEC 官方免费 API，无需 API Key。

使用示例:
    >>> import asyncio
    >>> from sec_edgar import SECEdgarSearcher
    >>> 
    >>> async def main():
    ...     searcher = SECEdgarSearcher()
    ...     papers = await searcher.search("AAPL", limit=1)
    ...     for paper in papers:
    ...         print(f"{paper.title}")
    ...     papers = await searcher.download(papers)
    >>> 
    >>> asyncio.run(main())

Author: SEC EDGAR Searcher
Version: 1.0.0
"""

import os
import re
import httpx
from typing import List, Union, Optional, Dict
from datetime import datetime

from paper import Paper
from config import Config


class SECEdgarSearcher:
    """
    SEC EDGAR 公司财务报告检索器
    
    从 SEC EDGAR 数据库提取公司关键财务信息，映射到 Paper 数据结构。
    
    Attributes:
        user_agent: SEC 要求的 User-Agent 标识
        base_url: SEC API 基础 URL
        headers: HTTP 请求头
    """
    
    def __init__(self, user_agent: str = None):
        """
        初始化 SECEdgarSearcher
        
        Args:
            user_agent: SEC 要求的 User-Agent，格式: "CompanyName email@example.com"
                       默认从 Config.SEC_EDGAR_USER_AGENT 读取
        """
        self.user_agent = user_agent or Config.SEC_EDGAR_USER_AGENT
        self.base_url = "https://data.sec.gov"
        self.sec_url = "https://www.sec.gov"
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        # 缓存公司代码映射
        self._ticker_cache: Optional[Dict] = None


    async def _load_ticker_mapping(self, client: httpx.AsyncClient) -> Dict:
        """
        加载股票代码到 CIK 的映射表
        
        Returns:
            Dict: {ticker: {"cik": str, "name": str}, ...}
        """
        if self._ticker_cache is not None:
            return self._ticker_cache
        
        url = f"{self.sec_url}/files/company_tickers.json"
        try:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # 构建映射: ticker -> {cik, name}
            self._ticker_cache = {}
            for item in data.values():
                ticker = item.get("ticker", "").upper()
                cik = str(item.get("cik_str", "")).zfill(10)
                name = item.get("title", "")
                if ticker:
                    self._ticker_cache[ticker] = {"cik": cik, "name": name}
            
            return self._ticker_cache
        except Exception as e:
            print(f"加载股票代码映射失败: {e}")
            return {}

    async def _get_cik_by_ticker(
        self, 
        client: httpx.AsyncClient, 
        query: str
    ) -> Optional[Dict]:
        """
        通过股票代码或公司名称获取 CIK
        
        Args:
            client: HTTP 客户端
            query: 股票代码或公司名称
            
        Returns:
            Dict: {"cik": str, "name": str, "ticker": str} 或 None
        """
        ticker_map = await self._load_ticker_mapping(client)
        if not ticker_map:
            return None
        
        query_upper = query.upper().strip()
        
        # 精确匹配股票代码
        if query_upper in ticker_map:
            info = ticker_map[query_upper]
            return {"cik": info["cik"], "name": info["name"], "ticker": query_upper}
        
        # 模糊匹配公司名称
        query_lower = query.lower()
        for ticker, info in ticker_map.items():
            if query_lower in info["name"].lower():
                return {"cik": info["cik"], "name": info["name"], "ticker": ticker}
        
        return None

    async def _get_company_submissions(
        self, 
        client: httpx.AsyncClient, 
        cik: str
    ) -> Optional[Dict]:
        """
        获取公司提交历史
        
        Args:
            client: HTTP 客户端
            cik: 公司 CIK（10位，含前导零）
            
        Returns:
            Dict: 公司提交信息，包含 name, filings 等
        """
        url = f"{self.base_url}/submissions/CIK{cik}.json"
        try:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print("警告: SEC API 限流，请稍后重试")
            else:
                print(f"获取公司提交历史失败: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"获取公司提交历史失败: {e}")
            return None


    def _get_latest_10k_info(self, submissions: Dict) -> Optional[Dict]:
        """
        从提交历史中获取最新 10-K 文件信息
        
        Args:
            submissions: 公司提交历史数据
            
        Returns:
            Dict: {"accession_number": str, "filing_date": str, "primary_document": str}
        """
        filings = submissions.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accession_numbers = filings.get("accessionNumber", [])
        filing_dates = filings.get("filingDate", [])
        primary_documents = filings.get("primaryDocument", [])
        
        # 查找最新的 10-K
        for i, form in enumerate(forms):
            if form == "10-K":
                return {
                    "accession_number": accession_numbers[i].replace("-", ""),
                    "accession_number_raw": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_documents[i]
                }
        
        # 如果没有 10-K，尝试 10-K/A（修正版）
        for i, form in enumerate(forms):
            if form == "10-K/A":
                return {
                    "accession_number": accession_numbers[i].replace("-", ""),
                    "accession_number_raw": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_documents[i]
                }
        
        return None

    async def _fetch_filing_html(
        self, 
        client: httpx.AsyncClient, 
        cik: str, 
        filing_info: Dict
    ) -> Optional[str]:
        """
        下载 10-K HTML 文件
        
        Args:
            client: HTTP 客户端
            cik: 公司 CIK
            filing_info: 文件信息
            
        Returns:
            str: HTML 内容
        """
        accession = filing_info["accession_number"]
        primary_doc = filing_info["primary_document"]
        
        # 构建 URL: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}
        url = f"{self.sec_url}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"
        
        try:
            response = await client.get(url, headers=self.headers, timeout=60.0)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            print(f"下载 10-K 文件失败: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"下载 10-K 文件失败: {e}")
            return None


    def _clean_html(self, html: str) -> str:
        """
        清理 HTML 标签，提取纯文本
        
        Args:
            html: HTML 内容
            
        Returns:
            str: 清理后的纯文本
        """
        # 移除 script 和 style 标签及其内容
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除所有 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 处理 HTML 实体
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        text = re.sub(r'&#\d+;', '', text)
        # 清理多余空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_risk_factors(self, html: str) -> str:
        """
        提取风险因素章节 (Item 1A)
        
        Args:
            html: 10-K HTML 内容
            
        Returns:
            str: 风险因素文本
        """
        # 多种模式匹配 Item 1A
        patterns = [
            # 模式1: Item 1A. Risk Factors 到 Item 1B
            r'(?:ITEM\s*1A\.?\s*[-–—]?\s*RISK\s*FACTORS)(.*?)(?:ITEM\s*1B\.?\s*[-–—]?\s*UNRESOLVED)',
            # 模式2: Item 1A 到 Item 1B（简化）
            r'(?:ITEM\s*1A[.\s])(.*?)(?:ITEM\s*1B[.\s])',
            # 模式3: Risk Factors 标题到下一个 Item
            r'(?:>RISK\s*FACTORS<)(.*?)(?:>ITEM\s*\d)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1)
                return self._clean_html(content)
        
        # 如果都没匹配到，尝试更宽松的匹配
        match = re.search(r'RISK\s*FACTORS(.*?)(?:ITEM\s*\d|PROPERTIES|LEGAL\s*PROCEEDINGS)', 
                         html, re.DOTALL | re.IGNORECASE)
        if match:
            return self._clean_html(match.group(1))
        
        return ""

    def _extract_mda(self, html: str) -> str:
        """
        提取管理层讨论与分析章节 (Item 7)
        
        Args:
            html: 10-K HTML 内容
            
        Returns:
            str: MD&A 文本
        """
        # 多种模式匹配 Item 7
        patterns = [
            # 模式1: Item 7. MD&A 到 Item 7A
            r'(?:ITEM\s*7\.?\s*[-–—]?\s*MANAGEMENT\'?S?\s*DISCUSSION)(.*?)(?:ITEM\s*7A\.?\s*[-–—]?\s*QUANTITATIVE)',
            # 模式2: Item 7 到 Item 7A（简化）
            r'(?:ITEM\s*7[.\s])(.*?)(?:ITEM\s*7A[.\s])',
            # 模式3: MD&A 到 Item 8
            r'(?:MANAGEMENT\'?S?\s*DISCUSSION\s*AND\s*ANALYSIS)(.*?)(?:ITEM\s*8[.\s])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1)
                return self._clean_html(content)
        
        # 更宽松的匹配
        match = re.search(r'MANAGEMENT\'?S?\s*DISCUSSION(.*?)(?:ITEM\s*\d|FINANCIAL\s*STATEMENTS)', 
                         html, re.DOTALL | re.IGNORECASE)
        if match:
            return self._clean_html(match.group(1))
        
        return ""

    def _summarize_text(self, text: str, max_length: int = 3000) -> str:
        """
        精简文本，适合大模型输入
        
        保留前 max_length 个字符，在句子边界截断
        
        Args:
            text: 原始文本
            max_length: 最大长度
            
        Returns:
            str: 精简后的文本
        """
        if not text or len(text) <= max_length:
            return text
        
        # 在 max_length 附近找句子结束位置
        truncated = text[:max_length]
        
        # 尝试在句号、问号、感叹号处截断
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('? '),
            truncated.rfind('! ')
        )
        
        if last_period > max_length * 0.7:  # 至少保留 70% 内容
            return truncated[:last_period + 1].strip() + "..."
        
        return truncated.strip() + "..."


    async def _get_financial_snapshot(
        self, 
        client: httpx.AsyncClient, 
        cik: str
    ) -> Dict:
        """
        从 XBRL API 获取财务快照
        
        Args:
            client: HTTP 客户端
            cik: 公司 CIK
            
        Returns:
            Dict: 财务快照数据
        """
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json"
        
        try:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"获取 XBRL 数据失败: {e}")
            return {}
        
        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})
        
        def get_latest_value(concept_name: str) -> Optional[float]:
            """获取指定概念的最新值"""
            concept = us_gaap.get(concept_name, {})
            units = concept.get("units", {})
            
            # 优先获取 USD 单位的值
            values = units.get("USD", [])
            if not values:
                # 尝试其他单位
                for unit_values in units.values():
                    if unit_values:
                        values = unit_values
                        break
            
            if not values:
                return None
            
            # 获取最新的年度值（10-K）
            annual_values = [v for v in values if v.get("form") == "10-K"]
            if annual_values:
                # 按 end 日期排序，取最新
                annual_values.sort(key=lambda x: x.get("end", ""), reverse=True)
                return annual_values[0].get("val")
            
            # 如果没有年度值，取最新的任意值
            values.sort(key=lambda x: x.get("end", ""), reverse=True)
            return values[0].get("val") if values else None
        
        def get_latest_period(concept_name: str) -> Optional[str]:
            """获取指定概念的最新报告期"""
            concept = us_gaap.get(concept_name, {})
            units = concept.get("units", {})
            values = units.get("USD", [])
            
            if not values:
                return None
            
            annual_values = [v for v in values if v.get("form") == "10-K"]
            if annual_values:
                annual_values.sort(key=lambda x: x.get("end", ""), reverse=True)
                return annual_values[0].get("end")
            
            return None
        
        # 提取关键财务指标
        # 收入可能有多种概念名称
        revenue = (
            get_latest_value("Revenues") or
            get_latest_value("RevenueFromContractWithCustomerExcludingAssessedTax") or
            get_latest_value("SalesRevenueNet") or
            get_latest_value("RevenueFromContractWithCustomerIncludingAssessedTax")
        )
        
        # 净利润
        net_income = (
            get_latest_value("NetIncomeLoss") or
            get_latest_value("ProfitLoss")
        )
        
        # 总资产
        total_assets = get_latest_value("Assets")
        
        # 总负债
        total_liabilities = (
            get_latest_value("Liabilities") or
            get_latest_value("LiabilitiesAndStockholdersEquity")
        )
        
        # 股东权益
        stockholders_equity = (
            get_latest_value("StockholdersEquity") or
            get_latest_value("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
        )
        
        # 获取报告期
        period = get_latest_period("Assets") or get_latest_period("Revenues")
        
        return {
            "revenue": revenue,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "stockholders_equity": stockholders_equity,
            "currency": "USD",
            "period": period
        }


    def _map_to_paper(
        self,
        company_info: Dict,
        filing_info: Dict,
        risk_factors: str,
        mda: str,
        financial_snapshot: Dict
    ) -> Paper:
        """
        将 SEC 数据映射到 Paper 对象
        
        Args:
            company_info: 公司信息 {cik, name, ticker}
            filing_info: 文件信息 {accession_number, filing_date, ...}
            risk_factors: 风险因素文本
            mda: MD&A 文本
            financial_snapshot: 财务快照
            
        Returns:
            Paper: 映射后的 Paper 对象
        """
        # 生成摘要（精简版，适合大模型）
        risk_summary = self._summarize_text(risk_factors, 1500)
        mda_summary = self._summarize_text(mda, 1500)
        
        abstract_parts = []
        if risk_summary:
            abstract_parts.append(f"【风险因素摘要】{risk_summary}")
        if mda_summary:
            abstract_parts.append(f"【管理层讨论与分析摘要】{mda_summary}")
        
        abstract = "\n\n".join(abstract_parts) if abstract_parts else "暂无摘要信息"
        
        # 解析日期
        filing_date = filing_info.get("filing_date", "")
        published_date = None
        if filing_date:
            try:
                published_date = datetime.strptime(filing_date, "%Y-%m-%d")
            except ValueError:
                pass
        
        # 构建 URL
        cik = company_info.get("cik", "").lstrip("0")
        accession = filing_info.get("accession_number", "")
        primary_doc = filing_info.get("primary_document", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
        
        # 构建 extra
        extra = {
            "cik": company_info.get("cik", ""),
            "ticker": company_info.get("ticker", ""),
            "form_type": "10-K",
            "fiscal_year": filing_date[:4] if filing_date else "",
            "risk_factors": risk_factors,
            "mda": mda,
            "financial_snapshot": financial_snapshot
        }
        
        return Paper(
            paper_id=filing_info.get("accession_number_raw", accession),
            title=company_info.get("name", ""),
            authors=["SEC EDGAR"],
            abstract=abstract,
            doi=filing_info.get("accession_number_raw", ""),
            published_date=published_date,
            pdf_url="",  # SEC 文件通常是 HTML
            url=url,
            source="sec_edgar",
            updated_date=None,
            categories=[],
            keywords=[company_info.get("ticker", "")],
            citations=0,
            references=[],
            extra=extra
        )

    async def search(self, query: str, limit: int = 5) -> List[Paper]:
        """
        搜索公司并提取关键财务信息
        
        Args:
            query: 公司名称或股票代码
            limit: 最大返回数量（目前每次查询返回一家公司）
            
        Returns:
            List[Paper]: 包含关键财务信息的 Paper 对象列表
        """
        papers = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. 获取公司 CIK
            company_info = await self._get_cik_by_ticker(client, query)
            if not company_info:
                print(f"未找到公司: {query}")
                return []
            
            cik = company_info["cik"]
            print(f"找到公司: {company_info['name']} (CIK: {cik}, Ticker: {company_info['ticker']})")
            
            # 2. 获取公司提交历史
            submissions = await self._get_company_submissions(client, cik)
            if not submissions:
                return []
            
            # 更新公司名称（使用 SEC 官方名称）
            company_info["name"] = submissions.get("name", company_info["name"])
            
            # 3. 获取最新 10-K 信息
            filing_info = self._get_latest_10k_info(submissions)
            if not filing_info:
                print(f"未找到 10-K 文件: {company_info['name']}")
                return []
            
            print(f"找到 10-K 文件: {filing_info['accession_number_raw']} ({filing_info['filing_date']})")
            
            # 4. 下载并解析 10-K HTML
            html = await self._fetch_filing_html(client, cik, filing_info)
            
            risk_factors = ""
            mda = ""
            if html:
                risk_factors = self._extract_risk_factors(html)
                mda = self._extract_mda(html)
                print(f"提取风险因素: {len(risk_factors)} 字符")
                print(f"提取 MD&A: {len(mda)} 字符")
            
            # 5. 获取财务快照
            financial_snapshot = await self._get_financial_snapshot(client, cik)
            if financial_snapshot.get("revenue"):
                print(f"获取财务快照: 收入 ${financial_snapshot['revenue']:,.0f}")
            
            # 6. 映射到 Paper
            paper = self._map_to_paper(
                company_info,
                filing_info,
                risk_factors,
                mda,
                financial_snapshot
            )
            papers.append(paper)
        
        return papers[:limit]


    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除或替换非法字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            str: 清理后的安全文件名
        """
        if not filename:
            return "unnamed"
        
        # Windows 和 Unix 禁止的字符
        forbidden_chars = '<>:"/\\|?*\x00\n\r\t'
        
        result = filename
        for char in forbidden_chars:
            result = result.replace(char, '_')
        
        # 移除前后空格
        result = result.strip()
        
        # 如果结果为空，返回默认名称
        if not result:
            return "unnamed"
        
        # 限制文件名长度
        if len(result) > 200:
            result = result[:200]
        
        return result

    def _generate_markdown(self, paper: Paper) -> str:
        """
        将 Paper 对象转换为 Markdown 格式
        
        Args:
            paper: Paper 对象
            
        Returns:
            str: Markdown 格式字符串
        """
        lines = []
        extra = paper.extra or {}
        
        # 标题
        lines.append(f"# {paper.title}")
        lines.append("")
        
        # 基本信息
        lines.append("## 基本信息")
        lines.append("")
        lines.append(f"- **来源**: SEC EDGAR")
        lines.append(f"- **股票代码**: {extra.get('ticker', 'N/A')}")
        lines.append(f"- **CIK**: {extra.get('cik', 'N/A')}")
        lines.append(f"- **报告类型**: {extra.get('form_type', '10-K')}")
        if paper.published_date:
            lines.append(f"- **提交日期**: {paper.published_date.strftime('%Y-%m-%d')}")
        lines.append(f"- **报告链接**: {paper.url}")
        lines.append("")
        
        # 财务快照
        lines.append("## 财务快照")
        lines.append("")
        snapshot = extra.get("financial_snapshot", {})
        
        def format_number(value, prefix="$"):
            if value is None:
                return "N/A"
            if abs(value) >= 1e9:
                return f"{prefix}{value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"{prefix}{value/1e6:.2f}M"
            else:
                return f"{prefix}{value:,.0f}"
        
        lines.append(f"- **收入**: {format_number(snapshot.get('revenue'))}")
        lines.append(f"- **净利润**: {format_number(snapshot.get('net_income'))}")
        lines.append(f"- **总资产**: {format_number(snapshot.get('total_assets'))}")
        lines.append(f"- **总负债**: {format_number(snapshot.get('total_liabilities'))}")
        lines.append(f"- **股东权益**: {format_number(snapshot.get('stockholders_equity'))}")
        if snapshot.get("period"):
            lines.append(f"- **报告期**: {snapshot.get('period')}")
        lines.append("")
        
        # 风险因素
        lines.append("## 风险因素 (Item 1A)")
        lines.append("")
        risk_factors = extra.get("risk_factors", "")
        if risk_factors:
            # 限制长度，避免文件过大
            if len(risk_factors) > 10000:
                risk_factors = risk_factors[:10000] + "\n\n... (内容已截断)"
            lines.append(risk_factors)
        else:
            lines.append("暂无风险因素信息")
        lines.append("")
        
        # 管理层讨论与分析
        lines.append("## 管理层讨论与分析 (Item 7)")
        lines.append("")
        mda = extra.get("mda", "")
        if mda:
            if len(mda) > 10000:
                mda = mda[:10000] + "\n\n... (内容已截断)"
            lines.append(mda)
        else:
            lines.append("暂无管理层讨论与分析信息")
        lines.append("")
        
        # 页脚
        lines.append("---")
        lines.append(f"*数据来源: SEC EDGAR | 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)

    async def _save_file(
        self,
        paper: Paper,
        markdown_content: str,
        save_path: str
    ) -> str:
        """
        保存 Markdown 文件
        
        Args:
            paper: Paper 对象
            markdown_content: Markdown 内容
            save_path: 保存目录
            
        Returns:
            str: 文件路径或错误信息
        """
        try:
            # 生成文件名：公司名称_股票代码.md
            ticker = paper.extra.get("ticker", "") if paper.extra else ""
            if ticker:
                filename = f"{self._sanitize_filename(paper.title)}_{ticker}.md"
            elif paper.paper_id:
                filename = f"{self._sanitize_filename(paper.paper_id)}.md"
            else:
                filename = f"sec_edgar_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            
            file_path = os.path.join(save_path, filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            return file_path
            
        except PermissionError as e:
            error_msg = f"Error: 权限不足 - {e}"
            print(error_msg)
            return error_msg
        except OSError as e:
            error_msg = f"Error: 文件系统错误 - {e}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: 保存失败 - {e}"
            print(error_msg)
            return error_msg

    async def download(
        self,
        papers: Union[Paper, List[Paper]],
        save_path: str = None
    ) -> List[Paper]:
        """
        将 Paper 对象导出为 Markdown 文件
        
        Args:
            papers: Paper 对象或列表
            save_path: 保存路径，默认使用 Config.DOC_SAVE_PATH
            
        Returns:
            List[Paper]: 更新了 saved_path 的 Paper 列表
        """
        # 处理输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)
        
        # 使用默认路径
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH
        
        # 确保目录存在
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as e:
                print(f"创建目录失败: {e}")
                for paper in paper_list:
                    if paper.extra is None:
                        paper.extra = {}
                    paper.extra["saved_path"] = f"Error: 无法创建目录 - {e}"
                return paper_list
        
        # 保存每个 Paper
        for paper in paper_list:
            markdown_content = self._generate_markdown(paper)
            saved_path = await self._save_file(paper, markdown_content, save_path)
            
            if paper.extra is None:
                paper.extra = {}
            paper.extra["saved_path"] = saved_path
            
            if not saved_path.startswith("Error:"):
                print(f"导出完成: {paper.title[:50]}... -> {saved_path}")
        
        return paper_list



async def main():
    """
    SECEdgarSearcher 使用示例
    
    展示 search 和 download 方法的使用方式
    """
    searcher = SECEdgarSearcher()
    
    print("=" * 60)
    print("SEC EDGAR 公司财务报告检索器使用示例")
    print("=" * 60)
    
    # 检查 User-Agent
    if not searcher.user_agent or "example" in searcher.user_agent.lower():
        print("\n⚠️  警告: 请配置有效的 User-Agent")
        print("SEC 要求提供格式: 'CompanyName email@example.com'")
        print("可通过环境变量 SEC_EDGAR_USER_AGENT 设置")
        print("\n以下示例将使用模拟数据演示功能...")
        
        # 使用模拟数据演示
        demo_paper = Paper(
            paper_id="0000320193-24-000123",
            title="Apple Inc.",
            authors=["SEC EDGAR"],
            abstract="【风险因素摘要】公司面临激烈的市场竞争、供应链风险、汇率波动等风险...\n\n【管理层讨论与分析摘要】本财年公司实现收入增长，主要得益于服务业务的强劲表现...",
            doi="0000320193-24-000123",
            published_date=datetime(2024, 11, 1),
            pdf_url="",
            url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
            source="sec_edgar",
            categories=[],
            keywords=["AAPL"],
            extra={
                "cik": "0000320193",
                "ticker": "AAPL",
                "form_type": "10-K",
                "fiscal_year": "2024",
                "risk_factors": "公司面临以下主要风险：\n\n1. 市场竞争风险：智能手机、平板电脑和个人电脑市场竞争激烈...\n\n2. 供应链风险：公司依赖全球供应链，可能受到地缘政治、自然灾害等因素影响...\n\n3. 技术变革风险：技术快速变化可能导致产品过时...",
                "mda": "管理层讨论与分析：\n\n本财年公司总收入达到 3943 亿美元，同比增长 2%。服务业务收入创历史新高，达到 850 亿美元。iPhone 收入保持稳定，Mac 和 iPad 业务有所下滑。\n\n展望未来，公司将继续投资人工智能和增强现实技术...",
                "financial_snapshot": {
                    "revenue": 394328000000,
                    "net_income": 96995000000,
                    "total_assets": 352583000000,
                    "total_liabilities": 290437000000,
                    "stockholders_equity": 62146000000,
                    "currency": "USD",
                    "period": "2024-09-28"
                }
            }
        )
        
        print("\n【演示】使用模拟数据导出 Markdown")
        print("-" * 40)
        
        downloaded = await searcher.download(demo_paper, save_path="./downloads")
        
        for paper in downloaded:
            saved_path = paper.extra.get("saved_path", "")
            if saved_path and not saved_path.startswith("Error:"):
                print(f"✓ 导出成功: {paper.title}")
                print(f"  保存路径: {saved_path}")
            else:
                print(f"✗ 导出失败: {paper.title}")
                print(f"  错误: {saved_path}")
        
        print("\n" + "=" * 60)
        print("演示完成")
        print("=" * 60)
        return
    
    # ========== 示例: 搜索并导出 ==========
    print("\n【示例】搜索公司并导出财务报告")
    print("-" * 40)
    
    query = "AAPL"  # 可以是股票代码或公司名称
    print(f"搜索: {query}")
    
    papers = await searcher.search(query, limit=1)
    
    if papers:
        paper = papers[0]
        print(f"\n公司: {paper.title}")
        print(f"股票代码: {paper.extra.get('ticker', 'N/A')}")
        print(f"提交日期: {paper.published_date}")
        
        # 显示财务快照
        snapshot = paper.extra.get("financial_snapshot", {})
        print(f"\n财务快照:")
        if snapshot.get("revenue"):
            print(f"  收入: ${snapshot['revenue']:,.0f}")
        if snapshot.get("net_income"):
            print(f"  净利润: ${snapshot['net_income']:,.0f}")
        if snapshot.get("total_assets"):
            print(f"  总资产: ${snapshot['total_assets']:,.0f}")
        
        # 导出为 Markdown
        print(f"\n导出为 Markdown...")
        downloaded = await searcher.download(papers, save_path="./downloads")
        
        for p in downloaded:
            saved_path = p.extra.get("saved_path", "")
            if saved_path and not saved_path.startswith("Error:"):
                print(f"✓ 导出成功: {saved_path}")
            else:
                print(f"✗ 导出失败: {saved_path}")
    else:
        print("未找到公司信息")
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

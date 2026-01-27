"""
Crunchbase 公司信息检索器

提供从 Crunchbase 数据库检索公司元数据并导出为 Markdown 文档的功能。
遵循与 PubMedSearcher、OpenAlexSearcher、SemanticScholarSearcher 相同的设计模式，
提供独立的 search 和 download 方法。

Crunchbase 是全球领先的商业信息平台，提供公司、投资、融资等数据。
API 需要 API Key 进行访问。

使用示例:
    >>> import asyncio
    >>> from crunchbase import CrunchbaseSearcher
    >>> 
    >>> async def main():
    ...     searcher = CrunchbaseSearcher(api_key="your-api-key")
    ...     # 检索公司
    ...     papers = await searcher.search("artificial intelligence", limit=5)
    ...     for paper in papers:
    ...         print(f"{paper.title} - {paper.abstract[:50]}...")
    ...     # 导出为 Markdown
    ...     papers = await searcher.download(papers)
    ...     for paper in papers:
    ...         print(f"Saved to: {paper.extra.get('saved_path')}")
    >>> 
    >>> asyncio.run(main())

主要功能:
    - search: 根据关键词检索 Crunchbase 公司信息，返回 Paper 对象列表
    - download: 将 Paper 对象导出为 Markdown 文件

Author: Crunchbase Searcher
Version: 1.0.0
"""

import os
import re
import httpx
from typing import List, Union, Optional
from datetime import datetime

from tools.core_tools.paper import Paper
from core.config import Config


class CrunchbaseSearcher:
    """
    Crunchbase 公司信息检索器类
    
    用于从 Crunchbase 数据库检索公司元数据并导出为 Markdown 文档。
    
    Attributes:
        api_key: Crunchbase API 密钥
        base_url: Crunchbase API 基础 URL
        headers: HTTP 请求头
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化 CrunchbaseSearcher
        
        Args:
            api_key: API 密钥，默认从 Config.CRUNCHBASE_API_KEY 读取
        """
        self.api_key = api_key or Config.CRUNCHBASE_API_KEY
        self.base_url = "https://api.crunchbase.com/v4/data"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CrunchbaseSearcher/1.0"
        }


    async def _search_organizations(
        self, 
        client: httpx.AsyncClient, 
        query: str, 
        limit: int
    ) -> List[dict]:
        """
        调用 Crunchbase Search API 搜索公司
        
        使用 POST /v4/data/searches/organizations 端点
        
        Args:
            client: HTTP 客户端
            query: 检索关键词
            limit: 最大返回数量
            
        Returns:
            List[dict]: Crunchbase Organization 对象列表
        """
        url = f"{self.base_url}/searches/organizations"
        params = {"user_key": self.api_key}
        
        # 构建请求体
        body = {
            "field_ids": [
                "identifier",
                "short_description",
                "founded_on",
                "funding_total",
                "num_funding_rounds",
                "last_funding_type",
                "num_employees_enum",
                "categories",
                "location_identifiers",
                "website_url",
                "linkedin_url",
                "rank_org_company",
                "revenue_range",
                "operating_status",
                "last_funding_at"
            ],
            "query": [
                {
                    "type": "predicate",
                    "field_id": "identifier",
                    "operator_id": "contains",
                    "values": [query]
                }
            ],
            "limit": limit
        }
        
        try:
            response = await client.post(
                url, 
                params=params, 
                json=body, 
                headers=self.headers
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("entities", [])
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"警告: API 限流，请稍后重试")
            elif e.response.status_code == 401:
                print(f"错误: API 密钥无效或未提供")
            elif e.response.status_code == 403:
                print(f"错误: 无权访问此资源")
            else:
                print(f"HTTP 错误: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            print(f"网络请求错误: {e}")
            return []
        except Exception as e:
            print(f"解析响应失败: {e}")
            return []


    def _map_to_paper(self, org_data: dict) -> Paper:
        """
        将 Crunchbase 组织数据映射到 Paper 类
        
        映射规则：
        - identifier.value (公司名称) → title
        - short_description → abstract
        - ["Crunchbase"] → authors (固定值)
        - funding_total, num_funding_rounds 等 → extra["financials"]
        - categories, location_identifiers → extra["related_data"]
        - "crunchbase" → source
        
        Args:
            org_data: Crunchbase Organization 对象（字典）
            
        Returns:
            Paper: 转换后的 Paper 对象
        """
        properties = org_data.get("properties", {})
        
        # 解析 paper_id (uuid)
        paper_id = org_data.get("uuid", "")
        
        # 解析 title (公司名称)
        identifier = properties.get("identifier", {})
        title = identifier.get("value", "") if isinstance(identifier, dict) else ""
        permalink = identifier.get("permalink", "") if isinstance(identifier, dict) else ""
        
        # 解析 abstract (业务描述)
        abstract = properties.get("short_description", "") or ""
        
        # authors 固定为 ["Crunchbase"]
        authors = ["Crunchbase"]
        
        # 解析 published_date (成立日期)
        founded_on = properties.get("founded_on")
        published_date = None
        if founded_on:
            try:
                published_date = datetime.strptime(founded_on, "%Y-%m-%d")
            except ValueError:
                try:
                    published_date = datetime.strptime(founded_on[:4], "%Y")
                except ValueError:
                    pass
        
        # 解析 url (公司网站)
        url = properties.get("website_url", "") or ""
        
        # 解析 categories
        categories_data = properties.get("categories", []) or []
        categories = []
        for cat in categories_data:
            if isinstance(cat, dict) and cat.get("value"):
                categories.append(cat["value"])
            elif isinstance(cat, str):
                categories.append(cat)
        
        # 解析 citations (公司排名)
        citations = properties.get("rank_org_company", 0) or 0
        
        # 构建 financials 信息
        funding_total = properties.get("funding_total", {})
        financials = {
            "funding_total": funding_total.get("value") if isinstance(funding_total, dict) else None,
            "funding_currency": funding_total.get("currency", "USD") if isinstance(funding_total, dict) else "USD",
            "num_funding_rounds": properties.get("num_funding_rounds", 0) or 0,
            "last_funding_type": properties.get("last_funding_type", "") or "",
            "last_funding_at": properties.get("last_funding_at", "") or "",
            "num_employees_enum": properties.get("num_employees_enum", "") or "",
            "revenue_range": properties.get("revenue_range", "") or "",
            "operating_status": properties.get("operating_status", "") or "",
            "rank_org_company": citations
        }
        
        # 构建 related_data 信息
        location_data = properties.get("location_identifiers", []) or []
        locations = []
        for loc in location_data:
            if isinstance(loc, dict) and loc.get("value"):
                locations.append(loc["value"])
            elif isinstance(loc, str):
                locations.append(loc)
        
        related_data = {
            "locations": locations,
            "linkedin_url": properties.get("linkedin_url", "") or "",
            "permalink": permalink
        }
        
        # 构建 extra
        extra = {
            "financials": financials,
            "related_data": related_data
        }
        
        return Paper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            doi=permalink,  # 使用 permalink 作为唯一标识
            published_date=published_date,
            pdf_url="",  # Crunchbase 无 PDF
            url=url,
            source="crunchbase",
            updated_date=None,
            categories=categories,
            keywords=[],
            citations=citations,
            references=[],
            extra=extra
        )


    async def search(self, query: str, limit: int = 5) -> List[Paper]:
        """
        根据查询关键词检索 Crunchbase 公司信息
        
        Args:
            query: 检索关键词
            limit: 最大返回数量，默认 5
            
        Returns:
            List[Paper]: 符合条件的 Paper 对象列表，包含公司元信息
        """
        papers = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 调用 _search_organizations 获取组织列表
            organizations = await self._search_organizations(client, query, limit)
            
            # 遍历调用 _map_to_paper 转换为 Paper 列表
            for org in organizations:
                try:
                    paper = self._map_to_paper(org)
                    papers.append(paper)
                except Exception as e:
                    print(f"解析公司数据失败: {e}")
                    continue
        
        return papers

    def _sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除或替换非法字符
        
        处理 Windows 和 Unix 文件系统的限制
        
        Args:
            filename: 原始文件名
            
        Returns:
            str: 清理后的安全文件名
        """
        if not filename:
            return "unnamed"
        
        # Windows 禁止的字符: < > : " / \ | ? *
        # 同时移除控制字符
        forbidden_chars = '<>:"/\\|?*\x00\n\r\t'
        
        result = filename
        for char in forbidden_chars:
            result = result.replace(char, '_')
        
        # 移除前后空格
        result = result.strip()
        
        # 如果结果为空，返回默认名称
        if not result:
            return "unnamed"
        
        # 限制文件名长度（Windows 最大 255 字符）
        if len(result) > 200:
            result = result[:200]
        
        return result

    def _generate_markdown(self, paper: Paper) -> str:
        """
        将 Paper 对象转换为 Markdown 格式字符串
        
        Args:
            paper: Paper 对象
            
        Returns:
            str: Markdown 格式的字符串
        """
        lines = []
        
        # 标题
        lines.append(f"# {paper.title}")
        lines.append("")
        
        # 基本信息
        lines.append("## 基本信息")
        lines.append("")
        lines.append(f"- **来源**: Crunchbase")
        if paper.published_date:
            lines.append(f"- **成立日期**: {paper.published_date.strftime('%Y-%m-%d')}")
        if paper.url:
            lines.append(f"- **网站**: {paper.url}")
        
        # 从 extra 获取 LinkedIn
        related_data = paper.extra.get("related_data", {}) if paper.extra else {}
        if related_data.get("linkedin_url"):
            lines.append(f"- **LinkedIn**: {related_data['linkedin_url']}")
        if related_data.get("permalink"):
            lines.append(f"- **Crunchbase**: https://www.crunchbase.com/organization/{related_data['permalink']}")
        lines.append("")
        
        # 业务概况
        lines.append("## 业务概况")
        lines.append("")
        lines.append(paper.abstract if paper.abstract else "暂无描述")
        lines.append("")
        
        # 财务信息
        lines.append("## 财务信息")
        lines.append("")
        financials = paper.extra.get("financials", {}) if paper.extra else {}
        
        funding_total = financials.get("funding_total")
        if funding_total:
            currency = financials.get("funding_currency", "USD")
            lines.append(f"- **总融资额**: {funding_total:,} {currency}")
        else:
            lines.append("- **总融资额**: 未公开")
        
        num_rounds = financials.get("num_funding_rounds", 0)
        lines.append(f"- **融资轮次**: {num_rounds}")
        
        last_funding_type = financials.get("last_funding_type", "")
        if last_funding_type:
            lines.append(f"- **最近融资类型**: {last_funding_type}")
        
        last_funding_at = financials.get("last_funding_at", "")
        if last_funding_at:
            lines.append(f"- **最近融资时间**: {last_funding_at}")
        
        num_employees = financials.get("num_employees_enum", "")
        if num_employees:
            # 转换员工数量枚举为可读格式
            employee_map = {
                "c_1_10": "1-10",
                "c_11_50": "11-50",
                "c_51_100": "51-100",
                "c_101_250": "101-250",
                "c_251_500": "251-500",
                "c_501_1000": "501-1000",
                "c_1001_5000": "1001-5000",
                "c_5001_10000": "5001-10000",
                "c_10001_plus": "10000+"
            }
            readable = employee_map.get(num_employees, num_employees)
            lines.append(f"- **员工规模**: {readable}")
        
        rank = financials.get("rank_org_company", 0)
        if rank:
            lines.append(f"- **公司排名**: {rank:,}")
        
        operating_status = financials.get("operating_status", "")
        if operating_status:
            status_map = {"active": "运营中", "closed": "已关闭", "acquired": "已被收购"}
            readable_status = status_map.get(operating_status, operating_status)
            lines.append(f"- **运营状态**: {readable_status}")
        
        revenue_range = financials.get("revenue_range", "")
        if revenue_range:
            lines.append(f"- **收入范围**: {revenue_range}")
        lines.append("")
        
        # 分类与标签
        lines.append("## 分类与标签")
        lines.append("")
        if paper.categories:
            lines.append(", ".join(paper.categories))
        else:
            lines.append("暂无分类")
        lines.append("")
        
        # 地理位置
        lines.append("## 地理位置")
        lines.append("")
        locations = related_data.get("locations", [])
        if locations:
            lines.append(", ".join(locations))
        else:
            lines.append("未知")
        lines.append("")
        
        # 页脚
        lines.append("---")
        lines.append(f"*数据来源: Crunchbase | 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)

    async def _save_file(
        self, 
        paper: Paper, 
        markdown_content: str, 
        save_path: str
    ) -> str:
        """
        保存 Markdown 文件到指定路径
        
        Args:
            paper: Paper 对象
            markdown_content: Markdown 内容
            save_path: 保存目录
            
        Returns:
            str: 文件保存路径或错误信息
        """
        try:
            # 生成安全的文件名
            if paper.paper_id:
                filename = f"{self._sanitize_filename(paper.paper_id)}.md"
            elif paper.title:
                filename = f"{self._sanitize_filename(paper.title)}.md"
            else:
                filename = f"company_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
            
            file_path = os.path.join(save_path, filename)
            
            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            return file_path
            
        except PermissionError as e:
            error_msg = f"Error: 权限不足，无法写入文件 - {e}"
            print(error_msg)
            return error_msg
        except OSError as e:
            error_msg = f"Error: 文件系统错误 - {e}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: 保存文件失败 - {e}"
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
            papers: 单个 Paper 对象或 Paper 列表
            save_path: 保存路径，默认使用 Config.DOC_SAVE_PATH
            
        Returns:
            List[Paper]: 更新后的 Paper 列表，包含保存路径信息
        """
        # 处理单个 Paper 或 Paper 列表输入
        if isinstance(papers, Paper):
            paper_list = [papers]
        else:
            paper_list = list(papers)
        
        # 使用默认保存路径
        if save_path is None:
            save_path = Config.DOC_SAVE_PATH
        
        # 确保保存目录存在
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as e:
                print(f"创建目录失败: {e}")
                # 为所有 paper 设置错误信息
                for paper in paper_list:
                    if paper.extra is None:
                        paper.extra = {}
                    paper.extra["saved_path"] = f"Error: 无法创建目录 - {e}"
                return paper_list
        
        # 遍历 Paper 列表，生成 Markdown 并保存
        for paper in paper_list:
            # 生成 Markdown 内容
            markdown_content = self._generate_markdown(paper)
            
            # 保存文件
            saved_path = await self._save_file(paper, markdown_content, save_path)
            
            # 更新 Paper.extra["saved_path"]
            if paper.extra is None:
                paper.extra = {}
            paper.extra["saved_path"] = saved_path
            
            print(f"导出完成: {paper.title[:50]}... -> {saved_path}")
        
        return paper_list



async def main():
    """
    CrunchbaseSearcher 使用示例
    
    展示 search 和 download 方法的独立使用方式：
    1. 单独使用 search 方法检索公司信息
    2. 单独使用 download 方法导出已有 Paper 对象为 Markdown
    3. 组合使用 search + download 完成完整的检索导出流程
    """
    # 创建检索器实例
    # 需要提供有效的 Crunchbase API Key
    searcher = CrunchbaseSearcher()
    
    print("=" * 60)
    print("Crunchbase 公司信息检索器使用示例")
    print("=" * 60)
    
    # 检查 API Key
    if not searcher.api_key:
        print("\n⚠️  警告: 未配置 Crunchbase API Key")
        print("请设置环境变量 CRUNCHBASE_API_KEY 或在初始化时传入 api_key 参数")
        print("示例: searcher = CrunchbaseSearcher(api_key='your-api-key')")
        print("\n以下示例将使用模拟数据演示功能...")
        
        # 使用模拟数据演示
        demo_paper = Paper(
            paper_id="demo-uuid-12345",
            title="OpenAI",
            authors=["Crunchbase"],
            abstract="OpenAI is an AI research and deployment company dedicated to ensuring that artificial general intelligence benefits all of humanity.",
            doi="openai",
            published_date=datetime(2015, 12, 11),
            pdf_url="",
            url="https://openai.com",
            source="crunchbase",
            categories=["Artificial Intelligence", "Machine Learning", "Software"],
            extra={
                "financials": {
                    "funding_total": 11300000000,
                    "funding_currency": "USD",
                    "num_funding_rounds": 7,
                    "last_funding_type": "secondary_market",
                    "num_employees_enum": "c_501_1000",
                    "rank_org_company": 15,
                    "operating_status": "active"
                },
                "related_data": {
                    "locations": ["San Francisco", "California", "United States"],
                    "linkedin_url": "https://linkedin.com/company/openai",
                    "permalink": "openai"
                }
            }
        )
        
        print("\n【演示】使用模拟数据导出 Markdown")
        print("-" * 40)
        
        # 导出为 Markdown
        downloaded = await searcher.download(demo_paper, save_path="./downloads")
        
        for paper in downloaded:
            saved_path = paper.extra.get("saved_path", "")
            if saved_path and not saved_path.startswith("Error:"):
                print(f"✓ 导出成功: {paper.title}")
                print(f"  保存路径: {saved_path}")
                
                # 显示生成的 Markdown 内容预览
                print(f"\n  Markdown 内容预览:")
                print("-" * 40)
                markdown = searcher._generate_markdown(paper)
                # 只显示前 500 个字符
                preview = markdown[:500] + "..." if len(markdown) > 500 else markdown
                for line in preview.split("\n"):
                    print(f"  {line}")
            else:
                print(f"✗ 导出失败: {paper.title}")
                print(f"  错误: {saved_path}")
        
        print("\n" + "=" * 60)
        print("演示完成")
        print("=" * 60)
        return
    
    # ========== 示例 1: 单独使用 search 方法 ==========
    print("\n【示例 1】单独使用 search 方法检索公司")
    print("-" * 40)
    
    query = "artificial intelligence"
    papers = await searcher.search(query, limit=3)
    
    print(f"检索关键词: {query}")
    print(f"检索到 {len(papers)} 家公司:\n")
    
    for i, paper in enumerate(papers, 1):
        print(f"[{i}] {paper.title}")
        print(f"    业务概况: {paper.abstract[:80]}..." if paper.abstract else "    业务概况: N/A")
        print(f"    成立日期: {paper.published_date.strftime('%Y-%m-%d') if paper.published_date else 'N/A'}")
        print(f"    网站: {paper.url or 'N/A'}")
        
        financials = paper.extra.get("financials", {})
        funding = financials.get("funding_total")
        if funding:
            print(f"    总融资额: ${funding:,}")
        print(f"    公司排名: {paper.citations or 'N/A'}")
        print()
    
    # ========== 示例 2: 单独使用 download 方法 ==========
    print("\n【示例 2】单独使用 download 方法导出 Markdown")
    print("-" * 40)
    
    if papers:
        print(f"将 {len(papers)} 家公司信息导出为 Markdown...")
        
        # 导出为 Markdown（使用默认保存路径）
        downloaded_papers = await searcher.download(papers)
        
        for paper in downloaded_papers:
            saved_path = paper.extra.get("saved_path", "")
            if saved_path and not saved_path.startswith("Error:"):
                print(f"✓ 导出成功: {paper.title[:50]}...")
                print(f"  保存路径: {saved_path}")
            else:
                print(f"✗ 导出失败: {paper.title[:50]}...")
    else:
        print("没有可导出的公司信息")
    
    # ========== 示例 3: 组合使用 search + download ==========
    print("\n【示例 3】组合使用 search + download 完成完整流程")
    print("-" * 40)
    
    query2 = "fintech"
    print(f"检索关键词: {query2}")
    
    papers2 = await searcher.search(query2, limit=2)
    print(f"检索到 {len(papers2)} 家公司")
    
    if papers2:
        # 指定自定义保存路径
        custom_save_path = "./downloads/fintech"
        downloaded = await searcher.download(papers2, save_path=custom_save_path)
        
        print(f"\n导出结果 (保存到 {custom_save_path}):")
        for paper in downloaded:
            saved_path = paper.extra.get("saved_path", "")
            status = "✓ 成功" if saved_path and not saved_path.startswith("Error:") else "✗ 失败"
            print(f"  {status}: {paper.title[:40]}...")
    
    # ========== 示例 4: 访问 Paper 对象的详细信息 ==========
    print("\n【示例 4】访问 Paper 对象的详细信息")
    print("-" * 40)
    
    if papers:
        paper = papers[0]
        print(f"公司详细信息:")
        print(f"  paper_id: {paper.paper_id}")
        print(f"  title (公司名称): {paper.title}")
        print(f"  authors: {paper.authors}")
        print(f"  abstract (业务概况): {paper.abstract[:100]}..." if paper.abstract else "  abstract: N/A")
        print(f"  doi (permalink): {paper.doi}")
        print(f"  published_date (成立日期): {paper.published_date}")
        print(f"  url (网站): {paper.url}")
        print(f"  source: {paper.source}")
        print(f"  categories: {paper.categories[:5] if paper.categories else []}")
        print(f"  citations (排名): {paper.citations}")
        
        # 显示财务信息
        financials = paper.extra.get("financials", {})
        print(f"\n  财务信息:")
        print(f"    总融资额: {financials.get('funding_total', 'N/A')}")
        print(f"    融资轮次: {financials.get('num_funding_rounds', 'N/A')}")
        print(f"    最近融资类型: {financials.get('last_funding_type', 'N/A')}")
        print(f"    员工规模: {financials.get('num_employees_enum', 'N/A')}")
        
        # 显示关联数据
        related = paper.extra.get("related_data", {})
        print(f"\n  关联数据:")
        print(f"    地理位置: {related.get('locations', [])}")
        print(f"    LinkedIn: {related.get('linkedin_url', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

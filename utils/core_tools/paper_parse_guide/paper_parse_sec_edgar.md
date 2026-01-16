# SEC EDGAR 数据解析到 Paper 类说明

本文档描述 `sec_edgar.py` 中如何将 SEC EDGAR 公司财务报告数据解析映射到 Paper 类。

## 数据来源

SEC EDGAR 是美国证券交易委员会的电子数据收集、分析和检索系统，提供上市公司的财务报告（10-K 年报、20-F 外国公司年报等）。

## 字段映射表

| Paper 字段 | SEC EDGAR 来源 | 解析逻辑 |
|-----------|---------------|---------|
| `paper_id` | `filing_info.accession_number_raw` | 使用 SEC 文件编号（如 `0000320193-24-000123`） |
| `title` | `company_info.name` | 公司名称（SEC 官方注册名称） |
| `authors` | - | 固定为 `["SEC EDGAR"]` |
| `abstract` | 多个章节组合 | 组合企业介绍、风险因素、MD&A 的精简摘要 |
| `doi` | `filing_info.accession_number_raw` | 使用文件编号作为唯一标识 |
| `published_date` | `filing_info.filing_date` | 解析 `YYYY-MM-DD` 格式的提交日期 |
| `pdf_url` | - | 空字符串（SEC 文件通常是 HTML 格式） |
| `url` | 构建 | `https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}` |
| `source` | - | 固定为 `"sec_edgar"` |
| `updated_date` | - | `None` |
| `categories` | - | 空列表 |
| `keywords` | `company_info.ticker` | 股票代码列表 |
| `citations` | - | `0` |
| `references` | - | 空列表 |

---

## Extra 字段详解

`extra` 字段用于存储 SEC EDGAR 特有的公司财务信息，是该数据源最重要的扩展数据。

### Extra 字段结构

```python
extra = {
    "cik": str,                    # 公司 CIK 编号
    "ticker": str,                 # 股票代码
    "form_type": str,              # 报告类型
    "fiscal_year": str,            # 财年
    "business_description": str,   # 企业介绍完整文本
    "risk_factors": str,           # 风险因素完整文本
    "mda": str,                    # 管理层讨论与分析完整文本
    "financial_snapshot": dict,    # 财务快照数据
    "saved_path": str              # 下载后的本地保存路径
}
```

### 各字段说明

| Extra 字段 | 来源 | 类型 | 说明 |
|-----------|------|------|------|
| `cik` | SEC 公司映射表 | `str` | 中央索引键（Central Index Key），SEC 唯一公司标识，10位数字含前导零 |
| `ticker` | SEC 公司映射表 | `str` | 股票代码，如 `"AAPL"`, `"MSFT"`, `"BABA"` |
| `form_type` | 提交历史 | `str` | 报告类型：`"10-K"`（美国公司年报）或 `"20-F"`（外国公司年报） |
| `fiscal_year` | `filing_date` | `str` | 财年，从提交日期提取年份 |
| `business_description` | Item 1 - Business | `str` | 企业介绍章节完整文本（HTML 清理后） |
| `risk_factors` | Item 1A - Risk Factors | `str` | 风险因素章节完整文本 |
| `mda` | Item 7 - MD&A | `str` | 管理层讨论与分析章节完整文本 |
| `financial_snapshot` | XBRL API | `dict` | 财务快照数据（见下表） |
| `saved_path` | - | `str` | 调用 `download()` 后填充，Markdown 文件保存路径 |

### Financial Snapshot 财务快照结构

```python
financial_snapshot = {
    "revenue": float,              # 收入（美元）
    "net_income": float,           # 净利润（美元）
    "total_assets": float,         # 总资产（美元）
    "total_liabilities": float,    # 总负债（美元）
    "stockholders_equity": float,  # 股东权益（美元）
    "currency": str,               # 货币单位，固定 "USD"
    "period": str                  # 报告期，如 "2024-09-28"
}
```

### 财务指标来源映射

| 指标 | US-GAAP 概念名称（按优先级） |
|-----|---------------------------|
| `revenue` | `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `SalesRevenueNet` |
| `net_income` | `NetIncomeLoss` → `ProfitLoss` |
| `total_assets` | `Assets` |
| `total_liabilities` | `Liabilities` → `LiabilitiesAndStockholdersEquity` |
| `stockholders_equity` | `StockholdersEquity` → `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` |

### 使用示例

```python
# 获取公司基本信息
ticker = paper.extra.get("ticker")
cik = paper.extra.get("cik")
form_type = paper.extra.get("form_type")

# 获取财务数据
snapshot = paper.extra.get("financial_snapshot", {})
revenue = snapshot.get("revenue")
net_income = snapshot.get("net_income")

if revenue:
    print(f"收入: ${revenue:,.0f}")
if net_income:
    profit_margin = net_income / revenue * 100 if revenue else 0
    print(f"净利润率: {profit_margin:.2f}%")

# 获取完整文本内容
risk_factors = paper.extra.get("risk_factors", "")
if risk_factors:
    print(f"风险因素字数: {len(risk_factors)}")

# 检查下载路径
saved_path = paper.extra.get("saved_path")
if saved_path and not saved_path.startswith("Error:"):
    print(f"文件已保存至: {saved_path}")
```

---

## 特殊处理逻辑

### 1. 公司查找流程

```
1. 加载 SEC 股票代码映射表 (company_tickers.json)
2. 精确匹配股票代码（如 "AAPL"）
3. 若无匹配，模糊匹配公司名称
4. 返回 {cik, name, ticker}
```

### 2. 年报类型优先级

```python
# 支持的年报类型，按优先级排序
annual_forms = ["10-K", "20-F", "10-K/A", "20-F/A"]

# 10-K: 美国公司年报
# 20-F: 外国公司年报（中概股等）
# /A 后缀: 修订版本
```

### 3. 章节提取正则模式

```python
# Item 1 - Business（企业介绍）
r'(?:ITEM\s*1\.?\s*[-–—]?\s*BUSINESS)(.*?)(?:ITEM\s*1A\.?\s*[-–—]?\s*RISK)'

# Item 1A - Risk Factors（风险因素）
r'(?:ITEM\s*1A\.?\s*[-–—]?\s*RISK\s*FACTORS)(.*?)(?:ITEM\s*1B\.?\s*[-–—]?\s*UNRESOLVED)'

# Item 7 - MD&A（管理层讨论与分析）
r'(?:ITEM\s*7\.?\s*[-–—]?\s*MANAGEMENT\'?S?\s*DISCUSSION)(.*?)(?:ITEM\s*7A\.?\s*[-–—]?\s*QUANTITATIVE)'
```

### 4. 摘要生成逻辑

```python
# abstract 由三部分组成，每部分最多 1500 字符
abstract_parts = [
    f"【企业介绍】{business_summary}",      # Item 1 精简版
    f"【风险因素摘要】{risk_summary}",       # Item 1A 精简版
    f"【管理层讨论与分析摘要】{mda_summary}" # Item 7 精简版
]
abstract = "\n\n".join(abstract_parts)
```

---

## 完整映射代码参考

```python
def _map_to_paper(
    self,
    company_info: Dict,
    filing_info: Dict,
    business_description: str,
    risk_factors: str,
    mda: str,
    financial_snapshot: Dict
) -> Paper:
    return Paper(
        paper_id=filing_info.get("accession_number_raw", ""),
        title=company_info.get("name", ""),
        authors=["SEC EDGAR"],
        abstract=self._generate_abstract(business_description, risk_factors, mda),
        doi=filing_info.get("accession_number_raw", ""),
        published_date=parse_date(filing_info.get("filing_date")),
        pdf_url="",
        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}",
        source="sec_edgar",
        updated_date=None,
        categories=[],
        keywords=[company_info.get("ticker", "")],
        citations=0,
        references=[],
        extra={
            "cik": company_info.get("cik", ""),
            "ticker": company_info.get("ticker", ""),
            "form_type": filing_info.get("form_type", "10-K"),
            "fiscal_year": filing_date[:4],
            "business_description": business_description,
            "risk_factors": risk_factors,
            "mda": mda,
            "financial_snapshot": financial_snapshot
        }
    )
```

明白了，我来简化需求文档，聚焦于核心功能：提取关键信息供大模型使用。

好的，让我详细说明解析这些关键信息的思路：

## SEC EDGAR 数据解析思路

### 1. 公司名称获取

**数据源**: `https://data.sec.gov/submissions/CIK{cik}.json`

SEC 官方 API 返回的 JSON 中包含 `name` 字段，直接获取公司名称。也可以通过 `https://www.sec.gov/files/company_tickers.json` 获取股票代码与公司名称的映射。

### 2. 风险因素（Risk Factors - Item 1A）

**解析思路**:

- 10-K/10-Q 报告是 HTML 格式，风险因素在 **Item 1A** 章节
- 通过正则表达式或 HTML 解析定位 `Item 1A` 到 `Item 1B` 之间的内容
- 常见标识：`<b>Item 1A</b>`、`ITEM 1A`、`Risk Factors` 等
- **精简策略**：提取前 2000-3000 字符作为摘要，或提取主要风险标题列表

### 3. 管理层讨论与分析（MD&A - Item 7）

**解析思路**:

- 位于 **Item 7** 章节，到 Item 7A 或 Item 8 之间
- 常见标识：`Item 7`、`Management's Discussion and Analysis`
- **精简策略**：提取概述段落（通常前几段是总结），或提取关键财务指标讨论部分

### 4. 财务快照（Financial Snapshot）

**数据源**: SEC XBRL API `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`

这个 API 返回结构化的财务数据，可以直接提取：

- **收入 (Revenue)**: `us-gaap:Revenues` 或 `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`
- **净利润 (Net Income)**: `us-gaap:NetIncomeLoss`
- **总资产 (Total Assets)**: `us-gaap:Assets`
- **总负债 (Total Liabilities)**: `us-gaap:Liabilities`
- **股东权益 (Stockholders Equity)**: `us-gaap:StockholdersEquity`

### 整体流程

```
1. 用户输入公司名称/股票代码
   ↓
2. 查询 company_tickers.json 获取 CIK
   ↓
3. 调用 submissions API 获取公司名称和最新 10-K 文件信息
   ↓
4. 下载 10-K HTML 文件，解析 Item 1A 和 Item 7
   ↓
5. 调用 XBRL API 获取财务数据
   ↓
6. 组装到 Paper 对象
```

这个思路是否清晰？如果没问题，我们可以继续进行设计阶段。
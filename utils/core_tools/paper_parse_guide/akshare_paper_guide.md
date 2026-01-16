# Paper 属性说明（AkShare金融数据）

## 核心属性

| 属性 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| paper_id | str | 股票代码 | `"000001"` |
| title | str | 公司全称 | `"平安银行股份有限公司"` |
| authors | List[str] | 固定值 | `["AkShare"]` |
| abstract | str | 主营业务（截取150字） | `"办理人民币存、贷、结算..."` |
| doi | str | 空 | `""` |
| published_date | datetime | 上市日期 | `1991-04-03` |
| pdf_url | str | 空 | `None` |
| url | str | 公司官网或东财详情页 | `"https://bank.pingan.com"` |
| source | str | 固定值 | `"akshare"` |
| categories | List[str] | 所属行业 | `["银行"]` |

---

## extra 三维度结构

### 维度1: basic_info（基本信息）

| 字段 | 说明 | 示例 |
|------|------|------|
| 股票代码 | 6位代码 | `"000001"` |
| 股票简称 | 简称 | `"平安银行"` |
| 公司全称 | 全称 | `"平安银行股份有限公司"` |
| 所属行业 | 行业分类 | `"银行"` |
| 上市日期 | 上市时间 | `"1991-04-03"` |
| 成立日期 | 成立时间 | `"1987-12-22"` |
| 官方网站 | 官网域名 | `"bank.pingan.com"` |

### 维度2: company_intro（公司简介）

| 字段 | 说明 | 示例 |
|------|------|------|
| 主营业务 | 业务描述（150字） | `"办理人民币存、贷、结算..."` |
| 法人代表 | 法人 | `"谢永林"` |
| 注册地址 | 地址 | `"广东省深圳市罗湖区..."` |

### 维度3: financial_info（财务信息）

| 字段 | 说明 | 单位 | 示例 |
|------|------|------|------|
| 总市值 | 总市值 | 元 | `217152224635.62` |
| 流通市值 | 流通市值 | 元 | `217148671307.07` |
| 总股本 | 总股本 | 股 | `19405918198.0` |
| 流通股 | 流通股 | 股 | `19405735198.0` |
| 注册资金 | 注册资本 | 万元 | `1940591.8198` |
| 最新价* | 当前价格 | 元 | `11.19` |
| 涨跌幅* | 涨跌幅 | % | `0.45` |
| 市盈率* | PE | - | `5.12` |
| 市净率* | PB | - | `0.58` |

> *标记字段仅在 `include_spot=True` 时有值

---

## 便捷方法

```python
paper.to_llm_text()      # 生成LLM友好的精简文本
paper.get_basic_info()   # 获取基本信息字典
paper.get_company_intro() # 获取公司简介字典
paper.get_financial_info() # 获取财务信息字典
paper.to_dict()          # 转换为完整字典
```

---

## 使用示例

```python
from akshare_searcher import AkShareSearcher
import asyncio

async def main():
    searcher = AkShareSearcher()
    papers = await searcher.search("000001")
    paper = papers[0]
    
    # 核心属性
    print(paper.title)       # 平安银行股份有限公司
    print(paper.paper_id)    # 000001
    print(paper.categories)  # ['银行']
    
    # 三维度数据
    print(paper.get_basic_info()["上市日期"])      # 1991-04-03
    print(paper.get_company_intro()["法人代表"])   # 谢永林
    print(paper.get_financial_info()["总市值"])    # 217152224635.62
    
    # 导出
    await searcher.download(papers)

asyncio.run(main())
```

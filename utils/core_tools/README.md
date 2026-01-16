arXiv, bioRxiv, medRxiv, PubMed, SSRN,IACR

## 概要获取

- **OpenAlex (最全的元数据层)**：它是目前全球最完整的学术图谱之一，几乎包含了 arXiv、bioRxiv、medRxiv、PubMed、SSRN 以及大量 Semantic Scholar 中的内容。
  - OpenAlex 的能力:以优秀的元数据提供给开发者，并提供相应的pdf下载连接
- **Semantic Scholar (AI增强层)**：由艾伦人工智能研究所（AI2）开发，它不仅有论文，还有基于 AI 的摘要、引文影响力分析。
  - 有着优秀的摘要能力，提供pdf下载
- **建议**：在项目初期，可以将 OpenAlex 作为主索引，Semantic Scholar 作为补充；

### 公司信息

- **Crunchbase (创业公司/投融资)**：侧重于初创公司信息、融资轮次、高管变动。
  - 尝试用it桔子代替Crunchbase，但是it桔子和Crunchbase同样都存在信息不易获取及api_key收费极高的问题
  - 采用RAG方式结合Tavily的实时搜索能力，来弥补Crunchbase信息信息的问题

- **SEC Edgar (美国证券交易委员会官网)**：侧重于美国上市公司的官方财务报表（10-K, 10-Q）和合规文件。

### 实时搜索

- Tavily 可能会抓取到 arXiv 或 Crunchbase 的网页版内容。但 Tavily 的优势在于**实时性**（如昨天的发布会或政策变动），而前述数据库优势在于**深度结构化数据**。

## 搜索网页总结

- **Perplexity**：直接读完前 10-20 个网页，然后写出一份带“角标注释”的总结报告。

  




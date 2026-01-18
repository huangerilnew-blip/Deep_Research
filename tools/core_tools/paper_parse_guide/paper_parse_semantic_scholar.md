# Semantic Scholar 数据解析到 Paper 类说明

本文档描述 `semantic_scholar.py` 中如何将 Semantic Scholar 论文数据解析映射到 Paper 类。

## 数据来源

Semantic Scholar 是由 Allen Institute for AI 开发的免费学术搜索引擎，包含超过 2 亿篇学术论文，提供丰富的元数据、引用分析和开放获取 PDF 链接。

## 字段映射表

| Paper 字段 | Semantic Scholar 字段 | 解析逻辑 |
|-----------|---------------------|---------|
| `paper_id` | `paperId` | 直接使用 Semantic Scholar 论文 ID（40位哈希） |
| `title` | `title` | 直接使用 |
| `authors` | `authors[].name` | 遍历 authors 数组，提取每个作者的 name |
| `abstract` | `abstract` | 直接使用，可能为空 |
| `doi` | `externalIds.DOI` | 从 externalIds 字典中提取 DOI |
| `published_date` | `publicationDate` 或 `year` | 优先解析完整日期，否则使用年份构建 |
| `pdf_url` | `openAccessPdf.url` | 从 openAccessPdf 对象提取 URL |
| `url` | `url` | Semantic Scholar 论文页面链接 |
| `source` | - | 固定为 `"semanticscholar"` |
| `updated_date` | - | `None`（API 不提供此字段） |
| `categories` | `s2FieldsOfStudy[].category` 或 `fieldsOfStudy` | 优先从 s2FieldsOfStudy 提取，否则用 fieldsOfStudy |
| `keywords` | - | 空列表（API 不提供关键词字段） |
| `citations` | `citationCount` | 直接使用，默认 0 |
| `references` | `referenceCount` | 存储为列表格式 `[str(count)]` |

---

## Extra 字段详解

`extra` 字段用于存储 Semantic Scholar 特有的学术元数据。

### Extra 字段结构

```python
extra = {
    "venue": str,                    # 发表场所
    "year": int,                     # 发表年份
    "publicationTypes": list,        # 出版类型列表
    "isOpenAccess": bool,            # 是否开放获取
    "influentialCitationCount": int, # 有影响力的引用数
    "externalIds": dict,             # 外部标识符集合
    "corpusId": int,                 # Semantic Scholar 语料库 ID
    "publicationVenue": dict,        # 发表场所详细信息
    "journal": dict,                 # 期刊信息
    "openAccessPdf": dict,           # 开放获取 PDF 信息
    "saved_path": str                # 下载后的本地保存路径
}
```

### 各字段说明

| Extra 字段 | Semantic Scholar 来源 | 类型 | 说明 |
|-----------|---------------------|------|------|
| `venue` | `venue` | `str` | 发表场所名称，如期刊名、会议名 |
| `year` | `year` | `int` | 发表年份，如 `2023` |
| `publicationTypes` | `publicationTypes` | `list` | 出版类型列表（见下表） |
| `isOpenAccess` | `isOpenAccess` | `bool` | 是否为开放获取论文 |
| `influentialCitationCount` | `influentialCitationCount` | `int` | 有影响力的引用数（高质量引用） |
| `externalIds` | `externalIds` | `dict` | 外部标识符集合（见下表） |
| `corpusId` | `corpusId` | `int` | Semantic Scholar 内部语料库 ID |
| `publicationVenue` | `publicationVenue` | `dict` | 发表场所详细信息，包含 id、name、type 等 |
| `journal` | `journal` | `dict` | 期刊信息，包含 name、volume、pages 等 |
| `openAccessPdf` | `openAccessPdf` | `dict` | 开放获取 PDF 信息，包含 url、status |
| `saved_path` | - | `str` | 调用 `download()` 后填充，PDF 文件保存路径 |

### Publication Types 出版类型值

| 类型值 | 含义 |
|-------|------|
| `JournalArticle` | 期刊文章 |
| `Conference` | 会议论文 |
| `Review` | 综述文章 |
| `Book` | 书籍 |
| `BookSection` | 书籍章节 |
| `Dataset` | 数据集 |
| `Patent` | 专利 |

### External IDs 外部标识符

```python
externalIds = {
    "DOI": str,           # Digital Object Identifier
    "ArXiv": str,         # arXiv 预印本 ID
    "PubMed": str,        # PubMed ID (PMID)
    "PubMedCentral": str, # PubMed Central ID (PMC)
    "DBLP": str,          # DBLP 数据库 ID
    "MAG": str,           # Microsoft Academic Graph ID
    "CorpusId": int       # Semantic Scholar Corpus ID
}
```

### Open Access PDF 结构

```python
openAccessPdf = {
    "url": str,     # PDF 下载链接
    "status": str   # 状态，如 "GREEN", "GOLD", "HYBRID"
}
```

### 使用示例

```python
# 检查是否为开放获取
if paper.extra.get("isOpenAccess"):
    print("该论文可免费获取全文")

# 获取有影响力的引用数
influential_citations = paper.extra.get("influentialCitationCount", 0)
total_citations = paper.citations
if total_citations > 0:
    influence_ratio = influential_citations / total_citations
    print(f"高质量引用占比: {influence_ratio:.2%}")

# 获取外部标识符
external_ids = paper.extra.get("externalIds", {})
arxiv_id = external_ids.get("ArXiv")
if arxiv_id:
    print(f"arXiv 链接: https://arxiv.org/abs/{arxiv_id}")

pubmed_id = external_ids.get("PubMed")
if pubmed_id:
    print(f"PubMed 链接: https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}")

# 获取发表场所
venue = paper.extra.get("venue", "")
pub_venue = paper.extra.get("publicationVenue", {})
if pub_venue:
    venue_type = pub_venue.get("type", "")  # "journal" 或 "conference"
    print(f"发表于 {venue} ({venue_type})")

# 检查下载路径
saved_path = paper.extra.get("saved_path")
if saved_path and saved_path != "No fulltext available":
    print(f"PDF 已保存至: {saved_path}")
```

---

## 特殊处理逻辑

### 1. PDF 链接提取 (`_extract_pdf_url`)

```python
def _extract_pdf_url(self, paper_data: dict) -> str:
    open_access_pdf = paper_data.get("openAccessPdf")
    if open_access_pdf and isinstance(open_access_pdf, dict):
        return open_access_pdf.get("url", "")
    return ""
```

### 2. 发布日期解析

```python
# 优先级：
# 1. publicationDate (YYYY-MM-DD 格式)
# 2. year (仅年份，构建为 YYYY-01-01)

pub_date_str = paper_data.get("publicationDate")
if pub_date_str:
    published_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
elif paper_data.get("year"):
    published_date = datetime(int(paper_data["year"]), 1, 1)
```

### 3. 研究领域提取

```python
# 优先从 s2FieldsOfStudy 提取（更精确）
s2_fields = paper_data.get("s2FieldsOfStudy", [])
if s2_fields:
    categories = [field["category"] for field in s2_fields if field.get("category")]
else:
    # 回退到 fieldsOfStudy
    categories = paper_data.get("fieldsOfStudy", [])

# 去重
categories = list(dict.fromkeys(categories))
```

### 4. 多源 PDF 下载策略

```python
# 下载优先级：
# 1. openAccessPdf.url（原始链接）
# 2. arXiv 直链（如果有 ArXiv ID）
# 3. Unpaywall API 查询（通过 DOI）
# 4. PubMed Central（如果有 PMC ID）

urls_to_try = []
if paper.pdf_url:
    urls_to_try.append(paper.pdf_url)

arxiv_id = external_ids.get("ArXiv")
if arxiv_id:
    urls_to_try.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

pmc_id = external_ids.get("PubMedCentral")
if pmc_id:
    urls_to_try.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/")
```

---

## API 请求字段配置

```python
_fields = [
    "paperId",              # 论文唯一标识
    "corpusId",             # 语料库 ID
    "url",                  # 论文页面链接
    "title",                # 标题
    "abstract",             # 摘要
    "venue",                # 发表场所
    "publicationVenue",     # 发表场所详情
    "year",                 # 发表年份
    "referenceCount",       # 参考文献数量
    "citationCount",        # 被引用次数
    "influentialCitationCount",  # 有影响力的引用数
    "isOpenAccess",         # 是否开放获取
    "openAccessPdf",        # 开放获取 PDF 信息
    "fieldsOfStudy",        # 研究领域
    "s2FieldsOfStudy",      # S2 研究领域（更精确）
    "publicationTypes",     # 出版类型
    "publicationDate",      # 发表日期
    "journal",              # 期刊信息
    "authors",              # 作者列表
    "externalIds"           # 外部标识符
]
```

---

## 完整映射代码参考

```python
def _map_to_paper(self, paper_data: dict) -> Paper:
    return Paper(
        paper_id=paper_data.get("paperId", ""),
        title=paper_data.get("title", ""),
        authors=[a["name"] for a in paper_data.get("authors", []) if a.get("name")],
        abstract=paper_data.get("abstract", "") or "",
        doi=paper_data.get("externalIds", {}).get("DOI", "") or "",
        published_date=parse_date(paper_data),
        pdf_url=self._extract_pdf_url(paper_data),
        url=paper_data.get("url", ""),
        source="semanticscholar",
        updated_date=None,
        categories=extract_categories(paper_data),
        keywords=[],
        citations=paper_data.get("citationCount", 0) or 0,
        references=[str(paper_data.get("referenceCount", 0))],
        extra={
            "venue": paper_data.get("venue", ""),
            "year": paper_data.get("year"),
            "publicationTypes": paper_data.get("publicationTypes", []),
            "isOpenAccess": paper_data.get("isOpenAccess", False),
            "influentialCitationCount": paper_data.get("influentialCitationCount", 0),
            "externalIds": paper_data.get("externalIds", {}),
            "corpusId": paper_data.get("corpusId"),
            "publicationVenue": paper_data.get("publicationVenue"),
            "journal": paper_data.get("journal"),
            "openAccessPdf": paper_data.get("openAccessPdf")
        }
    )
```

# OpenAlex 数据解析到 Paper 类说明

本文档描述 `openalex.py` 中如何将 OpenAlex Work 对象解析映射到 Paper 类。

## 字段映射表

| Paper 字段 | OpenAlex 字段 | 解析逻辑 |
|-----------|--------------|---------|
| `paper_id` | `id` | 直接使用 OpenAlex ID（如 `https://openalex.org/W2741809807`） |
| `title` | `display_name` | 直接使用 |
| `authors` | `authorships[].author.display_name` | 遍历 authorships 数组，提取每个作者的 display_name |
| `abstract` | `abstract_inverted_index` | 调用 `_convert_abstract()` 将倒排索引转为纯文本 |
| `doi` | `doi` | 去除 `https://doi.org/` 前缀 |
| `published_date` | `publication_date` | 解析 `YYYY-MM-DD` 格式，失败则尝试只解析年份 |
| `pdf_url` | `best_oa_location.pdf_url` 或 `locations[].pdf_url` | 调用 `_extract_pdf_url()` 按优先级提取 |
| `url` | `primary_location.landing_page_url` 或 `id` | 优先使用 landing_page_url，否则用 OpenAlex ID |
| `source` | - | 固定为 `"openalex"` |
| `updated_date` | `updated_date` | 解析 ISO 8601 格式 |
| `categories` | `topics[].display_name` 或 `concepts[].display_name` | 优先从 topics 提取，否则从 concepts |
| `keywords` | `keywords[].display_name` | 遍历 keywords 数组 |
| `citations` | `cited_by_count` | 直接使用，默认 0 |
| `references` | `referenced_works` | 直接使用引用列表 |

---

## Extra 字段详解

`extra` 字段用于存储 OpenAlex 特有的扩展元数据，不属于 Paper 类核心字段但具有参考价值。

### Extra 字段结构

```python
extra = {
    "type": str,              # 文献类型
    "language": str,          # 语言代码
    "is_oa": bool,            # 是否开放获取
    "oa_status": str,         # 开放获取状态
    "cited_by_api_url": str,  # 引用查询 API 链接
    "publication_year": int,  # 发布年份
    "saved_path": str         # 下载后的本地保存路径（download 方法填充）
}
```

### 各字段说明

| Extra 字段 | OpenAlex 来源 | 类型 | 说明 |
|-----------|--------------|------|------|
| `type` | `work.type` | `str` | 文献类型，如 `"article"`, `"book-chapter"`, `"preprint"`, `"review"` 等 |
| `language` | `work.language` | `str` | ISO 639-1 语言代码，如 `"en"`, `"zh"`, `"ja"` |
| `is_oa` | `work.open_access.is_oa` | `bool` | 是否为开放获取文献，`True` 表示可免费获取全文 |
| `oa_status` | `work.open_access.oa_status` | `str` | 开放获取状态类型（见下表） |
| `cited_by_api_url` | `work.cited_by_api_url` | `str` | 获取引用该文献的其他文献的 API 链接 |
| `publication_year` | `work.publication_year` | `int` | 发布年份，如 `2023` |
| `saved_path` | - | `str` | 调用 `download()` 后填充，表示 PDF 本地保存路径 |

### OA Status 开放获取状态值

| 状态值 | 含义 |
|-------|------|
| `gold` | 金色开放获取 - 发表在完全开放获取期刊上 |
| `green` | 绿色开放获取 - 作者自存档版本（如预印本服务器） |
| `hybrid` | 混合开放获取 - 订阅期刊中的开放获取文章 |
| `bronze` | 铜色开放获取 - 出版商网站免费可读但无明确许可 |
| `closed` | 封闭获取 - 需要订阅或付费 |

### 使用示例

```python
# 检查是否为开放获取
if paper.extra.get("is_oa"):
    print("该论文可免费获取全文")

# 根据 OA 状态判断
oa_status = paper.extra.get("oa_status", "")
if oa_status in ["gold", "green"]:
    print("高质量开放获取资源")

# 获取引用该论文的文献
cited_by_url = paper.extra.get("cited_by_api_url")
if cited_by_url:
    # 可调用该 API 获取引用列表
    pass

# 检查下载路径
saved_path = paper.extra.get("saved_path")
if saved_path and saved_path != "No fulltext available":
    print(f"PDF 已保存至: {saved_path}")
```

---

## 特殊处理逻辑

### 1. 摘要倒排索引转换 (`_convert_abstract`)

OpenAlex 使用倒排索引格式存储摘要以节省空间：

```python
# OpenAlex 存储格式（倒排索引）:
{
    "Despite": [0], 
    "growing": [1], 
    "interest": [2], 
    "in": [3, 57, 73],
    "deep": [4],
    "learning": [5]
}

# 转换逻辑：
# 1. 找到最大位置索引
# 2. 创建对应长度的词列表
# 3. 按位置填充单词
# 4. 拼接为完整文本

# 转换结果:
"Despite growing interest in deep learning..."
```

### 2. PDF 链接提取优先级 (`_extract_pdf_url`)

```
1. best_oa_location.pdf_url  （最佳开放获取位置）
2. locations[0].pdf_url      （第一个有 PDF 的位置）
3. 返回空字符串             （无可用 PDF）
```

### 3. 作者提取

```python
# OpenAlex authorships 结构:
authorships = [
    {
        "author": {"display_name": "张三", "id": "..."},
        "institutions": [...],
        "author_position": "first"
    },
    ...
]

# 提取逻辑: 遍历取 author.display_name
authors = ["张三", "李四", "王五"]
```

---

## 完整映射代码参考

```python
def _map_to_paper(self, work: dict) -> Paper:
    return Paper(
        paper_id=work.get("id", ""),
        title=work.get("display_name", ""),
        authors=[a.get("author", {}).get("display_name", "") 
                 for a in work.get("authorships", [])],
        abstract=self._convert_abstract(work.get("abstract_inverted_index")),
        doi=work.get("doi", "").replace("https://doi.org/", ""),
        published_date=parse_date(work.get("publication_date")),
        pdf_url=self._extract_pdf_url(work),
        url=work.get("primary_location", {}).get("landing_page_url") or work.get("id"),
        source="openalex",
        updated_date=parse_iso_date(work.get("updated_date")),
        categories=[t.get("display_name") for t in work.get("topics", [])],
        keywords=[k.get("display_name") for k in work.get("keywords", [])],
        citations=work.get("cited_by_count", 0),
        references=work.get("referenced_works", []),
        extra={
            "type": work.get("type", ""),
            "language": work.get("language", ""),
            "is_oa": work.get("open_access", {}).get("is_oa", False),
            "oa_status": work.get("open_access", {}).get("oa_status", ""),
            "cited_by_api_url": work.get("cited_by_api_url", ""),
            "publication_year": work.get("publication_year"),
        }
    )
```

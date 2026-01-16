# Tavily 数据解析到 Paper 类说明

本文档描述 `tavily.py` 中如何将 Tavily 网络搜索结果解析映射到 Paper 类。

## 数据来源

Tavily 是一个专为 AI 应用设计的搜索 API，提供高质量的网络搜索结果，包含标题、内容摘要、URL 和相关性评分。

## 字段映射表

| Paper 字段 | Tavily 字段 | 解析逻辑 |
|-----------|------------|---------|
| `paper_id` | `url` | 使用 URL 的 MD5 哈希值前 16 位 |
| `title` | `title` | 直接使用，默认 `"No Title"` |
| `authors` | - | 空列表（网页搜索无作者信息） |
| `abstract` | `content` | 直接使用搜索结果的内容摘要 |
| `doi` | - | 空字符串（网页无 DOI） |
| `published_date` | `publishedDate` | 解析毫秒时间戳或 ISO 格式，失败则使用当前时间 |
| `pdf_url` | - | `None`（网页搜索无 PDF） |
| `url` | `url` | 直接使用网页链接 |
| `source` | - | 固定为 `"tavily"` |

---

## Extra 字段详解

`extra` 字段用于存储 Tavily 特有的搜索元数据。

### Extra 字段结构

```python
extra = {
    "score": float,      # 相关性评分
    "saved_path": str    # 下载后的本地保存路径
}
```


### 各字段说明

| Extra 字段 | Tavily 来源 | 类型 | 说明 |
|-----------|------------|------|------|
| `score` | `score` | `float` | Tavily 返回的相关性评分，范围 0-1，值越高表示与查询越相关 |
| `saved_path` | - | `str` | 调用 `download()` 后填充，Markdown 文件保存路径 |

### 使用示例

```python
# 获取相关性评分
score = paper.extra.get("score", 0)
print(f"相关性评分: {score:.4f}")

# 按评分排序
papers_sorted = sorted(papers, key=lambda p: p.extra.get("score", 0), reverse=True)

# 检查下载路径
saved_path = paper.extra.get("saved_path")
if saved_path:
    print(f"文件已保存至: {saved_path}")
```

---

## 特殊处理逻辑

### 1. Paper ID 生成

```python
# 使用 URL 的 MD5 哈希值前 16 位作为唯一标识
import hashlib

if url:
    paper_id = hashlib.md5(url.encode()).hexdigest()[:16]
else:
    paper_id = f"tavily_{datetime.now().timestamp()}"
```

### 2. 发布日期解析

```python
published_date_raw = result.get("publishedDate")

if published_date_raw:
    if isinstance(published_date_raw, (int, float)):
        # Tavily 返回毫秒时间戳
        published_date = datetime.fromtimestamp(published_date_raw / 1000)
    else:
        # ISO 格式字符串
        published_date = datetime.fromisoformat(
            str(published_date_raw).replace('Z', '+00:00')
        )
else:
    # 无日期时使用当前时间
    published_date = datetime.now()
```

### 3. Download 方法特点

Tavily 的 `download` 方法与其他数据源不同：
- 不下载 PDF 文件
- 将所有搜索结果合并为一个 Markdown 文件
- 支持自定义文件名

```python
# 默认文件名格式
filename = f"tavily_search_{timestamp}.md"

# Markdown 文件结构
"""
# Tavily搜索结果

**搜索时间:** 2024-01-15 10:30:00
**结果数量:** 5

---

## 标题1
**来源:** [url](url)
**发布日期:** 2024-01-10
**相关性评分:** 0.9234

### 摘要
内容摘要...

---
"""
```

---

## 完整映射代码参考

```python
def _parse_result(self, result: dict) -> Paper:
    title = result.get("title", "No Title")
    content = result.get("content", "")
    url = result.get("url", "")
    score = result.get("score", 0.0)
    
    paper_id = hashlib.md5(url.encode()).hexdigest()[:16] if url else f"tavily_{datetime.now().timestamp()}"
    
    published_date = parse_published_date(result.get("publishedDate"))
    
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=[],
        abstract=content,
        doi="",
        published_date=published_date,
        pdf_url=None,
        url=url,
        source="tavily",
        extra={"score": score}
    )
```

---

## 与其他数据源的对比

| 特性 | Tavily | OpenAlex | Semantic Scholar | SEC EDGAR |
|-----|--------|----------|------------------|-----------|
| 数据类型 | 网页搜索 | 学术论文 | 学术论文 | 财务报告 |
| 有 PDF | ❌ | ✅ | ✅ | ❌ |
| 有作者 | ❌ | ✅ | ✅ | ❌ |
| 有 DOI | ❌ | ✅ | ✅ | ❌ |
| 有引用数 | ❌ | ✅ | ✅ | ❌ |
| 相关性评分 | ✅ | ✅ | ❌ | ❌ |
| 下载格式 | Markdown | PDF | PDF | Markdown |

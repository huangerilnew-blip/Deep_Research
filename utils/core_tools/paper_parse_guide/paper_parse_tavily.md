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

**注意：** 
- `abstract` 字段存储 Tavily 返回的简短摘要（`content`）
- 完整的网页原始内容存储在 `extra["raw_content"]` 中（需要设置 `include_raw_content=True`）

---

## Extra 字段详解

`extra` 字段用于存储 Tavily 特有的搜索元数据和完整内容。

### Extra 字段结构

```python
extra = {
    "score": float,         # 相关性评分
    "raw_content": str,     # 网页完整原始内容
    "saved_path": str       # 下载后的本地保存路径
}
```


### 各字段说明

| Extra 字段 | Tavily 来源 | 类型 | 说明 |
|-----------|------------|------|------|
| `score` | `score` | `float` | Tavily 返回的相关性评分，范围 0-1，值越高表示与查询越相关。**仅 score > 0.8 的结果会被解析为 Paper** |
| `raw_content` | `raw_content` | `str` | 网页的完整原始内容（需要 API 请求中设置 `include_raw_content=True`） |
| `saved_path` | - | `str` | 调用 `download()` 后填充，Markdown 文件保存路径 |

### 使用示例

```python
# 获取相关性评分
score = paper.extra.get("score", 0)
print(f"相关性评分: {score:.4f}")

# 按评分排序
papers_sorted = sorted(papers, key=lambda p: p.extra.get("score", 0), reverse=True)

# 获取完整原始内容
raw_content = paper.extra.get("raw_content", "")
if raw_content:
    print(f"完整内容长度: {len(raw_content)} 字符")
    print(f"内容预览: {raw_content[:200]}...")

# 对比摘要和完整内容
print(f"摘要长度: {len(paper.abstract)} 字符")
print(f"完整内容长度: {len(paper.extra.get('raw_content', ''))} 字符")

# 检查下载路径
saved_path = paper.extra.get("saved_path")
if saved_path:
    print(f"文件已保存至: {saved_path}")
```

---

## 特殊处理逻辑

### 1. Score 过滤机制

```python
# 只解析 score > 0.8 的高质量结果
for result in results:
    score = result.get("score", 0.0)
    
    if score <= 0.8:
        print(f"过滤低分结果 (score={score:.4f}): {result.get('title', 'N/A')[:50]}...")
        continue
    
    paper = self._parse_result(result)
    papers.append(paper)
```

**说明：** 为了保证结果质量，只有相关性评分大于 0.8 的搜索结果才会被解析为 Paper 对象。

### 2. Raw Content 获取

```python
# API 请求中启用 raw_content
payload = {
    "api_key": self.api_key,
    "query": query,
    "search_depth": "basic",
    "max_results": max_results,
    "include_raw_content": True  # 获取完整网页内容
}
```

**说明：** 
- `include_raw_content=True` 会让 Tavily 返回网页的完整原始内容
- `content` 字段：简短摘要（存储在 `abstract`）
- `raw_content` 字段：完整内容（存储在 `extra["raw_content"]`）

### 3. Paper ID 生成

```python
# 使用 URL 的 MD5 哈希值前 16 位作为唯一标识
import hashlib

if url:
    paper_id = hashlib.md5(url.encode()).hexdigest()[:16]
else:
    paper_id = f"tavily_{datetime.now().timestamp()}"
```

### 4. 发布日期解析

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

### 5. Download 方法特点

Tavily 的 `download` 方法与其他数据源不同：
- 不下载 PDF 文件
- 将所有搜索结果合并为一个 Markdown 文件
- 支持自定义文件名
- 包含摘要和完整原始内容（如果有）

```python
# 默认文件名格式
filename = f"tavily_search_{timestamp}.md"

# Markdown 文件结构
"""
# Tavily 搜索结果

**生成时间:** 2026-01-17 12:00:00
**结果数量:** 5

---

## 1. 标题1

### 基本信息
- **来源:** [url](url)
- **发布日期:** 2024-01-10
- **相关性评分:** 0.9234

### 摘要
内容摘要...

### 完整内容
网页的完整原始内容...

---
"""
```

---

## 完整映射代码参考

```python
def _parse_result(self, result: dict) -> Paper:
    title = result.get("title", "No Title")
    content = result.get("content", "")
    raw_content = result.get("raw_content", "")
    url = result.get("url", "")
    score = result.get("score", 0.0)
    
    paper_id = hashlib.md5(url.encode()).hexdigest()[:16] if url else f"tavily_{datetime.now().timestamp()}"
    
    published_date = parse_published_date(result.get("publishedDate"))
    
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=[],
        abstract=content,  # 简短摘要
        doi="",
        published_date=published_date,
        pdf_url=None,
        url=url,
        source="tavily",
        extra={
            "score": score,
            "raw_content": raw_content  # 完整原始内容
        }
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

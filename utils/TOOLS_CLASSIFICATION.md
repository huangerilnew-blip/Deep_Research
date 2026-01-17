# 工具分类说明

## 概述

`tools.py` 中的 `get_tools()` 函数现在支持工具分类，可以根据需要获取不同类型的工具。

## 工具分类

### 1. 搜索工具 (Search Tools)
用于搜索和检索数据，返回 Paper 对象列表。

**工具列表：**
- `wikipedia_search` - 搜索维基百科
- `openalex_search` - 搜索 OpenAlex 学术数据库
- `semantic_scholar_search` - 搜索 Semantic Scholar 学术数据库
- `tavily_search` - 网页搜索
- `sec_edgar_search` - 搜索美国证券交易委员会数据库
- `akshare_search` - 搜索中国股市数据

### 2. 下载工具 (Download Tools)
用于下载搜索结果到本地文件系统。

**工具列表：**
- `wikipedia_download` - 下载维基百科内容为 Markdown
- `openalex_download` - 下载学术论文 PDF
- `semantic_scholar_download` - 下载学术论文 PDF
- `tavily_download` - 下载网页内容为 Markdown
- `sec_edgar_download` - 下载证券文件为 Markdown
- `akshare_download` - 下载股市数据为 Markdown

## 使用方法

### 获取所有工具
```python
from tools import get_tools

all_tools = await get_tools(tool_type="all")
# 或者
all_tools = await get_tools()  # 默认返回所有工具
```

### 只获取搜索工具
```python
search_tools = await get_tools(tool_type="search")
```

### 只获取下载工具
```python
download_tools = await get_tools(tool_type="download")
```

## ExecutorAgent 中的使用

### 初始化时只加载搜索工具
```python
class ExecutorAgent:
    def __init__(self, pool, modelname):
        # 只获取搜索工具
        self.search_tools = self._get_search_tools()
        # 下载工具延迟加载
        self.download_tools = None
    
    def _get_search_tools(self):
        """获取搜索工具"""
        return asyncio.run(get_tools(tool_type="search"))
```

### 下载阶段动态加载下载工具
```python
async def _download_node(self, state):
    # 动态获取下载工具
    download_tools = await self._get_download_tools()
    
    # 使用下载工具
    for source, papers in papers_by_source.items():
        download_tool_name = f"{source}_download"
        download_tool = next(
            (t for t in download_tools if download_tool_name in t.name.lower()),
            None
        )
        if download_tool:
            result = await download_tool.ainvoke({
                "papers": papers,
                "save_path": Config.DOC_SAVE_PATH
            })
```

## 优势

### 1. 性能优化
- ExecutorAgent 初始化时只加载必需的搜索工具
- 下载工具在需要时才加载，减少内存占用

### 2. 职责分离
- 搜索阶段：只使用搜索工具
- 下载阶段：只使用下载工具
- 清晰的阶段划分

### 3. 灵活性
- 可以根据需要选择性加载工具
- 便于测试和调试

### 4. 可维护性
- 工具分类清晰
- 易于扩展新工具

## 测试

运行以下命令测试工具分类功能：

```bash
cd utils
python tools.py
```

输出示例：
```
============================================================
测试获取所有工具
============================================================
一共有 12 个工具可用。

============================================================
测试获取搜索工具
============================================================
一共有 6 个搜索工具：
1. wikipedia_search: 搜索 Wikipedia 百科...
2. openalex_search: 搜索 OpenAlex 学术数据库...
3. semantic_scholar_search: 搜索 Semantic Scholar 学术数据库...
4. tavily_search: 使用 Tavily 进行网页搜索...
5. sec_edgar_search: 搜索美国证券交易委员会（SEC EDGAR）数据库...
6. akshare_search: 搜索中国股市数据（AkShare）...

============================================================
测试获取下载工具
============================================================
一共有 6 个下载工具：
1. wikipedia_download: 下载 Wikipedia 搜索结果为 Markdown 文件
2. openalex_download: 下载 OpenAlex 搜索结果的 PDF 文件
3. semantic_scholar_download: 下载 Semantic Scholar 搜索结果的 PDF 文件
4. tavily_download: 下载 Tavily 搜索结果为 Markdown 文件
5. sec_edgar_download: 下载 SEC EDGAR 搜索结果为 Markdown 文件
6. akshare_download: 下载 AkShare 搜索结果为 Markdown 文件
```

## 注意事项

1. **异步调用**: `get_tools()` 是异步函数，需要使用 `await` 或 `asyncio.run()`
2. **工具命名**: 工具名称必须包含 "search" 或 "download" 才能被正确分类
3. **延迟加载**: 下载工具在 ExecutorAgent 中采用延迟加载策略，首次使用时才会加载
4. **MCP Server**: 所有工具都来自统一的 MCP Server，确保 MCP Server 正常运行

## 相关文件

- `utils/tools.py` - 工具获取和分类逻辑
- `agents/agents.py` - ExecutorAgent 使用工具的实现
- `utils/core_tools/mcp_server.py` - MCP Server 实现
- `utils/core_tools/README_MCP.md` - MCP Server 文档

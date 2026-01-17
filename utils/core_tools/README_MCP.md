# Paper Search MCP Server

这是一个整合了多个论文搜索工具的 MCP Server，使用 stdio 模式通信。

## 功能特性

### 支持的搜索工具

1. **OpenAlex** - 学术数据库搜索
   - `openalex_search`: 搜索学术论文
   - `openalex_download`: 下载 PDF 文件

2. **Semantic Scholar** - 学术数据库搜索
   - `semantic_scholar_search`: 搜索学术论文
   - `semantic_scholar_download`: 下载 PDF 文件

3. **Wikipedia** - 维基百科搜索
   - `wikipedia_search`: 搜索维基百科词条
   - `wikipedia_download`: 下载为 Markdown 文件

4. **Tavily** - 网页搜索
   - `tavily_search`: 高质量网页搜索（score > 0.8）
   - `tavily_download`: 下载为 Markdown 文件

5. **SEC EDGAR** - 美国证券交易委员会数据库
   - `sec_edgar_search`: 搜索公司证券信息
   - `sec_edgar_download`: 下载为 Markdown 文件

6. **AkShare** - 中国股市数据
   - `akshare_search`: 搜索中国上市公司信息
   - `akshare_download`: 下载为 Markdown 文件

## 安装

```bash
cd utils/core_tools
pip install -e .
```

## 配置

在 `.env` 文件中配置必要的 API 密钥：

```env
TAVILY_API_KEY=your_tavily_api_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

## 使用方式

### 1. 直接运行 MCP Server

```bash
python mcp_server.py
```

### 2. 在 Kiro 中配置

在 `.kiro/settings/mcp.json` 中添加配置：

```json
{
  "mcpServers": {
    "paper-search": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/utils/core_tools",
      "env": {
        "TAVILY_API_KEY": "your_api_key",
        "SEMANTIC_SCHOLAR_API_KEY": "your_api_key"
      },
      "transport": "stdio",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### 3. 在 LangGraph 中使用

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "paper-search": {
        "command": "python",
        "args": ["-m", "mcp_server"],
        "cwd": "/path/to/utils/core_tools",
        "transport": "stdio"
    }
})

# 获取所有工具
tools = await client.get_tools()

# 使用工具
result = await tools[0].ainvoke({"query": "artificial intelligence"})
```

## 工具使用规则

### 必调工具（每个子问题都必须调用）
- `wikipedia_search`
- `openalex_search`
- `semantic_scholar_search`
- `tavily_search`

### 条件调用工具
- `sec_edgar_search`: 仅当涉及美国证券市场时调用
- `akshare_search`: 仅当涉及中国上市公司时调用

## 数据清洗规则

需要清洗的数据源（用于 BGE Rerank）：
- **openalex**: 总是需要清洗 `paper.abstract`
- **semantic_scholar**: 仅当 `paper.abstract` 不为空时清洗
- **tavily**: 总是需要清洗 `paper.abstract`

## 摘要提取规则

- **akshare**: 使用 `paper.abstract`（公司基本情况）
- **openalex**: 使用 `paper.abstract`（学术基本摘要）
- **semantic_scholar**: 使用 `paper.abstract`（学术基本摘要）
- **sec_edgar**: 拼接 `<公司基本信息><风险情况><公司评估>`
- **tavily**: 使用 `paper.abstract`（网页摘要）
- **wiki**: 使用 `paper.abstract`（完整内容）

## 返回数据格式

所有搜索工具返回 Paper 对象的 JSON 数组：

```json
[
  {
    "paper_id": "unique_id",
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Abstract text...",
    "doi": "10.1234/example",
    "published_date": "2024-01-01T00:00:00",
    "pdf_url": "https://example.com/paper.pdf",
    "url": "https://example.com/paper",
    "source": "openalex",
    "categories": ["AI", "ML"],
    "keywords": ["deep learning"],
    "citations": 100,
    "references": ["ref1", "ref2"],
    "extra": {
      "score": 0.95,
      "raw_content": "Full content...",
      "saved_path": "/path/to/file.pdf"
    }
  }
]
```

## 开发

### 运行测试

```bash
pytest tests/
```

### 添加新工具

1. 在相应的搜索器文件中实现 `search` 和 `download` 方法
2. 在 `mcp_server.py` 中添加工具定义和处理逻辑
3. 更新文档

## 许可证

MIT License

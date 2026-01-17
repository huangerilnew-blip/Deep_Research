# ExecutorAgent 架构说明

## 概述

ExecutorAgent 负责执行 PlannerAgent 生成的单个子问题，使用 LangGraph 构建了一个多阶段处理流程。

## 架构设计

### 流程图

```
START
  ↓
llm_decision_node (Thought + Action)
  ↓
[条件边: _should_call_optional_tools]
  ↓                              ↓
optional_tool_node            search_node
(执行可选工具)                 (调用必需工具)
  ↓                              ↓
llm_decision_node              clean_and_rerank_node
(Observation + 再次决策)         ↓
  ↓                           download_node
[条件边: 继续循环?]               ↓
  ↓           ↓              summarize_node
  YES        NO                  ↓
  ↑           ↓                 END
  └─循环─→ search_node
           (合并所有结果)
              ↓
         clean_and_rerank_node
              ↓
         download_node
              ↓
         summarize_node
              ↓
            END
```

**流程说明**：

1. **第一次决策路径**：
   - `llm_decision_node` → 如果需要可选工具 → `optional_tool_node` → `llm_decision_node` (循环)
   - `llm_decision_node` → 如果不需要可选工具 → `search_node` → `clean_and_rerank` → ... → END

2. **循环路径**：
   - `optional_tool_node` 执行完后 → 返回 `llm_decision_node`
   - LLM 看到 Observation，再次决策
   - 如果还需要工具 → 继续循环
   - 如果信息足够 → 进入 `search_node`

3. **最终路径**：
   - 无论从哪个路径进入 `search_node`
   - 都会按顺序执行：`search` → `clean_and_rerank` → `download` → `summarize` → END

**关键点**：
- `optional_tool_node` 后必须返回 `llm_decision_node` (循环)
- `llm_decision_node` 决定不调用工具后，直接进入 `search_node`
- 进入 `search_node` 后，不再返回循环，直接执行后续节点

## 节点说明

### 1. llm_decision_node (LLM 决策节点)

### 1. llm_decision_node (LLM 决策节点 - ReAct 模式)

**功能**: 按照 ReAct (Reasoning + Acting) 模式让 LLM 进行推理和决策

**ReAct 模式**:
1. **Thought (思考)**: LLM 分析当前情况和已有信息
2. **Action (行动)**: LLM 决定调用哪个工具或完成决策
3. **Observation (观察)**: 工具返回结果（由 optional_tool_node 提供）
4. **循环**: 重复上述过程直到 LLM 认为信息足够

**实现**:
- 使用 `bind_tools` 将可选工具绑定到 LLM
- 构建 ReAct 风格的提示，包含问题、已有观察和可用工具
- 如果已经调用过可选工具，会格式化观察结果展示给 LLM
- LLM 返回思考内容和/或工具调用

**观察结果格式化**:
- 对于 sec_edgar: 提取前 200 字符 + "成功获取 XXX 公司的完整信息"
- 对于 akshare: 提取前 200 字符 + "成功获取 XXX 的完整信息"
- 让 LLM 知道已经成功获取了哪些公司的信息

**循环机制**:
- 第一次调用：LLM 分析子问题，决定是否需要可选工具
- 后续调用：LLM 看到观察结果（Observation），决定是否需要更多工具
- LLM 可以多次调用不同的可选工具，直到认为信息足够

**示例**:
```python
# 第一次决策
# Question: "Tesla 和比亚迪的对比分析"
# Thought: 需要获取 Tesla 和比亚迪的公司信息
# Action: 调用 sec_edgar_search (Tesla)

# 第二次决策（看到 Tesla 结果后）
# Observation: ✓ 成功获取 Tesla Inc. 的完整信息
#              来源: SEC EDGAR
#              预览: Tesla, Inc. designs, develops, manufactures...
# Thought: 已有 Tesla 信息，还需要比亚迪的信息
# Action: 调用 akshare_search (比亚迪)

# 第三次决策（看到两个结果后）
# Observation: ✓ 成功获取 Tesla Inc. 的完整信息
#              ✓ 成功获取比亚迪的完整信息
# Thought: 两家公司的信息都已获取，可以继续
# Action: 回复"信息已足够，可以继续"
```

### 2. _should_call_optional_tools (条件路由)

**功能**: 根据 LLM 的决策判断下一步路由

**逻辑**:
- 如果 AIMessage 包含 tool_calls → 路由到 `optional_tool_node`
- 如果 AIMessage 不包含 tool_calls → 路由到 `search_node`

**循环支持**:
- 从 `optional_tool_node` 返回后，会再次进入 `llm_decision_node`
- LLM 可以继续决策是否需要更多可选工具
- 形成循环直到 LLM 决定进入搜索阶段

### 3. optional_tool_node (可选工具节点 - Action Execution)

**功能**: 执行 LLM 决定调用的可选工具，返回观察结果（Observation）

**实现**:
- 从 AIMessage 的 tool_calls 中提取工具名和参数
- 查找对应的工具并执行（Action）
- 解析 MCP 工具返回的 JSON 结果
- 将结果追加到 `optional_search_results`（支持多次调用）
- 执行完成后返回到 `llm_decision_node` 进行下一轮决策

**观察结果**:
- 工具执行的结果会在下一次 `llm_decision_node` 中被格式化为观察信息
- LLM 可以看到具体获取了哪些公司的信息和内容预览

**支持的工具**:
- `sec_edgar_search`: 查询美国证券市场公司信息
- `akshare_search`: 查询中国上市公司信息

### 4. search_node (搜索节点)

**功能**: 并行调用必需的搜索工具，然后合并可选工具的结果

**必需工具**:
- `wikipedia_search`: 维基百科搜索
- `openalex_search`: 学术论文搜索
- `semantic_scholar_search`: 学术论文搜索
- `tavily_search`: 网页搜索

**实现**:
1. 并行调用所有必需工具
2. 解析 MCP 工具返回的 JSON 结果
3. 合并 `optional_search_results` 中的可选工具结果
4. 返回所有搜索结果

### 5. clean_and_rerank_node (清洗和 Rerank 节点)

**功能**: 使用 BGE Reranker 对搜索结果进行重排序

**清洗规则**:
- `openalex`: 总是清洗
- `semantic_scholar`: 仅当 abstract 不为空时清洗
- `tavily`: 总是清洗
- 其他来源: 不清洗

**Rerank 流程**:
1. 提取需要 rerank 的文档的摘要
2. 使用 LlamaIndex 创建临时索引
3. 使用 BGE Reranker 进行重排序
4. 过滤低分文档（score < RERANK_THRESHOLD）
5. 添加未参与 rerank 的文档

**配置**:
- `RERANK_MODEL`: "BAAI/bge-reranker-large"
- `RERANK_THRESHOLD`: 0.5
- `RERANK_TOP_N`: 20

### 6. download_node (下载节点)

**功能**: 根据 paper.source 按检索器类型进行下载

**实现**:
1. 按 source 分组文档
2. 为每个 source 调用对应的下载工具
3. 下载到 `Config.DOC_SAVE_PATH`
4. 解析下载结果并返回

**下载工具映射**:
- `wikipedia` → `wikipedia_download`
- `openalex` → `openalex_download`
- `semantic_scholar` → `semantic_scholar_download`
- `tavily` → `tavily_download`
- `sec_edgar` → `sec_edgar_download`
- `akshare` → `akshare_download`

### 7. summarize_node (摘要节点)

**功能**: 生成完整的结果摘要

**返回结构**:
```python
{
    "query": "子问题",
    "total_papers": 10,
    "papers_by_source": {
        "openalex": 3,
        "semantic_scholar": 2,
        "tavily": 2,
        "wikipedia": 1,
        "akshare": 2
    },
    "top_papers": [
        {
            "title": "论文标题",
            "source": "openalex",
            "rerank_score": 0.85,
            "url": "https://...",
            "saved_path": "/path/to/file",
            "abstract": "完整摘要...",
            "authors": ["作者1", "作者2"],
            "published_date": "2024-01-01",
            "doi": "10.1234/..."
        }
    ],
    "statistics": {
        "searched": 50,
        "after_rerank": 20,
        "downloaded": 10
    }
}
```

## 状态管理

### ExecutorState

```python
class ExecutorState(TypedDict):
    executor_messages: Annotated[list[AnyMessage], add_messages]
    current_query: str  # 当前处理的子问题
    optional_search_results: List[Dict]  # 可选工具的搜索结果
    search_results: List[Dict]  # 所有搜索结果（必需工具 + 可选工具）
    reranked_results: List[Dict]  # Rerank后的结果
    downloaded_papers: List[Dict]  # 已下载的论文
    executor_result: Dict  # 最终结果摘要
```

## 工具分类

### 搜索工具 (Search Tools)

通过 `get_tools(tool_type="search")` 获取：
- `wikipedia_search`
- `openalex_search`
- `semantic_scholar_search`
- `tavily_search`
- `sec_edgar_search` (可选)
- `akshare_search` (可选)

### 下载工具 (Download Tools)

通过 `get_tools(tool_type="download")` 延迟加载：
- `wikipedia_download`
- `openalex_download`
- `semantic_scholar_download`
- `tavily_download`
- `sec_edgar_download`
- `akshare_download`

## 关键特性

### 1. LLM 驱动的可选工具调用

使用 LangGraph 的 `bind_tools` 和条件边，让 LLM 自主决定是否调用可选工具，而不是硬编码规则。

### 2. 延迟加载下载工具

ExecutorAgent 初始化时只加载搜索工具，下载工具在需要时才加载，减少初始化开销。

### 3. 智能 Rerank

使用 BGE Reranker 模型对搜索结果进行重排序，提高结果相关性。

### 4. 完整摘要

不限制摘要长度，返回完整的文档摘要，确保信息完整性。

### 5. 健壮的 JSON 解析

支持多种返回类型（JSON 字符串、列表、字典），增强系统健壮性。

## 使用示例

```python
from agents import ExecutorAgent
from psycopg_pool import AsyncConnectionPool
from config import Config

# 创建数据库连接池
pool = AsyncConnectionPool(
    conninfo=Config.DB_URI,
    min_size=Config.MIN_SIZE,
    max_size=Config.MAX_SIZE
)

# 初始化 ExecutorAgent
executor = ExecutorAgent(pool)

# 执行子问题
result = await executor.invoke(
    query="深度求索公司的基本情况",
    thread_id="thread_123"
)

# 查看结果
print(f"搜索结果数: {result['statistics']['searched']}")
print(f"Rerank 后: {result['statistics']['after_rerank']}")
print(f"已下载: {result['statistics']['downloaded']}")
print(f"数据源分布: {result['papers_by_source']}")

# 清理资源
await executor._clean()
await pool.close()
```

## 配置参数

在 `config.py` 中配置：

```python
# Rerank 配置
RERANK_MODEL = "BAAI/bge-reranker-large"
RERANK_THRESHOLD = 0.5
RERANK_TOP_N = 20

# 下载路径
DOC_SAVE_PATH = "../../doc/downloads"

# LLM 配置
LLM_EXECUTOR = "qwen"
```

## 日志记录

所有关键步骤都有详细的日志记录：
- LLM 决策结果
- 工具调用情况
- 搜索结果数量
- Rerank 结果
- 下载进度
- 错误信息

日志文件: `logfile/app.log`

## 错误处理

- 工具调用失败: 记录错误并继续处理其他工具
- JSON 解析失败: 记录错误并跳过该结果
- Rerank 失败: 返回原始搜索结果
- 下载失败: 记录错误并继续下载其他文档

## 性能优化

1. **并行搜索**: 必需工具并行调用，减少等待时间
2. **延迟加载**: 下载工具按需加载
3. **智能过滤**: Rerank 过滤低相关性文档，减少下载量
4. **连接池**: 使用 PostgreSQL 连接池管理数据库连接

## 未来改进

1. 支持更多可选工具
2. 优化 Rerank 策略
3. 添加缓存机制
4. 支持增量下载
5. 添加重试机制

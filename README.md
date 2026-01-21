# pipeline
## PlannerAgent-1.0版本
- 功能:按照DAG思想，负责将用户的问题分解为多个子任务。
- 输出格式json：
- 样例
```
{
  "tasks": [
    {"id": "T1", "query": "子问题1", "dep": []},
    {"id": "T2", "query": "子问题2", "dep": ["T1"]},
    {"id": "T3", "query": "子问题3", "dep": ["T1"]}
    {"id": "T4", "query": "子问题4", "dep": ["T2"]}
  ]
}
```
## PlannerAgent-2.0版本

功能:只负责将用户的问题分解为多个子任务，不需要按照DAG思想。

## 性格优化

- 初始化parentagnt时候，先初始化多个executoragent，用于并发执行子问题

## ExecutorAgent

### 1. 核心定位与职责

- ExecutorAgent 是一个负责执行单个子问题完整处理流程的智能代理。它的主要职责包括：

  - 接收并处理来自 PlannerAgent 拆解后的子问题

  - 智能决策是否调用可选工具获取额外信息

  - 并行调用必需的搜索工具获取相关文档

  - 对搜索结果进行清洗和重排序（Rerank）

  - 下载相关文档到本地存储

  - 返回处理后的结构化结果


### 2. 架构设计

#### 2.1 工作流程（基于 LangGraph）

ExecutorAgent 采用 ReAct（Reasoning + Acting）模式，工作流程如下：

```
START 
  ↓
llm_decision_node (LLM 决策)
  ↓
[条件路由]
  ├─→ optional_tool_node (可选工具) → llm_decision_node (循环)
  └─→ search_node (必需搜索)
        ↓
      clean_and_rerank (清洗与重排序)
        ↓
      download (文档下载)
        ↓
      END
```

#### 2.2 核心组件

- **LLM 决策引擎**: 使用配置的大语言模型进行智能决策
- **工具系统**: 
  - 必需工具：wikipedia_search, openalex_search, semantic_scholar_search, tavily_search
  - 可选工具：sec_edgar_search, akshare_search
- **Reranker**: 基于 BGE 模型的文档重排序器
- **记忆系统**: 使用 PostgreSQL 存储对话历史和状态

### 3. 技术特性

#### 3.1 智能决策机制

ExecutorAgent 实现了基于 ReAct 模式的智能决策：

- **Thought（思考）**: LLM 分析当前信息是否足够
- **Action（行动）**: 决定是否调用可选工具
- **Observation（观察）**: 接收工具返回的结果
- **循环决策**: 可以多次调用可选工具，直到 LLM 认为信息充足

#### 3.2 并发处理能力

- 使用 `asyncio.gather` 并行调用多个必需搜索工具
- 即使某个工具失败也不会中断整体流程（`return_exceptions=True`）
- 支持批量文档下载的并发处理

#### 3.3 智能重排序（Rerank）

- 使用 TEI 部署的 bge-reranker 模型
- 只对配置的数据源（openalex, semantic_scholar）进行重排序
- 支持阈值过滤和 Top-N 截断
- 保留未参与重排序的其他来源文档

#### 3.4 观察信息裁剪

为了减少 Token 消耗和加快 LLM 决策速度，ExecutorAgent 实现了 `_crop_observation` 方法：

- 只保留关键信息（标题、来源、简短预览）
- 针对不同数据源采用不同的格式化策略
- 将完整的 JSON 数据裁剪为易读的摘要信息

### 4. 数据流转

#### 4.1 状态管理（ExecutorState）

```python
{
    "executor_messages": List[AnyMessage],      # 对话历史
    "current_query": str,                       # 当前子问题
    "optional_search_results": List[ToolMessage], # 可选工具结果
    "search_results": List[Dict],               # 所有搜索结果
    "reranked_results": List[Dict],             # 重排序后的结果
    "downloaded_papers": List[Dict],            # 已下载的文档
    "executor_result": Dict                     # 最终结果摘要
}
```

#### 4.2 数据源分类处理

ExecutorAgent 对不同数据源采用差异化处理策略：

- **学术论文源**（openalex, semantic_scholar）：
  - 提取摘要用于 Rerank
  - 参与相关性评分
  - 按分数排序后保留 Top-N

- **专业数据源**（sec_edgar, akshare）：
  - 作为可选工具，由 LLM 决定是否调用
  - 不参与 Rerank，直接保留
  - 适用于特定领域的深度查询

- **通用搜索源**（wikipedia, tavily）：
  - 作为必需工具，每次都调用
  - 不参与 Rerank，直接保留
  - 提供基础背景信息

## build_vector_store
- 功能:构建向量数据库
- 向量数据库：chroma
- 框架：llamaindex
- 流程:
    - 使用mineru对pdf进行解析，提取文本内容
    - crunchbase作为基础数据+markdown文件上层数据
    - 入库
## search_info
- 功能:检索相关信息
- pre_questions:通过question_pool构建search目标
- question_pool:由planner和markdown文件改写获得
- 过滤：rerank+过滤
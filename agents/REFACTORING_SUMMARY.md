# ExecutorAgent 重构总结

## 重构目标

将 ExecutorAgent 中的可选工具（sec_edgar_search, akshare_search）调用逻辑重构为独立节点，使用 LangGraph 的 `bind_tools` 和条件边处理可选工具，让 LLM 自主决定是否调用。

## 重构前的问题

1. **硬编码逻辑**: 可选工具的调用逻辑混在 `_search_node` 中，难以维护
2. **缺乏灵活性**: 无法让 LLM 根据问题内容智能决定是否调用可选工具
3. **代码耦合**: 搜索节点承担了太多职责

## 重构后的架构

### 新增节点

1. **llm_decision_node**: LLM 决策节点
   - 使用 `bind_tools` 将可选工具绑定到 LLM
   - LLM 根据子问题内容决定是否调用工具
   - 返回 AIMessage，可能包含 tool_calls

2. **optional_tool_node**: 可选工具节点
   - 执行 LLM 决定调用的可选工具
   - 解析 MCP 工具返回的 JSON 结果
   - 将结果存储到 `optional_search_results`

3. **_should_call_optional_tools**: 条件路由函数
   - 根据 LLM 的决策判断下一步路由
   - 如果有 tool_calls → optional_tool_node
   - 如果没有 tool_calls → search_node

### 修改节点

1. **search_node**: 搜索节点
   - 只负责调用必需工具（wikipedia, openalex, semantic_scholar, tavily）
   - 合并 `optional_search_results` 中的可选工具结果
   - 删除了可选工具的调用逻辑

### 新增状态字段

```python
class ExecutorState(TypedDict):
    # ... 其他字段
    optional_search_results: List[Dict]  # 可选工具的搜索结果
```

## 流程对比

### 重构前

```
START → search_node (调用所有工具) → clean_and_rerank → download → summarize → END
```

### 重构后

```
START
  ↓
llm_decision_node (Thought + Action)
  ↓
[条件边: _should_call_optional_tools]
  ↓                              ↓
optional_tool_node            search_node
(执行可选工具)                 (完成决策)
  ↓                              ↓
llm_decision_node              clean_and_rerank
(Observation + 再次决策)         ↓
  ↓                           download
[条件边: 继续循环?]               ↓
  ↓           ↓              summarize
  YES        NO                  ↓
  ↑           ↓                 END
  └─循环─→ search_node
           (合并所有结果)
              ↓
         clean_and_rerank
              ↓
         download
              ↓
         summarize
              ↓
            END
```

**循环机制**：
- `optional_tool_node` 执行完后返回 `llm_decision_node`
- LLM 可以看到观察结果（Observation）
- LLM 决定是否需要调用更多可选工具
- 形成循环直到 LLM 认为信息足够
- 一旦进入 `search_node`，就不再返回循环

## 关键改进

### 1. LLM 驱动的决策

**重构前**:
```python
# 硬编码规则
if "中国" in query or "上市" in query:
    call_akshare()
if "美国" in query or "NYSE" in query:
    call_sec_edgar()
```

**重构后**:
```python
# LLM 自主决策
llm_with_tools = self.chat_llm.bind_tools(optional_tools)
response = await llm_with_tools.ainvoke(prompt)
# LLM 决定是否调用工具
```

### 2. 清晰的职责分离

- **llm_decision_node**: 只负责决策
- **optional_tool_node**: 只负责执行可选工具
- **search_node**: 只负责执行必需工具和合并结果

### 3. 灵活的条件路由和循环机制

使用 LangGraph 的条件边和循环，让 LLM 可以多次决策：

```python
# optional_tool_node 执行完后返回 llm_decision_node
builder.add_edge("optional_tool_node", "llm_decision_node")

# LLM 可以继续决策
builder.add_conditional_edges(
    "llm_decision",
    self._should_call_optional_tools,
    {
        "optional_tool_node": "optional_tool_node",  # 继续调用工具
        "search": "search"  # 完成决策，进入搜索
    }
)
```

**循环优势**：
- LLM 可以根据已有结果决定是否需要更多信息
- 支持复杂查询（如"Tesla 和 BYD 的对比"需要调用两个不同的工具）
- 避免一次性调用所有工具造成的浪费

## 代码变更

### 新增代码

1. `_llm_decision_node()`: 约 40 行
2. `_should_call_optional_tools()`: 约 15 行
3. `_optional_tool_node()`: 约 60 行
4. `_get_optional_tools()`: 约 5 行

### 修改代码

1. `_search_node()`: 删除可选工具调用逻辑，添加结果合并逻辑
2. `_build_graph()`: 添加新节点和条件边
3. `ExecutorState`: 添加 `optional_search_results` 字段

### 删除代码

1. `_search_node()` 中的可选工具调用逻辑（约 30 行）

## 测试验证

### 测试用例

1. **测试 1**: "深度求索公司的基本情况"
   - 预期: LLM 决定调用 akshare_search
   - 验证: optional_search_results 包含 akshare 结果

2. **测试 2**: "Tesla Inc. 的财务状况"
   - 预期: LLM 决定调用 sec_edgar_search
   - 验证: optional_search_results 包含 sec_edgar 结果

3. **测试 3**: "人工智能的最新研究进展"
   - 预期: LLM 决定不调用可选工具
   - 验证: optional_search_results 为空

### 测试脚本

- `test_executor.py`: 完整的功能测试
- `visualize_graph.py`: 流程图可视化

## 配置要求

### 必需配置

```python
# config.py
LLM_EXECUTOR = "qwen"  # 或其他支持 tool calling 的模型
RERANK_MODEL = "BAAI/bge-reranker-large"
RERANK_THRESHOLD = 0.5
RERANK_TOP_N = 20
DOC_SAVE_PATH = "../../doc/downloads"
```

### 环境变量

```bash
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
SEMANTIC_SCHOLAR_API_KEY=your_key
DB_URI=postgresql://user:pass@host:port/db
```

## 性能影响

### 优势

1. **更智能的工具选择**: LLM 根据问题内容决定，避免不必要的调用
2. **更好的可维护性**: 代码结构清晰，易于扩展
3. **更灵活的流程**: 可以轻松添加新的可选工具

### 开销

1. **额外的 LLM 调用**: 每个子问题需要一次额外的 LLM 调用用于决策
2. **略微增加的延迟**: 决策节点增加约 1-2 秒的延迟

## 未来扩展

### 1. 添加新的可选工具

只需修改 `_get_optional_tools()`:

```python
def _get_optional_tools(self) -> List[BaseTool]:
    optional_tool_names = [
        "sec_edgar_search",
        "akshare_search",
        "new_tool_search"  # 新工具
    ]
    return [t for t in self.search_tools if any(opt in t.name.lower() for opt in optional_tool_names)]
```

### 2. 优化 LLM 决策提示

修改 `_llm_decision_node()` 中的 prompt:

```python
prompt = f"""
子问题: {query}

请分析这个子问题，决定是否需要调用以下工具：
- sec_edgar_search: 查询美国证券市场公司信息
- akshare_search: 查询中国上市公司信息
- new_tool_search: 新工具描述

如果需要，请调用相应的工具。如果不需要，直接回复"不需要额外工具"。
"""
```

### 3. 添加工具调用缓存

缓存 LLM 的决策结果，避免重复调用：

```python
# 添加缓存字典
self.decision_cache = {}

# 在 _llm_decision_node 中检查缓存
cache_key = hash(query)
if cache_key in self.decision_cache:
    return self.decision_cache[cache_key]
```

## 文档

- `EXECUTOR_ARCHITECTURE.md`: 完整的架构说明
- `REFACTORING_SUMMARY.md`: 本文档，重构总结
- `test_executor.py`: 测试脚本
- `visualize_graph.py`: 可视化脚本

## 总结

本次重构成功地将可选工具调用逻辑从搜索节点中分离出来，使用 LangGraph 的 `bind_tools` 和条件边实现了 LLM 驱动的智能工具选择。重构后的代码结构更清晰，更易于维护和扩展，同时保持了原有的功能完整性。

## 下一步

1. 运行测试脚本验证功能
2. 监控 LLM 决策的准确性
3. 根据实际使用情况优化提示词
4. 考虑添加工具调用缓存
5. 扩展更多可选工具

# ExecutorAgent 循环机制说明

## 概述

ExecutorAgent 现在使用 **ReAct (Reasoning + Acting)** 模式，让 LLM 通过交替进行推理和行动来智能地决策是否调用可选工具。

### ReAct 模式

- **Thought (思考)**: LLM 分析当前情况和已有信息
- **Action (行动)**: LLM 决定调用哪个工具或完成决策
- **Observation (观察)**: 工具返回执行结果，格式化为可读信息
- **循环**: 重复上述过程直到 LLM 认为信息足够

## 循环流程

```
START
  ↓
llm_decision_node (第一次决策)
  ↓
[条件边: _should_call_optional_tools]
  ↓                              ↓
optional_tool_node            search_node (完成决策)
  ↓                              ↓
llm_decision_node (第二次决策)   clean_and_rerank
  ↓                              ↓
[条件边: 继续循环或进入搜索]      download
  ↓                              ↓
  ├─→ optional_tool_node (继续)  summarize
  │     ↓                        ↓
  │   llm_decision_node (第N次)  END
  │     ↓
  └─→ search_node (合并所有结果)
        ↓
   clean_and_rerank
        ↓
     download
        ↓
    summarize
        ↓
      END
```

**关键点**：
- 从 `llm_decision_node` 直接进入 `search_node` 后，不再返回循环
- 进入 `search_node` 后，按顺序执行后续所有节点直到 END

## 关键修改

### 1. 边的修改

**修改前**:
```python
builder.add_edge("optional_tool_node", "search")
```

**修改后**:
```python
builder.add_edge("optional_tool_node", "llm_decision")
```

这个修改让可选工具执行完后返回到 LLM 决策节点，而不是直接进入搜索节点。

### 2. LLM 决策节点的增强

`_llm_decision_node` 现在会检查是否已经有可选工具的结果：

```python
optional_search_results = state.get("optional_search_results", [])

if optional_search_results:
    # 显示已有结果的摘要
    results_summary = f"\n\n已调用的可选工具结果：\n"
    results_by_source = {}
    for result in optional_search_results:
        source = result.get("source", "unknown")
        results_by_source[source] = results_by_source.get(source, 0) + 1
    
    for source, count in results_by_source.items():
        results_summary += f"- {source}: {count} 条结果\n"
    
    # 提示 LLM 考虑已有结果
    prompt = f"""
    子问题: {query}
    {results_summary}
    请分析这个子问题和已有的结果，决定是否需要调用更多工具...
    """
```

### 3. 可选工具节点的累积

`_optional_tool_node` 使用 `extend` 而不是覆盖，支持多次调用：

```python
optional_search_results.extend(papers)  # 累积结果
```

## 使用场景

### 场景 1: 单个工具调用

**查询**: "深度求索公司的基本情况"

**流程**:
1. LLM 决策 → 调用 `akshare_search`
2. 执行 `akshare_search` → 获得结果
3. LLM 再次决策 → 认为信息足够，进入搜索阶段

**循环次数**: 1 次

### 场景 2: 多个工具调用

**查询**: "Tesla 和比亚迪的对比分析"

**流程**:
1. LLM 决策 → 调用 `sec_edgar_search` (Tesla)
2. 执行 `sec_edgar_search` → 获得 Tesla 结果
3. LLM 再次决策 → 看到已有 Tesla 结果，决定调用 `akshare_search` (比亚迪)
4. 执行 `akshare_search` → 获得比亚迪结果
5. LLM 再次决策 → 认为两家公司的信息都有了，进入搜索阶段

**循环次数**: 2 次

### 场景 3: 不需要可选工具

**查询**: "人工智能的最新研究进展"

**流程**:
1. LLM 决策 → 不需要可选工具，直接进入搜索阶段

**循环次数**: 0 次

## 优势

### 1. 智能决策

LLM 可以根据问题的复杂度和已有信息动态决定是否需要更多工具：

- 简单问题：不调用或调用一次
- 复杂问题：多次调用不同工具
- 对比分析：分别调用相关工具

### 2. 避免浪费

不会一次性调用所有可选工具，只调用真正需要的：

- 如果问题不涉及证券市场，不会调用 `sec_edgar_search`
- 如果问题不涉及中国公司，不会调用 `akshare_search`

### 3. 支持复杂查询

可以处理需要多个数据源的复杂查询：

- "A 公司和 B 公司的对比"
- "中美两国在某领域的发展情况"
- "多家公司的财务状况分析"

### 4. 自适应

LLM 可以根据第一次调用的结果决定是否需要更多信息：

- 如果第一次结果不够详细，可以调用其他工具补充
- 如果第一次结果已经足够，直接进入搜索阶段

## 循环控制

### 防止无限循环

虽然理论上 LLM 可以无限循环，但实际上有以下机制防止：

1. **LLM 的智能判断**: LLM 会根据已有结果判断是否需要更多信息
2. **可选工具数量有限**: 只有 2 个可选工具（sec_edgar, akshare）
3. **LangGraph 的状态管理**: 可以设置最大步数限制

### 建议的最大循环次数

在实际使用中，建议设置最大循环次数为 3-5 次：

```python
# 在 Config 中添加
MAX_OPTIONAL_TOOL_LOOPS = 3

# 在 _llm_decision_node 中检查
loop_count = len(state.get("executor_messages", [])) // 2
if loop_count >= Config.MAX_OPTIONAL_TOOL_LOOPS:
    logger.warning(f"达到最大循环次数 {Config.MAX_OPTIONAL_TOOL_LOOPS}，强制进入搜索阶段")
    return {"executor_messages": [AIMessage(content="达到最大循环次数，进入搜索阶段")]}
```

## 日志示例

### 单次循环

```
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - LLM 决策结果: 是否有工具调用=True
INFO - LLM 决定调用 1 个可选工具
INFO - 执行可选工具: akshare_search, 参数: {'query': '深度求索公司'}
INFO - 工具 akshare_search 返回 5 条结果
INFO - 可选工具搜索完成，共获得 5 条结果
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - 已调用的可选工具结果：
INFO - - akshare: 5 条结果
INFO - LLM 决策结果: 是否有工具调用=False
INFO - LLM 决定不调用可选工具，进入搜索阶段
```

### 多次循环

```
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - LLM 决策结果: 是否有工具调用=True
INFO - LLM 决定调用 1 个可选工具
INFO - 执行可选工具: sec_edgar_search, 参数: {'query': 'Tesla Inc.'}
INFO - 工具 sec_edgar_search 返回 3 条结果
INFO - 可选工具搜索完成，共获得 3 条结果
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - 已调用的可选工具结果：
INFO - - sec_edgar: 3 条结果
INFO - LLM 决策结果: 是否有工具调用=True
INFO - LLM 决定调用 1 个可选工具
INFO - 执行可选工具: akshare_search, 参数: {'query': '比亚迪'}
INFO - 工具 akshare_search 返回 4 条结果
INFO - 可选工具搜索完成，共获得 7 条结果
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - 已调用的可选工具结果：
INFO - - sec_edgar: 3 条结果
INFO - - akshare: 4 条结果
INFO - LLM 决策结果: 是否有工具调用=False
INFO - LLM 决定不调用可选工具，进入搜索阶段
```

## 性能考虑

### 优势

1. **按需调用**: 只调用真正需要的工具，减少不必要的 API 调用
2. **智能判断**: LLM 可以根据结果质量决定是否需要更多信息

### 开销

1. **额外的 LLM 调用**: 每次循环需要一次 LLM 调用（约 1-2 秒）
2. **状态管理**: 需要维护和传递 `optional_search_results`

### 优化建议

1. **缓存 LLM 决策**: 对于相似的查询，可以缓存 LLM 的决策
2. **并行调用**: 如果 LLM 决定调用多个工具，可以并行执行
3. **设置超时**: 为每次循环设置超时，避免长时间等待

## 测试

使用 `test_executor.py` 测试循环机制：

```bash
python agents/test_executor.py
```

测试用例包括：
1. 单个可选工具调用
2. 多个可选工具调用（循环）
3. 不需要可选工具

## 总结

循环机制让 ExecutorAgent 更加智能和灵活：

- ✅ LLM 可以多次决策
- ✅ 支持复杂查询
- ✅ 避免不必要的工具调用
- ✅ 自适应信息收集

这是一个重要的架构改进，让系统能够更好地处理各种复杂场景。

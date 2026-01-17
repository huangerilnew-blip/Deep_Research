# ReAct 模式在 ExecutorAgent 中的应用

## 什么是 ReAct？

ReAct (Reasoning + Acting) 是一种让 LLM 交替进行推理和行动的模式，由 Yao et al. (2022) 提出。

### ReAct 的核心思想

1. **Thought (思考)**: LLM 分析当前情况，推理下一步应该做什么
2. **Action (行动)**: LLM 决定调用哪个工具或采取什么行动
3. **Observation (观察)**: 工具返回执行结果
4. **循环**: 重复上述过程直到问题解决

## 在 ExecutorAgent 中的实现

### 流程图

```
START
  ↓
┌─────────────────────────────────────┐
│ llm_decision_node (Thought + Action)│
│ - 分析问题和已有信息                 │
│ - 决定是否需要调用工具               │
└─────────────────────────────────────┘
  ↓
[条件边: 是否有 tool_calls?]
  ↓                              ↓
  YES                           NO
  ↓                              ↓
┌─────────────────┐         ┌─────────────┐
│optional_tool_node│         │ search_node │
│ (Action)         │         │ (完成决策)   │
│ - 执行工具       │         └─────────────┘
│ - 获取结果       │              ↓
└─────────────────┘         clean_and_rerank
  ↓                              ↓
┌─────────────────────────────────────┐  download
│ llm_decision_node (Observation)     │     ↓
│ - 看到工具执行结果                   │  summarize
│ - 推理是否需要更多信息               │     ↓
└─────────────────────────────────────┘    END
  ↓
[条件边: 继续循环或进入搜索?]
  ↓           ↓
  YES        NO
  ↑           ↓
  └─循环─→ search_node (合并所有结果)
              ↓
         clean_and_rerank
              ↓
         download
              ↓
         summarize
              ↓
            END
```

### 代码实现

#### 1. Thought + Action (llm_decision_node)

```python
async def _llm_decision_node(self, state: ExecutorState) -> Dict:
    """
    LLM 决策节点：按照 ReAct 模式让 LLM 进行推理和决策
    """
    query = state["current_query"]
    optional_search_results = state.get("optional_search_results", [])
    
    # 如果有已执行的工具结果，格式化为观察信息
    if optional_search_results:
        observation = self._format_observation(optional_search_results)
        
        prompt = f"""你是一个智能研究助手，正在使用 ReAct 方法解决问题。

问题 (Question): {query}

已执行的操作和观察结果:
Observation:
{observation}

现在请进行推理和决策：
1. Thought (思考): 分析已有信息是否足够回答问题
2. Action (行动): 如果需要更多信息，调用相应的工具
"""
    else:
        prompt = f"""你是一个智能研究助手，正在使用 ReAct 方法解决问题。

问题 (Question): {query}

请按照以下格式思考和行动：
1. Thought (思考): 分析这个问题需要什么信息
2. Action (行动): 决定是否需要调用工具
"""
    
    response = await llm_with_tools.ainvoke(prompt)
    return {"executor_messages": [response]}
```

#### 2. Observation 格式化 (_format_observation)

```python
def _format_observation(self, papers: List[Dict]) -> str:
    """
    格式化工具执行结果为观察信息
    """
    observations = []
    for paper in papers:
        source = paper.get("source", "unknown")
        
        # 对于 sec_edgar，提取部分摘要内容
        if source == "sec_edgar":
            abstract = paper.get("abstract", "")
            preview = abstract[:200] + "..." if len(abstract) > 200 else abstract
            company_name = paper.get("extra", {}).get("company_name", title)
            observations.append(
                f"✓ 成功获取 {company_name} 的完整信息\n"
                f"  来源: SEC EDGAR\n"
                f"  预览: {preview}"
            )
    
    return "\n\n".join(observations)
```

#### 3. Action Execution (optional_tool_node)

```python
async def _optional_tool_node(self, state: ExecutorState) -> Dict:
    """
    可选工具节点：执行 LLM 决定调用的可选工具
    """
    last_message = state["executor_messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        
        # 执行工具 (Action)
        tool = next((t for t in optional_tools if t.name == tool_name), None)
        result = await tool.ainvoke(tool_args)
        
        # 收集结果 (Observation)
        new_results.extend(result)
    
    return {"optional_search_results": all_results}
```

## 实际案例

### 案例 1: 单个公司查询

**Question**: "深度求索公司的基本情况"

**第一轮**:
```
Thought: 深度求索是一家中国公司，需要使用 akshare_search 获取其基本信息
Action: 调用 akshare_search(query="深度求索")
```

**Observation**:
```
✓ 成功获取深度求索的完整信息
  来源: AkShare (中国股市数据)
  预览: 深度求索（DeepSeek）是一家专注于人工智能技术研发的公司...
```

**第二轮**:
```
Thought: 已经成功获取了深度求索的完整信息，包括公司基本情况，信息已足够
Action: 回复"信息已足够，可以继续"
```

### 案例 2: 多公司对比

**Question**: "Tesla 和比亚迪的对比分析"

**第一轮**:
```
Thought: 这个问题涉及两家公司的对比。Tesla 是美国上市公司，比亚迪是中国上市公司。
         需要分别获取两家公司的信息。先获取 Tesla 的信息。
Action: 调用 sec_edgar_search(query="Tesla Inc.")
```

**Observation**:
```
✓ 成功获取 Tesla Inc. 的完整信息
  来源: SEC EDGAR
  预览: Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles...
```

**第二轮**:
```
Thought: 已经获取了 Tesla 的信息，但还缺少比亚迪的信息。需要调用 akshare_search。
Action: 调用 akshare_search(query="比亚迪")
```

**Observation**:
```
✓ 成功获取 Tesla Inc. 的完整信息
  来源: SEC EDGAR
  预览: Tesla, Inc. designs, develops, manufactures...

✓ 成功获取比亚迪的完整信息
  来源: AkShare (中国股市数据)
  预览: 比亚迪股份有限公司成立于1995年，是一家中国汽车制造商...
```

**第三轮**:
```
Thought: 已经成功获取了 Tesla 和比亚迪两家公司的完整信息，可以进行对比分析了。
Action: 回复"信息已足够，可以继续"
```

### 案例 3: 不需要可选工具

**Question**: "人工智能的最新研究进展"

**第一轮**:
```
Thought: 这个问题是关于人工智能的学术研究，不涉及特定的上市公司信息。
         不需要使用 sec_edgar_search 或 akshare_search。
Action: 回复"不需要专业工具"
```

## ReAct 模式的优势

### 1. 透明的推理过程

LLM 的思考过程是可见的，我们可以看到：
- LLM 如何分析问题
- LLM 为什么选择某个工具
- LLM 如何根据观察结果调整策略

### 2. 更智能的决策

通过看到工具执行的结果，LLM 可以：
- 判断是否需要更多信息
- 避免重复调用相同的工具
- 根据已有信息调整查询策略

### 3. 支持复杂任务

ReAct 模式天然支持多步骤任务：
- 多个公司的对比分析
- 需要多个数据源的综合查询
- 需要根据中间结果调整策略的任务

### 4. 更好的错误处理

如果工具调用失败或返回不完整的结果，LLM 可以：
- 识别问题
- 尝试其他工具
- 调整查询参数

## 与传统方法的对比

### 传统方法（硬编码规则）

```python
if "中国" in query or "上市" in query:
    call_akshare()
if "美国" in query or "NYSE" in query:
    call_sec_edgar()
```

**问题**:
- 规则固定，难以处理复杂情况
- 无法根据中间结果调整
- 可能调用不必要的工具

### ReAct 方法（LLM 驱动）

```python
# LLM 自主推理和决策
Thought: 分析问题，判断需要什么信息
Action: 调用相应的工具
Observation: 看到结果
Thought: 根据结果判断是否需要更多信息
Action: 继续或完成
```

**优势**:
- 灵活适应各种情况
- 根据实际结果调整策略
- 只调用真正需要的工具

## 日志示例

### ReAct 循环的日志

```
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - LLM 决策结果: 是否有工具调用=True
INFO - LLM 思考内容: Thought: 这个问题涉及 Tesla 公司，需要使用 sec_edgar_search...
INFO - LLM 决定调用 1 个可选工具
INFO - 执行可选工具 (Action): sec_edgar_search, 参数: {'query': 'Tesla Inc.'}
INFO - 工具 sec_edgar_search 返回 3 条结果 (Observation)
INFO - 可选工具搜索完成，本次新增 3 条结果，累计 3 条结果
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - LLM 思考内容: Thought: 已经获取了 Tesla 的信息，还需要比亚迪的信息...
INFO - LLM 决策结果: 是否有工具调用=True
INFO - LLM 决定调用 1 个可选工具
INFO - 执行可选工具 (Action): akshare_search, 参数: {'query': '比亚迪'}
INFO - 工具 akshare_search 返回 4 条结果 (Observation)
INFO - 可选工具搜索完成，本次新增 4 条结果，累计 7 条结果
INFO - 让 LLM 决定是否调用可选工具: ['sec_edgar_search', 'akshare_search']
INFO - LLM 思考内容: Thought: 两家公司的信息都已获取，可以进行对比分析了...
INFO - LLM 决策结果: 是否有工具调用=False
INFO - LLM 决定不调用可选工具，进入搜索阶段
```

## 配置和优化

### 提示词优化

可以在 `_llm_decision_node` 中调整提示词：

```python
prompt = f"""你是一个智能研究助手，正在使用 ReAct 方法解决问题。

问题 (Question): {query}

请按照以下格式思考：
1. Thought (思考): 详细分析问题，列出需要的信息
2. Action (行动): 明确说明要调用哪个工具和原因

注意事项：
- 避免重复调用相同的工具
- 如果信息已足够，明确说明"信息已足够"
- 对于对比类问题，确保每个对象都有相应的数据
"""
```

### 观察结果优化

可以在 `_format_observation` 中调整展示格式：

```python
def _format_observation(self, papers: List[Dict]) -> str:
    """格式化观察结果"""
    observations = []
    
    for paper in papers:
        source = paper.get("source", "unknown")
        
        if source == "sec_edgar":
            # 提取更多关键信息
            abstract = paper.get("abstract", "")
            preview = abstract[:300]  # 增加预览长度
            
            # 提取关键指标
            extra = paper.get("extra", {})
            metrics = []
            if extra.get("revenue"):
                metrics.append(f"营收: {extra['revenue']}")
            if extra.get("market_cap"):
                metrics.append(f"市值: {extra['market_cap']}")
            
            observation = f"✓ 成功获取 {company_name} 的完整信息\n"
            observation += f"  来源: SEC EDGAR\n"
            if metrics:
                observation += f"  关键指标: {', '.join(metrics)}\n"
            observation += f"  预览: {preview}"
            
            observations.append(observation)
    
    return "\n\n".join(observations)
```

## 总结

ReAct 模式让 ExecutorAgent 更加智能和透明：

- ✅ **透明推理**: 可以看到 LLM 的思考过程
- ✅ **智能决策**: 根据观察结果动态调整策略
- ✅ **支持复杂任务**: 天然支持多步骤、多工具的任务
- ✅ **更好的错误处理**: 可以根据结果调整策略
- ✅ **避免浪费**: 只调用真正需要的工具

这是一个重要的架构改进，让系统能够更好地模拟人类的推理和行动过程。

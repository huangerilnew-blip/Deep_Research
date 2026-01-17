# ToolMessage 类型迁移说明

## 概述

将 `ExecutorState` 中的 `optional_search_results` 字段从 `List[Dict]` 迁移到 `Annotated[list[ToolMessage], add_messages]`，以更好地利用 LangGraph 的消息管理机制。

## 修改内容

### 1. 状态定义修改

**修改前**:
```python
class ExecutorState(TypedDict):
    executor_messages: Annotated[list[AnyMessage], add_messages]
    current_query: str
    optional_search_results: List[Dict]  # 可选工具的搜索结果
    search_results: List[Dict]
    reranked_results: List[Dict]
    downloaded_papers: List[Dict]
    executor_result: Dict
```

**修改后**:
```python
class ExecutorState(TypedDict):
    executor_messages: Annotated[list[AnyMessage], add_messages]
    current_query: str
    optional_search_results: Annotated[list[ToolMessage], add_messages]  # 可选工具的搜索结果
    search_results: List[Dict]
    reranked_results: List[Dict]
    downloaded_papers: List[Dict]
    executor_result: Dict
```

### 2. 导入语句修改

添加 `ToolMessage` 导入：

```python
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
```

### 3. `_format_observation` 方法修改

**修改前**:
```python
def _format_observation(self, papers: List[Dict]) -> str:
    """格式化工具执行结果为观察信息"""
    if not papers:
        return "未找到相关信息"
    
    observations = []
    for paper in papers:
        source = paper.get("source", "unknown")
        # ... 处理 paper
    
    return "\n\n".join(observations)
```

**修改后**:
```python
def _format_observation(self, tool_messages: List[ToolMessage]) -> List[ToolMessage]:
    """格式化工具执行结果为观察信息，返回格式化后的 ToolMessage 列表"""
    if not tool_messages:
        return []
    
    formatted_messages = []
    
    for tool_msg in tool_messages:
        # 从 ToolMessage 中提取内容
        content = tool_msg.content
        
        # 解析 JSON 内容
        try:
            if isinstance(content, str):
                papers = json.loads(content)
            elif isinstance(content, list):
                papers = content
            else:
                formatted_messages.append(tool_msg)
                continue
            
            # 格式化每个 paper 的观察信息
            observations = []
            for paper in papers:
                source = paper.get("source", "unknown")
                # ... 处理 paper，生成观察文本
                observations.append(observation_text)
            
            # 创建格式化后的 ToolMessage
            if observations:
                formatted_content = "\n\n".join(observations)
                formatted_msg = ToolMessage(
                    content=formatted_content,
                    tool_call_id=tool_msg.tool_call_id,
                    name=tool_msg.name
                )
                formatted_messages.append(formatted_msg)
        
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"解析 ToolMessage 内容失败: {e}")
            formatted_messages.append(tool_msg)  # 保留原始消息
            continue
    
    return formatted_messages
```

**关键变化**:
- 返回类型从 `str` 改为 `List[ToolMessage]`
- 将格式化后的文本包装回 ToolMessage
- 保留原始的 `tool_call_id` 和 `name`
- 解析失败时保留原始消息

### 4. `_optional_tool_node` 方法修改

**修改前**:
```python
async def _optional_tool_node(self, state: ExecutorState) -> Dict:
    """执行可选工具"""
    optional_search_results = state.get("optional_search_results", [])
    new_results = []
    
    for tool_call in last_message.tool_calls:
        # 执行工具
        result = await tool.ainvoke(tool_args)
        
        # 解析结果
        if isinstance(result, str):
            papers = json.loads(result)
        elif isinstance(result, list):
            papers = result
        
        new_results.extend(papers)
    
    # 累积结果
    all_results = optional_search_results + new_results
    return {"optional_search_results": all_results}
```

**修改后**:
```python
async def _optional_tool_node(self, state: ExecutorState) -> Dict:
    """执行可选工具，返回 ToolMessage"""
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id', '')
        
        # 执行工具
        result = await tool.ainvoke(tool_args)
        
        if result:
            # 创建 ToolMessage
            tool_message = ToolMessage(
                content=result if isinstance(result, str) else json.dumps(result),
                tool_call_id=tool_call_id,
                name=tool_name
            )
            tool_messages.append(tool_message)
    
    return {"optional_search_results": tool_messages}
```

### 5. `_search_node` 方法修改

**修改前**:
```python
# 合并可选工具的搜索结果
optional_results = state.get("optional_search_results", [])
if optional_results:
    logger.info(f"合并 {len(optional_results)} 条可选工具搜索结果")
    search_results.extend(optional_results)
```

**修改后**:
```python
# 合并可选工具的搜索结果（从 ToolMessage 中提取）
optional_tool_messages = state.get("optional_search_results", [])
if optional_tool_messages:
    logger.info(f"合并 {len(optional_tool_messages)} 条可选工具搜索结果")
    
    for tool_msg in optional_tool_messages:
        try:
            content = tool_msg.content
            
            # 解析 ToolMessage 的内容
            if isinstance(content, str):
                papers = json.loads(content)
            elif isinstance(content, list):
                papers = content
            else:
                continue
            
            if isinstance(papers, list):
                search_results.extend(papers)
                logger.info(f"从 ToolMessage 中提取 {len(papers)} 条结果")
        
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"解析 ToolMessage 内容失败: {e}")
            continue
```

## 优势

### 1. 更好的消息管理

使用 `Annotated[list[ToolMessage], add_messages]` 可以利用 LangGraph 的消息累积机制：
- 自动管理消息列表
- 支持消息的追加和合并
- 与其他消息类型（AIMessage, HumanMessage）保持一致

### 2. 更清晰的类型定义

`ToolMessage` 是 LangChain 的标准消息类型，包含：
- `content`: 工具返回的内容
- `tool_call_id`: 工具调用的唯一标识
- `name`: 工具名称

### 3. 格式化后仍保持 ToolMessage 类型

`_format_observation` 方法返回 `List[ToolMessage]`，而不是字符串：
- 保持类型一致性
- 保留 `tool_call_id` 和 `name` 信息
- 可以在后续流程中继续使用
- 格式化后的内容更易于 LLM 理解

### 4. 更好的可追溯性

每个 ToolMessage 都关联到特定的 tool_call，可以追溯：
- 哪个工具被调用
- 调用的参数是什么
- 返回的结果是什么
- 格式化后的观察信息

### 5. 符合 LangGraph 最佳实践

使用标准的消息类型而不是自定义的 Dict，更符合 LangGraph 的设计理念。

## 兼容性

### 向后兼容

由于 `add_messages` 的累积特性，旧的代码逻辑（累积结果）仍然有效：
- 每次调用 `_optional_tool_node` 都会追加新的 ToolMessage
- `_llm_decision_node` 可以看到所有历史的 ToolMessage
- `_search_node` 可以提取所有 ToolMessage 的内容

### 数据提取

从 ToolMessage 提取数据的模式：
```python
for tool_msg in tool_messages:
    content = tool_msg.content
    
    # 解析 JSON
    if isinstance(content, str):
        data = json.loads(content)
    elif isinstance(content, list):
        data = content
    
    # 使用 data
```

## 测试建议

### 1. 单元测试

测试 ToolMessage 的创建和解析：
```python
def test_tool_message_creation():
    result = json.dumps([{"title": "Test", "source": "test"}])
    tool_msg = ToolMessage(
        content=result,
        tool_call_id="test_id",
        name="test_tool"
    )
    
    assert tool_msg.content == result
    assert tool_msg.tool_call_id == "test_id"
    assert tool_msg.name == "test_tool"
```

### 2. 集成测试

测试完整的循环流程：
```python
async def test_optional_tool_loop():
    executor = ExecutorAgent(pool)
    result = await executor.invoke(
        query="Tesla 和比亚迪的对比",
        thread_id="test_thread"
    )
    
    # 验证结果包含两个公司的信息
    assert "Tesla" in str(result)
    assert "比亚迪" in str(result)
```

### 3. 消息累积测试

测试多次调用可选工具：
```python
async def test_message_accumulation():
    # 第一次调用
    state1 = await executor._optional_tool_node(state)
    assert len(state1["optional_search_results"]) == 1
    
    # 第二次调用（应该累积）
    state2 = await executor._optional_tool_node(state1)
    assert len(state2["optional_search_results"]) == 2
```

## 注意事项

### 1. JSON 解析

ToolMessage 的 content 可能是：
- JSON 字符串：需要 `json.loads()`
- 列表：直接使用
- 其他类型：需要特殊处理

### 2. 错误处理

始终使用 try-except 处理 JSON 解析：
```python
try:
    papers = json.loads(content)
except (json.JSONDecodeError, TypeError) as e:
    logger.warning(f"解析失败: {e}")
    continue
```

### 3. tool_call_id

确保 tool_call_id 正确传递：
```python
tool_call_id = tool_call.get('id', '')  # 从 AIMessage 的 tool_calls 中获取
```

## 总结

这次迁移将 `optional_search_results` 从简单的 `List[Dict]` 升级为标准的 `Annotated[list[ToolMessage], add_messages]`，带来了：

- ✅ 更好的类型安全
- ✅ 更清晰的消息管理
- ✅ 更好的可追溯性
- ✅ 符合 LangGraph 最佳实践
- ✅ 保持向后兼容

所有相关代码已经更新，包括：
- `_format_observation`: 从 ToolMessage 提取内容
- `_optional_tool_node`: 返回 ToolMessage
- `_search_node`: 从 ToolMessage 提取数据

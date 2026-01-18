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

- 工具：只配置search类工具
- 必要工具：
  - wiki_search（md格式保存）：没有摘要，返回的是完整文档
  - openalex_search（pdf格式保存）:一般都有摘要--paper.abstract
  - sematic_scholar_search（pdf格式保存）：经常没有摘要，有的话就在--paper.abstract
  - Tavily_search（md格式保存）:有摘要（已设置），保存在--paper.abstract
- 可选工具
  - sec_edgar_search（md格式保存）：无摘要，paper.abstract保存了较为完整的信息
  - akshare_search(md格式)：无摘要，paper.abstract保存了较为完整的信息
- 流程:
  - _llm_decision_node:llm判断是否要调用可选工具
  - _optional_tool_node：条件工具的执行
  - search_node:必要工具的执行（异步执行）
  - 

## build_vector_store
- 功能:构建向量数据库
- 流程:
    - 使用mineru对pdf进行解析，提取文本内容
    - 其他文档内容，如txt,md等，使用常规llamaindex方法进行处理
    - 数据清洗
    - 入库
## search_info
- 功能:检索相关信息
- 流程:
# pipeline
## PlannerAgent
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
## ExecutorAgent
- 功能:负责执行PlannerAgent生成的子任务，通过工具进行调用，并返回结果。
- 问题：
  1. 如果直接返回完整的结果，容易导致上下文长度太长
  2. 使用检索器的排序，并不准确
- 解决方案：
  1. 返回结果摘要，控制上下文长度
  2. 使用bge的bert模型进行rerank
  - 输出格式json：
```
  {
      "results": [
          {"id": "T1", "result": "子问题1的结果摘要","paper_id":"id-1",score": 0.95"},
          {"id": "T2", "result": "子问题2的结果摘要","paper_id":"id-90","score": 0.90"},
          {"id": "T3", "result": "子问题3的结果摘要","paper_id":"id-0054","score": 0.88"},
          {"id": "T4", "result": "子问题4的结果摘要","paper_id":"id-7009","score": 0.80"}
      ]
  }
```

- 流程
    - 按照DAG思想，对任务进行拓扑排序，确保依赖的任务先执行
    - 依次执行每个任务，调用相应的工具
    - 收集每个任务的多个结果，根据摘要进行rerank，选择最优结果返回
    - 根据paper_id和来源，下载pdf

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
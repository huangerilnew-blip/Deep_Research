# 实现计划: Multi-Agent 深度搜索系统

## 概述

本实现计划将设计文档转化为一系列可执行的编码任务。每个任务都是增量式的，基于前面的任务构建，最终完成整个系统的实现。

## 任务列表

- [x] 1. 配置管理扩展
  - 在 `core/config.py` 中添加新的配置项
  - 添加 `EXECUTOR_POOL_SIZE`（ExecutorAgent 池大小，默认 3）
  - 添加 `MAX_CHUNK_SIZE`（Markdown 切割最大长度，默认 1000）
  - 添加 `MINERU_BASE_URL`（MinerU 服务地址）
  - 添加 `VLLM_BASE_URL`（vllm Embedding 服务地址）
  - 添加 `EMBEDDING_MODEL_NAME`（Embedding 模型名称）
  - _需求: 9.1, 9.3_

- [x] 2. 向量存储管理器实现
  - [x] 2.1 创建 `core/vector_store_manager.py`
    - 实现 `VectorStoreManager` 类
    - 实现 `_get_local_embedding_model()` 方法，使用 vllm 部署的本地模型
    - 实现 `load_base_vector_store()` 方法，加载 restructure_company_info.json
    - 实现 `add_documents()` 方法，支持增量添加文档
    - 实现 `get_retriever()` 方法，返回检索器
    - _需求: 5.1, 5.2, 5.7_
  
  - [ ]* 2.2 编写属性测试：本地 Embedding 模型使用
    - **属性 12: 本地 Embedding 模型使用**
    - **验证需求: 5.2**
  
  - [ ]* 2.3 编写属性测试：向量库增量更新
    - **属性 17: 向量库增量更新**
    - **验证需求: 5.7**

- [x] 3. 文档处理器实现
  - [x] 3.1 创建 `core/document_processor.py`
    - 实现 `DocumentProcessor` 类
    - 实现 `_pdf_to_markdown()` 方法，调用 vllm 部署的 MinerU 服务
    - 实现 `_split_by_headers()` 方法，按 H1/H2/H3 标题切割
    - 实现 `_split_by_sentences()` 方法，按语句边界切割超长文本
    - 实现 `_rewrite_to_questions()` 方法，使用 LLM 从文档提取问题
    - 实现 `process_documents()` 方法，整合所有处理步骤
    - _需求: 5.3, 5.4, 5.5, 6.3, 6.4_
  
  - [ ]* 3.2 编写属性测试：PDF 到 Markdown 转换
    - **属性 13: PDF 到 Markdown 转换**
    - **验证需求: 5.3**
  
  - [ ]* 3.3 编写属性测试：Markdown 按标题切割
    - **属性 14: Markdown 按标题切割**
    - **验证需求: 5.4**
  
  - [ ]* 3.4 编写属性测试：Chunk 长度限制
    - **属性 15: Chunk 长度限制**
    - **验证需求: 5.5**
  
  - [ ]* 3.5 编写属性测试：问题改写格式
    - **属性 18: 问题改写格式**
    - **验证需求: 6.3**

- [x] 4. Checkpoint - 确保文档处理模块测试通过
  - 确保所有测试通过，如有问题请询问用户

- [x] 5. Questions Pool 数据模型
  - [x] 5.1 创建 `core/models.py`
    - 实现 `QuestionsPool` 数据类
    - 实现 `get_all_questions()` 方法
    - 实现 `add_rewritten_questions()` 方法
    - 实现去重逻辑
    - 实现 `DocumentMetadata`、`MarkdownChunk`、`RetrievedContext` 数据类
    - _需求: 1.3, 6.4, 6.5_
  
  - [ ]* 5.2 编写属性测试：Questions Pool 去重
    - **属性 20: Questions Pool 去重**
    - **验证需求: 6.5**

- [x] 6. ExecutorAgent Pool 实现
  - [x] 6.1 创建 `agents/executor_pool.py`
    - 实现 `ExecutorAgentPool` 类
    - 实现 `_initialize_agents()` 方法，创建指定数量的 ExecutorAgent
    - 实现 `execute_questions()` 方法，并发执行多个子问题
    - 实现异常处理，确保单个失败不影响其他
    - 实现 `cleanup()` 方法，清理所有 Agent 资源
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 6.2 编写属性测试：ExecutorAgent Pool 大小一致性
    - **属性 4: ExecutorAgent Pool 大小一致性**
    - **验证需求: 2.1**
  
  - [ ]* 6.3 编写属性测试：并发执行完整性
    - **属性 5: 并发执行完整性**
    - **验证需求: 2.2, 2.4**
  
  - [ ]* 6.4 编写属性测试：执行独立性和容错性
    - **属性 6: 执行独立性和容错性**
    - **验证需求: 2.3, 2.5**

- [x] 7. RAG 模块实现
  - [x] 7.1 创建 `core/rag_module.py`
    - 实现 `RAGModule` 类
    - 实现 `_retrieve_contexts()` 方法，使用 Questions Pool 检索
    - 实现 `_deduplicate_contexts()` 方法，基于内容哈希去重
    - 实现 `_rerank_contexts()` 方法，调用 BGEReranker 重排序
    - 实现 `_generate_answer()` 方法，构建提示词并调用 LLM
    - 实现 `retrieve_and_generate()` 方法，整合完整流程
    - _需求: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3_
  
  - [ ]* 7.2 编写属性测试：检索查询覆盖
    - **属性 22: 检索查询覆盖**
    - **验证需求: 7.2**
  
  - [ ]* 7.3 编写属性测试：检索结果去重
    - **属性 24: 检索结果去重**
    - **验证需求: 7.4**
  
  - [ ]* 7.4 编写属性测试：Rerank 结果排序
    - **属性 25: Rerank 结果排序**
    - **验证需求: 7.5**
  
  - [ ]* 7.5 编写属性测试：答案生成提示词完整性
    - **属性 26: 答案生成提示词完整性**
    - **验证需求: 8.1**

- [x] 8. Checkpoint - 确保 RAG 模块测试通过
  - 确保所有测试通过，如有问题请询问用户

- [x] 9. MultiAgent 协调器实现
  - [x] 9.1 创建 `agents/multi_agent.py`
    - 实现 `MultiAgent` 类
    - 实现 `__init__()` 方法，初始化 PlannerAgent、ExecutorAgentPool、VectorStoreManager
    - 实现 `_initialize_vector_store()` 方法，加载基础向量库
    - 实现 `process_query()` 方法，编排完整流程：
      - 调用 PlannerAgent 拆解查询
      - 初始化 Questions Pool
      - 调用 ExecutorAgent Pool 并发执行
      - 调用 DocumentProcessor 处理文档
      - 更新 Questions Pool（添加改写问题）
      - 向量化并入库
      - 调用 RAG Module 检索和生成
    - 实现 `_cleanup()` 方法，清理所有资源
    - _需求: 1.1, 1.2, 1.3, 2.1, 5.1, 6.4, 7.1, 8.4_
  
  - [ ]* 9.2 编写属性测试：查询拆解最小数量
    - **属性 1: 查询拆解最小数量**
    - **验证需求: 1.1**
  
  - [ ]* 9.3 编写属性测试：子问题存储完整性
    - **属性 2: 子问题存储完整性**
    - **验证需求: 1.3**
  
  - [ ]* 9.4 编写属性测试：Questions Pool 更新
    - **属性 19: Questions Pool 更新**
    - **验证需求: 6.4**

- [x] 10. 错误处理和日志增强
  - [x] 10.1 在所有模块中添加错误处理
    - 在 `MultiAgent.process_query()` 中添加 try-except 块
    - 在 `ExecutorAgentPool.execute_questions()` 中处理部分失败
    - 在 `DocumentProcessor` 中处理转换失败
    - 在 `RAGModule` 中实现降级策略
    - 添加友好的错误消息
    - _需求: 11.1, 11.2, 11.3_
  
  - [ ]* 10.2 编写属性测试：异常日志记录
    - **属性 35: 异常日志记录**
    - **验证需求: 11.1**
  
  - [ ]* 10.3 编写属性测试：错误消息友好性
    - **属性 36: 错误消息友好性**
    - **验证需求: 11.3**

- [ ] 11. API 接口实现
  - [x] 11.1 创建 `api/routes.py`
    - 实现 `/query` POST 端点，接收用户查询
    - 实现请求验证
    - 调用 `MultiAgent.process_query()`
    - 返回结构化响应（包含回答、检索来源、元数据）
    - 实现错误处理和状态码
    - _需求: 8.4_
  
  - [x] 11.2 更新 `main.py`
    - 初始化 PostgreSQL 连接池
    - 初始化 MultiAgent 实例
    - 注册路由
    - 添加启动和关闭事件处理
    - _需求: 9.1, 10.1_

- [x] 12. 集成测试
  - [ ]* 12.1 编写端到端集成测试
    - 测试完整查询流程
    - 测试并发执行
    - 测试错误恢复
    - 测试向量库更新
    - _需求: 所有需求_

- [x] 13. Checkpoint - 最终测试和验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 14. 文档和配置
  - [x] 14.1 创建 `docs/API.md`
    - 记录 API 接口规范
    - 提供请求/响应示例
    - 说明错误码
  
  - [x] 14.2 创建 `docs/DEPLOYMENT.md`
    - 说明依赖服务部署（PostgreSQL, Chroma, vllm, TEI）
    - 提供配置示例
    - 说明环境变量设置
  
  - [x] 14.3 更新 `requirements.txt`
    - 添加新依赖：hypothesis（属性测试）
    - 添加 llama-index 相关包
    - 确保所有版本兼容

## 注意事项

- 任务标记 `*` 的为可选测试任务，可以跳过以加快 MVP 开发
- 每个任务都引用了具体的需求编号，便于追溯
- Checkpoint 任务用于确保增量验证
- 属性测试每个至少运行 100 次迭代
- 所有代码应该在 WSL 的 conda 环境 `agent_backend` 中测试
- 安装新包时使用清华镜像源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>`

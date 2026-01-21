# 需求文档

## 介绍

本文档定义了一个 multi-agent 深度搜索系统的需求，该系统通过多智能体协作实现深度文献检索、处理和问答生成。系统将用户的复杂查询拆解为多个子问题，并发执行搜索和文档处理，最终基于检索增强生成（RAG）技术生成高质量的回答。

## 术语表

- **System**: Multi-Agent 深度搜索系统
- **PlannerAgent**: 查询规划智能体，负责将用户查询拆解为子问题
- **ExecutorAgent**: 执行智能体，负责处理单个子问题的完整流程
- **ExecutorAgent_Pool**: 执行智能体池，包含多个 ExecutorAgent 实例，支持并发执行
- **MultiAgent**: 多智能体协调器，负责整体流程编排
- **Questions_Pool**: 子问题池，存储 PlannerAgent 生成的子问题以及从 Markdown 文档中改写的问题
- **Vector_Store**: 向量数据库，使用 Chroma 存储文档向量
- **RAG_Module**: 检索增强生成模块，负责文档检索和答案生成
- **Document_Processor**: 文档处理器，负责 PDF 转 Markdown 和文档切割
- **Reranker**: 重排序器，使用本地部署的 BGE Reranker 模型
- **Embedding_Model**: 嵌入模型，使用 vllm 部署的本地模型
- **User_Query**: 用户输入的原始查询
- **Sub_Question**: 从 User_Query 拆解出的子问题
- **Search_Result**: 搜索工具返回的文档结果
- **Reranked_Result**: 经过 Reranker 重排序后的文档结果
- **Downloaded_Document**: 已下载到本地的文档
- **Markdown_Chunk**: 切割后的 Markdown 文档片段
- **Retrieved_Context**: 从 Vector_Store 检索到的相关语料
- **Final_Answer**: 系统生成的最终回答

## 需求

### 需求 1: 查询拆解

**用户故事:** 作为用户，我希望系统能够将我的复杂查询拆解为多个具体的子问题，以便进行更深入和全面的搜索。

#### 验收标准

1. WHEN 用户提交 User_Query, THEN THE PlannerAgent SHALL 将其拆解为至少 3 个 Sub_Question
2. WHEN PlannerAgent 拆解查询时, THEN THE PlannerAgent SHALL 确保每个 Sub_Question 具体且明确
3. WHEN PlannerAgent 完成拆解, THEN THE System SHALL 将所有 Sub_Question 存储到 Questions_Pool
4. WHEN PlannerAgent 拆解失败超过配置的最大迭代次数, THEN THE System SHALL 返回错误信息并终止流程
5. WHEN PlannerAgent 生成 Sub_Question, THEN THE System SHALL 以 JSON 格式输出子问题列表

### 需求 2: 并发执行子问题

**用户故事:** 作为系统架构师，我希望系统能够并发处理多个子问题，以提高整体处理效率。

#### 验收标准

1. WHEN MultiAgent 初始化时, THEN THE System SHALL 创建 ExecutorAgent_Pool 包含可配置数量的 ExecutorAgent 实例
2. WHEN Questions_Pool 包含多个 Sub_Question, THEN THE System SHALL 并发分配给 ExecutorAgent_Pool 中的可用 ExecutorAgent
3. WHEN ExecutorAgent 处理 Sub_Question 时, THEN THE System SHALL 独立执行搜索、重排序和下载流程
4. WHEN 所有 ExecutorAgent 完成任务, THEN THE System SHALL 收集所有 Search_Result 和 Downloaded_Document
5. WHEN ExecutorAgent 执行失败, THEN THE System SHALL 记录错误但不中断其他 ExecutorAgent 的执行

### 需求 3: 文档搜索和下载

**用户故事:** 作为用户，我希望系统能够从多个数据源搜索相关文档并下载到本地，以便后续处理。

#### 验收标准

1. WHEN ExecutorAgent 处理 Sub_Question, THEN THE System SHALL 并发调用必需的搜索工具（wikipedia_search, openalex_search, semantic_scholar_search, tavily_search）
2. WHEN 搜索工具返回结果, THEN THE System SHALL 合并所有 Search_Result 到统一列表
3. WHEN ExecutorAgent 需要专业数据源, THEN THE System SHALL 允许 LLM 决策是否调用可选工具（sec_edgar_search, akshare_search）
4. WHEN 搜索完成, THEN THE System SHALL 下载所有 Search_Result 中的文档到配置的 DOC_SAVE_PATH
5. WHEN 下载失败, THEN THE System SHALL 记录错误并继续处理其他文档

### 需求 4: 文档重排序

**用户故事:** 作为系统架构师，我希望系统能够使用 Reranker 对搜索结果进行重排序，以提高检索质量。

#### 验收标准

1. WHEN ExecutorAgent 获得 Search_Result, THEN THE System SHALL 对配置的数据源（openalex, semantic_scholar）的文档进行重排序
2. WHEN Reranker 处理文档, THEN THE System SHALL 使用本地部署的 BGE Reranker 模型计算相关性分数
3. WHEN Reranker 完成评分, THEN THE System SHALL 过滤掉分数低于配置阈值的文档
4. WHEN 过滤完成, THEN THE System SHALL 保留 top_n 个最相关的文档
5. WHEN Reranker 失败, THEN THE System SHALL 返回原始 Search_Result 并记录警告

### 需求 5: 文档处理和向量化

**用户故事:** 作为用户，我希望系统能够将下载的 PDF 文档转换为 Markdown 并切割入库，以便进行语义检索。

#### 验收标准

1. WHEN System 初始化 Vector_Store, THEN THE System SHALL 首先加载 restructure_company_info.json 构造的基础向量数据库
2. WHEN System 构建或更新向量数据库, THEN THE System SHALL 使用本地 vllm 部署的 Embedding_Model 而非远程 API
3. WHEN System 获得 Downloaded_Document, THEN THE Document_Processor SHALL 将 PDF 格式文档转换为 Markdown 格式
4. WHEN Document_Processor 处理 Markdown 文档, THEN THE System SHALL 按标题等级（H1, H2, H3）进行智能切割
5. WHEN Markdown_Chunk 超过配置的最大长度, THEN THE System SHALL 按语句边界进一步切割
6. WHEN 切割完成, THEN THE System SHALL 使用本地 Embedding_Model 对每个 Markdown_Chunk 进行向量化
7. WHEN 向量化完成, THEN THE System SHALL 将新的向量节点添加到已加载的 Vector_Store 中

### 需求 6: 问题改写和扩充

**用户故事:** 作为用户，我希望系统能够从下载的 Markdown 文档中提取关键信息并改写为问题，以丰富 Questions_Pool 并提高检索覆盖率。

#### 验收标准

1. WHEN Document_Processor 完成 Markdown 切割, THEN THE System SHALL 分析每个 Markdown_Chunk 的内容
2. WHEN System 分析 Markdown_Chunk, THEN THE System SHALL 识别其中的关键信息和主题
3. WHEN 关键信息被识别, THEN THE System SHALL 将其改写为具体的问题形式
4. WHEN 问题改写完成, THEN THE System SHALL 将改写的问题添加到 Questions_Pool
5. WHEN Questions_Pool 更新, THEN THE System SHALL 确保新问题与原始 Sub_Question 相关且不重复

### 需求 7: RAG 检索

**用户故事:** 作为用户，我希望系统能够基于子问题池从向量数据库中检索相关语料，以支持答案生成。

#### 验收标准

1. WHEN RAG_Module 开始检索, THEN THE System SHALL 确保 Vector_Store 已加载基础公司信息数据和 Markdown 文档节点
2. WHEN RAG_Module 查询 Vector_Store, THEN THE System SHALL 使用 Questions_Pool 中的所有问题（包括原始子问题和改写问题）作为查询
3. WHEN RAG_Module 对每个问题查询, THEN THE System SHALL 检索 top_k 个最相关的节点（包括公司信息和 Markdown_Chunk）
4. WHEN 检索完成, THEN THE System SHALL 对所有 Retrieved_Context 进行去重处理
5. WHEN 去重完成, THEN THE System SHALL 使用 Reranker 对 Retrieved_Context 进行二次重排序
6. WHEN 重排序完成, THEN THE System SHALL 过滤掉低相关性的 Retrieved_Context

### 需求 8: 答案生成

**用户故事:** 作为用户，我希望系统能够基于检索到的语料和子问题池生成高质量的最终回答。

#### 验收标准

1. WHEN RAG_Module 完成检索, THEN THE System SHALL 整合 Questions_Pool 和 Retrieved_Context 构建提示词
2. WHEN 提示词构建完成, THEN THE System SHALL 调用 LLM 生成 Final_Answer
3. WHEN LLM 生成回答, THEN THE System SHALL 确保 Final_Answer 引用 Retrieved_Context 中的具体来源
4. WHEN Final_Answer 生成完成, THEN THE System SHALL 返回给用户并记录日志
5. WHEN LLM 生成失败, THEN THE System SHALL 返回错误信息并提供部分检索结果

### 需求 9: 配置管理

**用户故事:** 作为系统管理员，我希望系统的关键参数可配置，以便根据不同场景调整性能。

#### 验收标准

1. WHEN System 初始化, THEN THE System SHALL 从 Config 类加载所有配置参数
2. WHEN 配置参数包括并发数, THEN THE System SHALL 根据配置创建相应数量的 ExecutorAgent 实例
3. WHEN 配置参数包括批处理大小, THEN THE System SHALL 在 Reranker 和 Embedding 时使用配置的批处理大小
4. WHEN 配置参数包括阈值, THEN THE System SHALL 在过滤和重排序时使用配置的阈值
5. WHEN 配置参数无效, THEN THE System SHALL 使用默认值并记录警告

### 需求 10: 短期记忆管理

**用户故事:** 作为系统架构师，我希望系统能够管理 Agent 的短期记忆，以支持多轮对话和状态持久化。

#### 验收标准

1. WHEN PlannerAgent 或 ExecutorAgent 初始化, THEN THE System SHALL 使用 AsyncPostgresSaver 作为短期记忆存储
2. WHEN Agent 处理任务, THEN THE System SHALL 将状态和消息保存到短期记忆
3. WHEN Agent 需要恢复状态, THEN THE System SHALL 从短期记忆加载历史状态
4. WHEN Agent 完成任务, THEN THE System SHALL 清理过期的短期记忆数据
5. WHEN 短期记忆连接失败, THEN THE System SHALL 记录错误并尝试重新连接

### 需求 11: 错误处理和日志

**用户故事:** 作为系统管理员，我希望系统能够妥善处理错误并记录详细日志，以便问题排查和系统监控。

#### 验收标准

1. WHEN System 遇到任何异常, THEN THE System SHALL 记录详细的错误堆栈到日志文件
2. WHEN 工具调用失败, THEN THE System SHALL 记录失败原因但不中断整体流程
3. WHEN 关键组件失败, THEN THE System SHALL 返回友好的错误信息给用户
4. WHEN 日志文件达到配置的大小限制, THEN THE System SHALL 自动轮转日志文件
5. WHEN System 运行, THEN THE System SHALL 记录关键操作的时间戳和执行结果

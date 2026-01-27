# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于多智能体协作的深度文献检索和问答系统，能够从多个数据源并行收集信息，并通过 RAG 技术生成高质量回答。

**技术栈**: Python 3.10+, FastAPI, LangChain, LangGraph, PostgreSQL, ChromaDB, vLLM, TEI

## 核心架构

### 多智能体系统

系统采用分层多智能体设计，核心组件包括：

1. **PlannerAgent** (`agents/agents.py`) - 将复杂查询拆解为多个子问题
2. **ExecutorAgent** (`agents/agents.py`) - 执行单个子问题的完整信息检索流程，基于 ReAct 模式
3. **ExecutorAgent Pool** (`agents/executor_pool.py`) - 并发执行多个 ExecutorAgent，默认池大小为 3
4. **MultiAgent** (`agents/multi_agent.py`) - 整体流程编排协调器

### 数据源分类

**必需数据源**（每次都会调用）：
- Wikipedia（基础知识）
- OpenAlex（学术论文元数据）
- Semantic Scholar（AI 增强论文分析）
- Tavily（实时网络搜索）

**可选数据源**（由 LLM 智能决策）：
- SEC EDGAR（美国上市公司财报）
- AkShare（金融数据）

**基础数据源**（通过 RAG 整合）：
- Crunchbase（创业公司信息）

### RAG 架构

- 文档处理：PDF → Markdown（MinerU） → 切割 → 向量化 → ChromaDB
- 检索：向量检索 + 关键词匹配 → BGE Reranker 重排序 → 上下文构建

## 开发命令

### 环境设置
```bash
# 创建虚拟环境
conda create -n agent_backend python=3.10 -y
conda activate agent_backend

# 安装依赖（使用清华镜像源）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-test.txt
```

### 启动服务
```bash
# 开发模式
python main.py

# 或使用 uvicorn（支持热重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 测试
```bash
# 运行集成测试
bash run_tests.sh

# 或直接运行
pytest tests/test_integration.py -v -s

# 运行单个测试
pytest tests/test_integration.py::test_specific_function -v -s
```

### 构建向量库
```bash
# 初始化基础向量数据库
python core/build_base_vector_store.py
```

## 关键配置

### LLM 配置 (`core/config/`)
支持多种模型提供商：OpenAI、Qwen、OneAPI、Ollama

### 环境变量 (`.env`)
- `DB_URI`: PostgreSQL 连接字符串
- `OPENAI_API_KEY`: OpenAI API 密钥
- `VLLM_BASE_URL`: vLLM Embedding 服务地址
- `RERANK_BASE_URL`: TEI Reranker 服务地址

### ExecutorAgent 配置
- 池大小：默认 3 个
- Reranker 阈值：0.5
- 保留文档数：Top 20

## 重要注意事项

### 长上下文处理
- ExecutorAgent 收集到的 paper 应做裁剪或摘要供 LLM 判断
- 完整 paper 信息需保存在 state 中用于 clean 和 download

### 错误处理
- 使用 `asyncio.gather(*tasks, return_exceptions=True)` 确保单个工具失败不中断整体流程
- PlannerAgent 失败时返回原始查询作为降级策略

### 工具实现位置
- 核心搜索工具（MCP 服务器）：`tools/core_tools/`
- 普通工具：`tools/normal_tool/`

### API 文档
- API 接口文档：`docs/API.md`
- 部署指南：`docs/DEPLOYMENT.md`

### 语言
- 尽量使用中文回复

## 禁止编辑的文件
- config.py
### 规则
- 不确定的时候一定要勤查资料，不要动不动的猜测，不然会造成很多的错误。
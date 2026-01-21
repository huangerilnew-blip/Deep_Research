# 集成测试文档

## 概述

本目录包含 Multi-Agent 深度搜索系统的端到端集成测试。测试覆盖以下方面：

1. **完整查询流程测试** - 验证从查询拆解到答案生成的完整流程
2. **并发执行测试** - 验证 ExecutorAgent Pool 的并发执行能力
3. **错误恢复测试** - 验证系统的容错能力和降级策略
4. **向量库更新测试** - 验证向量库的加载和增量更新
5. **Questions Pool 集成测试** - 验证问题池的完整工作流
6. **API 集成测试** - 验证 HTTP API 接口

## 测试环境要求

### 依赖服务

测试需要以下服务运行：

1. **PostgreSQL** - 短期记忆存储
   - 地址: `localhost:5432`
   - 数据库: `postgres`
   - 用户: `kevin`
   - 密码: `123456`

2. **Chroma** - 向量存储（嵌入在代码中，无需单独启动）

3. **vllm** - 本地 Embedding 模型（可选，测试中使用 Mock）
   - 地址: `http://localhost:8001`

4. **TEI** - BGE Reranker（可选，测试中使用 Mock）
   - 地址: `http://localhost:8080`

### Python 环境

- Python 3.10+
- Conda 虚拟环境: `agent_backend`

## 安装测试依赖

```bash
# 激活虚拟环境
conda activate agent_backend

# 安装测试依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-test.txt
```

## 运行测试

### 方式 1: 使用脚本（推荐）

```bash
# 赋予执行权限
chmod +x run_tests.sh

# 运行测试
./run_tests.sh
```

### 方式 2: 直接使用 pytest

```bash
# 激活虚拟环境
conda activate agent_backend

# 运行所有集成测试
pytest tests/test_integration.py -v

# 运行特定测试类
pytest tests/test_integration.py::TestEndToEndIntegration -v

# 运行特定测试方法
pytest tests/test_integration.py::TestEndToEndIntegration::test_complete_query_flow -v

# 显示详细输出
pytest tests/test_integration.py -v -s

# 生成覆盖率报告
pytest tests/test_integration.py --cov=agents --cov=core --cov=api --cov-report=html
```

## 测试说明

### TestEndToEndIntegration

#### test_complete_query_flow
测试完整的查询处理流程：
- 查询拆解（PlannerAgent）
- 并发执行（ExecutorAgent Pool）
- 文档处理（DocumentProcessor）
- 向量库更新（VectorStoreManager）
- RAG 检索和生成（RAGModule）

**验证点**:
- 返回结果包含所有必需字段
- 子问题数量正确
- 答案非空
- 元数据完整

#### test_concurrent_execution
测试 ExecutorAgent Pool 的并发执行能力：
- 验证池大小正确
- 验证任务正确分配
- 验证所有结果正确收集

**验证点**:
- Agent 数量等于配置的池大小
- 所有子问题都被执行
- 结果数量正确

#### test_error_recovery
测试系统的错误恢复能力：
- 模拟部分 ExecutorAgent 失败
- 模拟文档处理失败
- 验证系统使用降级策略继续执行

**验证点**:
- 部分失败不影响整体流程
- 系统仍然返回有效结果
- 没有抛出异常

#### test_vector_store_update
测试向量库的加载和更新：
- 加载基础向量库
- 添加新文档
- 验证向量库更新

**验证点**:
- 基础向量库正确加载
- 新文档成功添加
- 可以检索到新文档

### TestQuestionsPoolIntegration

#### test_questions_pool_workflow
测试 Questions Pool 的完整工作流：
- 添加原始子问题
- 添加改写问题
- 验证去重功能

**验证点**:
- 原始问题正确存储
- 改写问题正确添加
- 重复问题被去除
- 获取所有问题正确

### TestAPIIntegration

#### test_api_query_endpoint
测试 API 查询端点：
- 请求验证
- MultiAgent 调用
- 响应格式

**验证点**:
- HTTP 状态码正确
- 响应格式符合规范
- 包含所有必需字段

#### test_api_health_endpoint
测试健康检查端点：
- 验证服务状态

**验证点**:
- 返回健康状态
- 包含服务名称

## Mock 策略

为了加快测试速度和避免依赖外部服务，测试中使用了以下 Mock：

1. **PlannerAgent** - Mock 查询拆解结果
2. **ExecutorAgent Pool** - Mock 搜索和下载结果
3. **DocumentProcessor** - Mock 文档处理结果
4. **RAGModule** - Mock 检索和生成结果
5. **LLM 调用** - Mock LLM 响应
6. **搜索工具** - Mock 搜索结果

## 覆盖率报告

运行测试后，覆盖率报告会生成在 `htmlcov/` 目录：

```bash
# 生成覆盖率报告
pytest tests/test_integration.py --cov=agents --cov=core --cov=api --cov-report=html

# 查看报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 故障排查

### PostgreSQL 连接失败

```
psycopg.OperationalError: connection failed
```

**解决方案**:
1. 确认 PostgreSQL 服务正在运行
2. 检查 `core/config.py` 中的 `DB_URI` 配置
3. 确认数据库用户和密码正确

### 导入错误

```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**:
1. 确认已激活正确的 conda 环境
2. 安装缺失的依赖: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xxx`

### 测试超时

```
asyncio.TimeoutError
```

**解决方案**:
1. 检查网络连接
2. 增加测试超时时间
3. 确认依赖服务正常运行

## 持续集成

测试可以集成到 CI/CD 流程中：

```yaml
# .github/workflows/test.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: 123456
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        run: pytest tests/test_integration.py -v
```

## 贡献指南

添加新的集成测试时，请遵循以下规范：

1. **测试命名** - 使用描述性的测试名称，以 `test_` 开头
2. **测试文档** - 在测试函数中添加文档字符串，说明测试目的和验证点
3. **Mock 使用** - 合理使用 Mock，避免依赖外部服务
4. **断言清晰** - 使用清晰的断言消息
5. **清理资源** - 确保测试后清理临时资源

## 参考资料

- [Pytest 文档](https://docs.pytest.org/)
- [Pytest-asyncio 文档](https://pytest-asyncio.readthedocs.io/)
- [FastAPI 测试文档](https://fastapi.tiangolo.com/tutorial/testing/)
- [Hypothesis 文档](https://hypothesis.readthedocs.io/)

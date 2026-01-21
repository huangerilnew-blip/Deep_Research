# API 接口文档

## 概述

Multi-Agent 深度搜索系统提供 RESTful API 接口，支持用户提交复杂查询并获取基于多源检索和 RAG 技术生成的高质量回答。

## 基础信息

- **Base URL**: `http://localhost:8000`
- **Content-Type**: `application/json`
- **字符编码**: UTF-8

## 接口列表

### 1. 健康检查

检查服务是否正常运行。

**端点**: `GET /health`

**请求示例**:
```bash
curl -X GET http://localhost:8000/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-21T10:30:00Z"
}
```

**状态码**:
- `200 OK`: 服务正常运行

---

### 2. 提交查询

提交用户查询，系统将执行完整的多智能体深度搜索流程并返回结果。

**端点**: `POST /query`

**请求参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| query | string | 是 | 用户查询内容，最少 10 个字符 |
| thread_id | string | 否 | 会话线程 ID，用于多轮对话。如不提供，系统自动生成 |
| config | object | 否 | 可选配置参数 |
| config.executor_pool_size | integer | 否 | ExecutorAgent 池大小，默认 3 |
| config.top_k | integer | 否 | 每个问题检索的文档数，默认 5 |
| config.rerank_top_n | integer | 否 | Rerank 后保留的文档数，默认 3 |

**请求示例**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "深度求索公司的主要产品和技术优势是什么？",
    "thread_id": "user123_session456",
    "config": {
      "executor_pool_size": 3,
      "top_k": 5
    }
  }'
```

**响应参数**:

| 参数名 | 类型 | 说明 |
|--------|------|------|
| success | boolean | 请求是否成功 |
| thread_id | string | 会话线程 ID |
| query | string | 原始用户查询 |
| answer | string | 生成的最终回答 |
| metadata | object | 元数据信息 |
| metadata.sub_questions | array | 拆解的子问题列表 |
| metadata.rewritten_questions | array | 从文档改写的问题列表 |
| metadata.documents_processed | integer | 处理的文档数量 |
| metadata.chunks_created | integer | 创建的文档片段数量 |
| metadata.sources | array | 检索到的来源列表 |
| metadata.execution_time | float | 执行时间（秒） |
| error | string | 错误信息（仅在失败时返回） |
| error_code | string | 错误码（仅在失败时返回） |

**成功响应示例**:
```json
{
  "success": true,
  "thread_id": "user123_session456",
  "query": "深度求索公司的主要产品和技术优势是什么？",
  "answer": "根据检索到的信息，深度求索（DeepSeek）是一家专注于人工智能技术的公司，主要产品和技术优势包括：\n\n1. **大语言模型**：DeepSeek 开发了自主研发的大语言模型，具有强大的自然语言理解和生成能力。\n\n2. **技术创新**：公司在模型架构、训练方法和推理优化方面具有独特的技术优势...\n\n来源：\n- Wikipedia: 深度求索\n- OpenAlex: DeepSeek AI Research Papers\n- 公司信息数据库",
  "metadata": {
    "sub_questions": [
      "深度求索公司的主要产品有哪些？",
      "深度求索的核心技术是什么？",
      "深度求索相比其他 AI 公司有什么优势？"
    ],
    "rewritten_questions": [
      "DeepSeek 的大语言模型有哪些特点？",
      "深度求索在 AI 领域的创新点是什么？"
    ],
    "documents_processed": 8,
    "chunks_created": 45,
    "sources": [
      {
        "title": "深度求索 - Wikipedia",
        "source": "wikipedia",
        "url": "https://zh.wikipedia.org/wiki/深度求索"
      },
      {
        "title": "DeepSeek: Advanced Language Models",
        "source": "openalex",
        "url": "https://openalex.org/works/W1234567890"
      }
    ],
    "execution_time": 45.3
  }
}
```

**错误响应示例**:
```json
{
  "success": false,
  "thread_id": "user123_session456",
  "query": "深度求索公司的主要产品和技术优势是什么？",
  "error": "抱歉，无法将您的查询拆解为子问题。请尝试重新表述您的问题。",
  "error_code": "E001"
}
```

**状态码**:
- `200 OK`: 请求成功处理
- `400 Bad Request`: 请求参数错误
- `500 Internal Server Error`: 服务器内部错误
- `503 Service Unavailable`: 依赖服务不可用

---

## 错误码说明

系统使用标准化的错误码来标识不同类型的错误：

| 错误码 | 说明 | 可能原因 | 建议操作 |
|--------|------|----------|----------|
| E001 | 查询拆解失败 | LLM 无法生成有效的子问题列表 | 重新表述查询，使其更加具体和明确 |
| E002 | 所有 Executor 失败 | 所有搜索和下载任务都失败 | 检查网络连接和数据源可用性 |
| E003 | 文档处理失败 | PDF 转换或切割失败 | 检查文档格式和 MinerU 服务状态 |
| E004 | 向量库操作失败 | Chroma 连接或操作失败 | 检查 Chroma 服务状态 |
| E005 | RAG 检索失败 | 向量检索或 Rerank 失败 | 检查向量库和 Reranker 服务 |
| E006 | 答案生成失败 | LLM 调用失败或超时 | 稍后重试或检查 LLM 服务 |
| E007 | 短期记忆连接失败 | PostgreSQL 连接失败 | 检查数据库连接配置 |
| E008 | 配置错误 | 配置参数无效 | 检查配置文件和环境变量 |

## 使用示例

### Python 客户端示例

```python
import requests
import json

def query_deep_search(query: str, thread_id: str = None):
    """调用深度搜索 API"""
    url = "http://localhost:8000/query"
    
    payload = {
        "query": query,
        "thread_id": thread_id,
        "config": {
            "executor_pool_size": 3,
            "top_k": 5
        }
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print(f"回答: {result['answer']}")
            print(f"\n处理了 {result['metadata']['documents_processed']} 个文档")
            print(f"执行时间: {result['metadata']['execution_time']:.2f} 秒")
        else:
            print(f"错误: {result['error']} (错误码: {result['error_code']})")
    else:
        print(f"HTTP 错误: {response.status_code}")

# 使用示例
query_deep_search("深度求索公司的主要产品和技术优势是什么？")
```

### JavaScript 客户端示例

```javascript
async function queryDeepSearch(query, threadId = null) {
  const url = 'http://localhost:8000/query';
  
  const payload = {
    query: query,
    thread_id: threadId,
    config: {
      executor_pool_size: 3,
      top_k: 5
    }
  };
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('回答:', result.answer);
      console.log(`处理了 ${result.metadata.documents_processed} 个文档`);
      console.log(`执行时间: ${result.metadata.execution_time.toFixed(2)} 秒`);
    } else {
      console.error(`错误: ${result.error} (错误码: ${result.error_code})`);
    }
  } catch (error) {
    console.error('请求失败:', error);
  }
}

// 使用示例
queryDeepSearch('深度求索公司的主要产品和技术优势是什么？');
```

### cURL 示例

```bash
# 基本查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "深度求索公司的主要产品和技术优势是什么？"}'

# 带配置参数的查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能在医疗领域的应用有哪些？",
    "thread_id": "medical_ai_session",
    "config": {
      "executor_pool_size": 5,
      "top_k": 10,
      "rerank_top_n": 5
    }
  }'
```

## 性能指标

### 预期响应时间

- **简单查询**（1-2 个子问题）: 20-40 秒
- **中等复杂度查询**（3-5 个子问题）: 40-60 秒
- **复杂查询**（5+ 个子问题）: 60-90 秒

### 并发限制

- **推荐并发数**: 5-10 个请求
- **最大并发数**: 20 个请求（取决于服务器配置）

### 速率限制

当前版本暂无速率限制，但建议：
- 单个用户每分钟不超过 10 个请求
- 避免在短时间内提交大量相似查询

## 最佳实践

### 1. 查询优化

- **具体明确**: 提供具体的查询内容，避免过于宽泛
- **结构化**: 如果查询包含多个方面，可以明确列出
- **上下文**: 提供必要的背景信息

**好的查询示例**:
```
"深度求索公司在 2024 年发布了哪些大语言模型？这些模型的参数规模和性能指标是什么？"
```

**不好的查询示例**:
```
"AI 公司"  # 过于宽泛
```

### 2. 线程 ID 管理

- 对于多轮对话，使用相同的 `thread_id`
- 线程 ID 应该是唯一的，建议格式：`{user_id}_{session_id}`
- 定期清理过期的线程数据

### 3. 错误处理

- 实现重试机制（建议最多重试 3 次）
- 对于 E001 错误，引导用户重新表述查询
- 对于 E002-E008 错误，记录日志并通知管理员

### 4. 结果缓存

- 对于相同或相似的查询，可以考虑缓存结果
- 缓存有效期建议：1-24 小时（取决于数据更新频率）

## 版本历史

### v1.0.0 (2026-01-21)
- 初始版本发布
- 支持基本的查询和回答功能
- 集成多源搜索和 RAG 技术

## 联系方式

如有问题或建议，请联系开发团队。

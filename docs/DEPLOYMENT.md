# 部署指南

## 概述

本文档详细说明 Multi-Agent 深度搜索系统的部署流程，包括依赖服务的安装配置、环境变量设置和系统启动步骤。

## 系统要求

### 硬件要求

- **CPU**: 8 核心或以上（推荐 16 核心）
- **内存**: 32GB 或以上（推荐 64GB）
- **存储**: 100GB 可用空间（用于向量库和文档存储）
- **GPU**: 可选，用于加速 Embedding 和 Reranker（推荐 NVIDIA GPU，显存 16GB+）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+ 或其他发行版) / WSL2
- **Python**: 3.10 或以上
- **Conda**: Miniconda 或 Anaconda
- **Docker**: 20.10 或以上（用于部署依赖服务）
- **Docker Compose**: 1.29 或以上

## 架构概览

系统依赖以下服务：

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent 深度搜索系统                    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ PlannerAgent │  │ ExecutorPool │  │  RAG Module  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
         │                  │                  │
         ├──────────────────┼──────────────────┤
         │                  │                  │
    ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │PostgreSQL│      │   Chroma  │     │   vllm    │
    │(短期记忆) │      │ (向量库)   │     │(Embedding)│
    └─────────┘      └───────────┘     └───────────┘
                                              │
                                        ┌─────▼─────┐
                                        │    TEI    │
                                        │ (Reranker)│
                                        └───────────┘
```

## 依赖服务部署

### 1. PostgreSQL 部署

PostgreSQL 用于存储 Agent 的短期记忆（会话状态）。

#### 使用 Docker 部署

```bash
# 创建 PostgreSQL 容器
docker run -d \
  --name postgres-agent \
  -e POSTGRES_USER=agent_user \
  -e POSTGRES_PASSWORD=agent_password \
  -e POSTGRES_DB=agent_memory \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15

# 验证部署
docker ps | grep postgres-agent
```

#### 使用 Docker Compose 部署

创建 `docker/postgresql/docker-compose.yml`（已存在）：

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: postgres-agent
    environment:
      POSTGRES_USER: agent_user
      POSTGRES_PASSWORD: agent_password
      POSTGRES_DB: agent_memory
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

启动服务：

```bash
cd docker/postgresql
docker-compose up -d
```

#### 连接测试

```bash
# 使用 psql 测试连接
docker exec -it postgres-agent psql -U agent_user -d agent_memory

# 或使用 Python 测试
python -c "
import psycopg
conn = psycopg.connect('postgresql://agent_user:agent_password@localhost:5432/agent_memory')
print('PostgreSQL 连接成功！')
conn.close()
"
```

---

### 2. Chroma 向量数据库部署

Chroma 用于存储和检索文档向量。

#### 使用 Docker 部署

```bash
# 创建 Chroma 容器
docker run -d \
  --name chroma-vectordb \
  -p 8001:8000 \
  -v chroma_data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  chromadb/chroma:latest

# 验证部署
curl http://localhost:8001/api/v1/heartbeat
```

#### 本地部署（开发环境）

```bash
# 激活虚拟环境
conda activate agent_backend

# 安装 Chroma
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple chromadb

# Chroma 将以嵌入式模式运行，数据存储在本地目录
# 配置路径在 core/config.py 中的 VECTOR_STORE_PATH
```

---

### 3. vllm Embedding 服务部署

vllm 用于部署本地 Embedding 模型和 MinerU PDF 转换服务。

#### 安装 vllm

```bash
# 激活虚拟环境
conda activate agent_backend

# 安装 vllm（需要 GPU 支持）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm

# 或使用 Docker 部署
docker pull vllm/vllm-openai:latest
```

#### 部署 Embedding 模型

```bash
# 下载 Embedding 模型（例如 BAAI/bge-large-zh-v1.5）
# 模型路径：/path/to/models/bge-large-zh-v1.5

# 启动 vllm Embedding 服务
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/models/bge-large-zh-v1.5 \
  --port 8002 \
  --host 0.0.0.0 \
  --served-model-name bge-large-zh-v1.5

# 或使用 Docker
docker run -d \
  --name vllm-embedding \
  --gpus all \
  -p 8002:8000 \
  -v /path/to/models:/models \
  vllm/vllm-openai:latest \
  --model /models/bge-large-zh-v1.5 \
  --served-model-name bge-large-zh-v1.5
```

#### 部署 MinerU PDF 转换服务

```bash
# 安装 MinerU
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple magic-pdf

# 启动 MinerU 服务（假设在端口 8003）
# 具体启动命令取决于 MinerU 的部署方式
# 参考 MinerU 官方文档：https://github.com/opendatalab/MinerU
```

#### 验证 vllm 服务

```bash
# 测试 Embedding 服务
curl http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-large-zh-v1.5",
    "input": "测试文本"
  }'
```

---

### 4. TEI Reranker 服务部署

TEI (Text Embeddings Inference) 用于部署 BGE Reranker 模型。

#### 使用 Docker 部署

```bash
# 下载 Reranker 模型（例如 BAAI/bge-reranker-large）
# 模型路径：/path/to/models/bge-reranker-large

# 启动 TEI Reranker 服务
docker run -d \
  --name tei-reranker \
  --gpus all \
  -p 8004:80 \
  -v /path/to/models/bge-reranker-large:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id /data \
  --revision main

# 验证部署
curl http://localhost:8004/health
```

#### 本地部署

```bash
# 激活虚拟环境
conda activate agent_backend

# 安装 FlagEmbedding（包含 BGE Reranker）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FlagEmbedding

# Reranker 将在应用中直接加载，无需单独服务
# 配置在 core/reranker.py 中
```

---

## 应用部署

### 1. 环境准备

#### 创建 Conda 虚拟环境

```bash
# 创建虚拟环境
conda create -n agent_backend python=3.10 -y

# 激活虚拟环境
conda activate agent_backend
```

#### 克隆代码仓库

```bash
# 克隆仓库（假设已有代码）
cd /path/to/workspace

# 或从 Git 克隆
# git clone <repository_url>
# cd <repository_name>
```

### 2. 安装依赖

```bash
# 激活虚拟环境
conda activate agent_backend

# 安装依赖（使用清华镜像源）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 如果需要安装测试依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-test.txt
```

### 3. 环境变量配置

创建 `.env` 文件（或编辑 `core/.env`）：

```bash
# ============================================
# 数据库配置
# ============================================
# PostgreSQL 连接字符串
DATABASE_URL=postgresql://agent_user:agent_password@localhost:5432/agent_memory

# ============================================
# 向量数据库配置
# ============================================
# Chroma 持久化路径
VECTOR_STORE_PATH=./vector_storage
# Chroma 集合名称
VECTTOR_BASE_COLLECTION_NAME=company_info_base

# ============================================
# Embedding 模型配置
# ============================================
# vllm Embedding 服务地址
VLLM_BASE_URL=http://localhost:8002/v1
# Embedding 模型名称
EMBEDDING_MODEL_NAME=bge-large-zh-v1.5
# Embedding 维度
EMBEDDING_DIM=1024

# ============================================
# MinerU 配置
# ============================================
# MinerU PDF 转换服务地址
MINERU_BASE_URL=http://localhost:8003

# ============================================
# Reranker 配置
# ============================================
# Reranker 模型路径（本地部署）
RERANKER_MODEL_PATH=/path/to/models/bge-reranker-large
# 或 TEI 服务地址（Docker 部署）
# RERANKER_API_URL=http://localhost:8004

# ============================================
# LLM 配置
# ============================================
# PlannerAgent 使用的模型
LLM_PLANNER=gpt-4
# ExecutorAgent 使用的模型
LLM_EXECUTOR=gpt-4
# RAG 生成使用的模型
LLM_GENERATOR=gpt-4

# OpenAI API 配置（如果使用 OpenAI 模型）
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# ============================================
# 系统配置
# ============================================
# ExecutorAgent 池大小
EXECUTOR_POOL_SIZE=3
# Markdown 切割最大长度
MAX_CHUNK_SIZE=1000
# 文档保存路径
DOC_SAVE_PATH=./data/downloads
# 基础数据路径
BASEDATA_RESTRUCTURE_PATH=./data/crunchbase_data/restructure_data/restructure_company_info.json

# ============================================
# 检索配置
# ============================================
# 每个问题检索的文档数
TOP_K=5
# Rerank 后保留的文档数
RERANK_TOP_N=3
# Rerank 分数阈值
RERANK_THRESHOLD=0.5

# ============================================
# 日志配置
# ============================================
# 日志级别
LOG_LEVEL=INFO
# 日志文件路径
LOG_FILE_PATH=./logfile/app.log
# 日志文件最大大小（MB）
LOG_MAX_SIZE=100
# 日志备份数量
LOG_BACKUP_COUNT=5

# ============================================
# API 配置
# ============================================
# API 服务端口
API_PORT=8000
# API 服务主机
API_HOST=0.0.0.0
```

### 4. 初始化向量数据库

```bash
# 激活虚拟环境
conda activate agent_backend

# 运行基础向量库构建脚本
python core/build_base_vector_store.py

# 验证向量库创建成功
ls -lh vector_storage/
```

### 5. 启动应用

#### 开发模式

```bash
# 激活虚拟环境
conda activate agent_backend

# 启动应用
python main.py

# 或使用 uvicorn 启动（支持热重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 生产模式

```bash
# 使用 gunicorn + uvicorn workers
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --access-logfile ./logfile/access.log \
  --error-logfile ./logfile/error.log
```

#### 使用 systemd 管理服务

创建 `/etc/systemd/system/agent-backend.service`：

```ini
[Unit]
Description=Multi-Agent Deep Search Backend
After=network.target postgresql.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/workspace
Environment="PATH=/home/your_username/miniconda3/envs/agent_backend/bin"
ExecStart=/home/your_username/miniconda3/envs/agent_backend/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
# 重载 systemd 配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start agent-backend

# 设置开机自启
sudo systemctl enable agent-backend

# 查看服务状态
sudo systemctl status agent-backend

# 查看日志
sudo journalctl -u agent-backend -f
```

### 6. 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 测试查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "深度求索公司的主要产品和技术优势是什么？"
  }'
```

---

## 配置优化

### 性能优化

#### 1. ExecutorAgent Pool 大小

根据服务器 CPU 核心数调整：

```bash
# 推荐值：CPU 核心数 / 2
EXECUTOR_POOL_SIZE=4  # 对于 8 核 CPU
```

#### 2. 批处理大小

根据 GPU 显存调整：

```bash
# Embedding 批处理大小
EMBEDDING_BATCH_SIZE=32  # 16GB 显存
EMBEDDING_BATCH_SIZE=64  # 24GB 显存

# Reranker 批处理大小
RERANK_BATCH_SIZE=16
```

#### 3. 向量检索参数

根据精度和速度需求调整：

```bash
# 高精度配置
TOP_K=10
RERANK_TOP_N=5

# 高速度配置
TOP_K=3
RERANK_TOP_N=2
```

### 内存优化

#### 1. Chroma 配置

```python
# 在 core/vector_store_manager.py 中配置
chroma_settings = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    # 限制内存使用
    chroma_db_impl="duckdb+parquet",
    persist_directory=Config.VECTOR_STORE_PATH
)
```

#### 2. PostgreSQL 连接池

```python
# 在 main.py 中配置
pool = AsyncConnectionPool(
    conninfo=Config.DATABASE_URL,
    min_size=2,
    max_size=10,  # 根据并发需求调整
    timeout=30
)
```

---

## 监控和日志

### 日志管理

日志文件位置：
- 应用日志：`./logfile/app.log`
- 访问日志：`./logfile/access.log`（生产模式）
- 错误日志：`./logfile/error.log`（生产模式）

查看日志：

```bash
# 实时查看应用日志
tail -f logfile/app.log

# 搜索错误日志
grep "ERROR" logfile/app.log

# 查看最近 100 行
tail -n 100 logfile/app.log
```

### 性能监控

#### 使用 Prometheus + Grafana

1. 安装 Prometheus 客户端：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple prometheus-client
```

2. 在应用中添加指标收集（可选）

3. 配置 Grafana 仪表板监控关键指标：
   - 请求响应时间
   - 并发执行数量
   - 向量检索延迟
   - 错误率

---

## 故障排查

### 常见问题

#### 1. PostgreSQL 连接失败

**症状**: `psycopg.OperationalError: connection failed`

**解决方案**:
```bash
# 检查 PostgreSQL 服务状态
docker ps | grep postgres-agent

# 检查连接字符串
echo $DATABASE_URL

# 测试连接
docker exec -it postgres-agent psql -U agent_user -d agent_memory
```

#### 2. Chroma 向量库错误

**症状**: `chromadb.errors.InvalidCollectionException`

**解决方案**:
```bash
# 删除并重建向量库
rm -rf vector_storage/
python core/build_base_vector_store.py
```

#### 3. vllm Embedding 服务超时

**症状**: `requests.exceptions.Timeout`

**解决方案**:
```bash
# 检查 vllm 服务状态
curl http://localhost:8002/health

# 增加超时时间（在 core/vector_store_manager.py 中）
# timeout=60
```

#### 4. MinerU PDF 转换失败

**症状**: `PDF to Markdown conversion failed`

**解决方案**:
```bash
# 检查 MinerU 服务
curl http://localhost:8003/health

# 检查 PDF 文件格式
file /path/to/document.pdf

# 手动测试转换
magic-pdf -p /path/to/document.pdf -o /tmp/output
```

#### 5. 内存不足

**症状**: `MemoryError` 或系统变慢

**解决方案**:
```bash
# 减少 ExecutorAgent Pool 大小
EXECUTOR_POOL_SIZE=2

# 减少批处理大小
EMBEDDING_BATCH_SIZE=16
RERANK_BATCH_SIZE=8

# 限制检索数量
TOP_K=3
RERANK_TOP_N=2
```

---

## 备份和恢复

### 数据备份

#### 1. PostgreSQL 备份

```bash
# 备份数据库
docker exec postgres-agent pg_dump -U agent_user agent_memory > backup_$(date +%Y%m%d).sql

# 恢复数据库
docker exec -i postgres-agent psql -U agent_user agent_memory < backup_20260121.sql
```

#### 2. Chroma 向量库备份

```bash
# 备份向量库目录
tar -czf vector_storage_backup_$(date +%Y%m%d).tar.gz vector_storage/

# 恢复向量库
tar -xzf vector_storage_backup_20260121.tar.gz
```

#### 3. 文档备份

```bash
# 备份下载的文档
tar -czf documents_backup_$(date +%Y%m%d).tar.gz data/downloads/

# 恢复文档
tar -xzf documents_backup_20260121.tar.gz
```

---

## 安全建议

### 1. 网络安全

- 使用防火墙限制端口访问
- 仅开放必要的端口（8000）
- 使用 HTTPS（配置 Nginx 反向代理）

### 2. 数据库安全

- 使用强密码
- 限制数据库访问 IP
- 定期更新 PostgreSQL 版本

### 3. API 安全

- 实现 API 密钥认证
- 添加速率限制
- 记录所有 API 访问日志

### 4. 日志安全

- 不在日志中记录敏感信息
- 定期清理旧日志
- 限制日志文件访问权限

---

## 升级指南

### 应用升级

```bash
# 1. 备份数据
./scripts/backup.sh

# 2. 停止服务
sudo systemctl stop agent-backend

# 3. 更新代码
git pull origin main

# 4. 更新依赖
conda activate agent_backend
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 5. 运行迁移脚本（如有）
python scripts/migrate.py

# 6. 启动服务
sudo systemctl start agent-backend

# 7. 验证升级
curl http://localhost:8000/health
```

---

## 附录

### A. 完整的 Docker Compose 配置

创建 `docker-compose.yml`：

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: postgres-agent
    environment:
      POSTGRES_USER: agent_user
      POSTGRES_PASSWORD: agent_password
      POSTGRES_DB: agent_memory
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  chroma:
    image: chromadb/chroma:latest
    container_name: chroma-vectordb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      IS_PERSISTENT: "TRUE"
    restart: unless-stopped

  vllm-embedding:
    image: vllm/vllm-openai:latest
    container_name: vllm-embedding
    ports:
      - "8002:8000"
    volumes:
      - /path/to/models:/models
    command: >
      --model /models/bge-large-zh-v1.5
      --served-model-name bge-large-zh-v1.5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  tei-reranker:
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    container_name: tei-reranker
    ports:
      - "8004:80"
    volumes:
      - /path/to/models/bge-reranker-large:/data
    command: >
      --model-id /data
      --revision main
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  postgres_data:
  chroma_data:
```

启动所有服务：

```bash
docker-compose up -d
```

### B. 环境变量模板

参考 `core/.env.example`（需要创建）。

### C. 性能基准测试

运行基准测试：

```bash
conda activate agent_backend
python scripts/benchmark.py
```

---

## 联系方式

如有部署问题，请联系开发团队或查阅项目文档。

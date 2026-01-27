import os

class Config:
    """统一的配置类，集中管理所有常量"""
    # 日志持久化存储
    MAX_BYTES = 5*1024*1024
    BACKUP_COUNT = 3
    LOG_FILE = "logfile/app.log"
    # PostgreSQL数据库配置参数
    DB_URI = os.getenv("DB_URI", "postgresql://kevin:123456@localhost:5432/postgres?sslmode=disable")
    MIN_SIZE = 5
    MAX_SIZE = 10

    # Redis数据库配置参数
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    SESSION_TIMEOUT = 3600

    # openai:调用gpt模型,qwen:调用阿里通义千问大模型,oneapi:调用oneapi方案支持的模型,ollama:调用本地开源大模型
    LLM_TYPE = "qwen"
    LLM_PLANNER="qwen"
    LLM_EXECUTOR="qwen"
    PLANNER_EPOCH=3
    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8001
    EMAIL="huang.eril.new@gmail.com" ##pubmed 中最好提供邮箱，防止封id

    # 向量存储和文档路径配置
    DOC_SAVE_PATH="/home/qinshan/deepresearch/data/downloads" # 网站下载的文档存储路径
    VECTOR_STORE_PATH="vector_storage" # 向量存储路径
    VECTTOR_BASE_COLLECTION_NAME="base_crunchbase_collection" #基础向量集合名称
    VECTTOR_BASEDATA_PATH="/home/qinshan/deepresearch/data/crunchbase_data" #基础数据路径
    VECTOR_DIM=1024 #向量维度
    BASEDATA_RESTRUCTURE_PATH="data/crunchbase_data/restructure_data/restructure_company_info.json" #清洗与重构后的基础数据路径
    TOP_K=5 #向量检索top_k
    TAVILY_NUM=3 #Tavily文献检索返回数量
    EXA_NUM=10 #EXA检索的结果
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    WIKI_NUM=3 #wiki检索返回数量
    WIKI_LANGUAGE="en" #wiki检索语言版本
    SEC_EDGAR_USER_AGENT="Trina Solar,huang.eril.new@gmail.com"
    
    # Rerank 配置
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # BGE Reranker 模型名称
    RERANK_BASE_URL ="http://localhost:8100"  # TEI rerank API 地址
    RERANK_API_KEY = "EMPTY"  # TEI API Key（如果不需要可以设为 EMPTY）
    RERANK_THRESHOLD = 0.5  # Rerank 分数阈值
    RERANK_TOP_N = 20  # Rerank 后保留的文档数量
    RERANK_BATCH_SIZE = 32  # Rerank 批处理大小
    RERANK_MAX_CONCURRENT = 5  # Rerank 最大并发请求数
    
    # 需要进行清洗和 Rerank 的数据源
    PAPER_CLEAN = ["openalex", "semantic_scholar"]
    
    # MultiAgent 配置
    EXECUTOR_POOL_SIZE = 3  # ExecutorAgent 池大小，默认 3
    MAX_CHUNK_SIZE = 1000  # Markdown 切割最大长度，默认 1000 字符
    
    # 文档处理配置
    MINERU_BASE_URL = "http://localhost:8080"  # MinerU 服务地址
    VLLM_BASE_URL =  "http://localhost:8081"  # vllm Embedding 服务地址
    EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # Embedding 模型名称


def get_rotating_file_handler(config=None):
    """
    创建 RotatingFileHandler，自动确保日志目录存在

    Args:
        config: Config 实例，默认为 Config()

    Returns:
        ConcurrentRotatingFileHandler 实例

    Usage:
        from concurrent_log_handler import ConcurrentRotatingFileHandler
        from core.config.config import get_rotating_file_handler

        handler = get_rotating_file_handler()
    """
    from concurrent_log_handler import ConcurrentRotatingFileHandler

    if config is None:
        config = Config()

    # 通过访问 LOG_FILE 属性，触发目录创建
    return ConcurrentRotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.MAX_BYTES,
        backupCount=config.BACKUP_COUNT
    )
import os


class Config:
    """统一的配置类，集中管理所有常量"""
    # 日志持久化存储
    LOG_FILE = "logfile/app.log"
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    MAX_BYTES = 5*1024*1024,
    BACKUP_COUNT = 3

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
    DOC_SAVE_PATH="../../doc/downloads" # 网站下载的文档存储路径
    VECTOR_STORE_PATH="../vector_storage" # 向量存储路径
    VECTTOR_BASE_COLLECTION_NAME="base_collection" #基础向量集合名称
    VECTTOR_BASEDATA_PATH="../doc/crunchbase_data" #基础数据路径
    VECTOR_DIM=1024 #向量维度
    BASEDATA_RESTRUCTURE_PATH="../doc/crunchbase_data/restructure_data/restructure_company_info.json" #清洗与重构后的基础数据路径
    TOP_K=5 #向量检索top_k
    SEARCH_SIZE=10 #文献检索返回数量
    TAVILY_NUM=3 #Tavily文献检索返回数量
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    WIKI_NUM=3 #wiki检索返回数量
    WIKI_LANGUAGE="en" #wiki检索语言版本
    SEC_EDGAR_USER_AGENT="Trina Solar,m6qdum90f@zzzz.lingeringp.com"
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext,Settings,StorageContext
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.readers.json import JSONReader
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from chromadb import PersistentClient
from config import Config
import os,json,logging,pandas as pd
from pathlib import Path
from concurrent_log_handler import ConcurrentRotatingFileHandler
from extract_company_info import extract_company_info
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)
load_dotenv()
class MyJSONReader:
    """自己定义的json加载器，将重构的json数据按行加载为多个document对象"""
    def __init__(self,path:str):
        self.path = path
    def load(self):
        documents = []
        with open(self.path, 'r', encoding='utf-8') as f:
            json_file=json.load(f)
            for line in json_file:
                documents.append(Document(text=json.dumps(line)))
        return documents
def restructure_crunchbase_data(data_path:str=Config.VECTTOR_BASEDATA_PATH,save_path:str=Config.BASEDATA_RESTRUCTURE_PATH):
    """对crunchbase数据进行清洗，去除一些无用的列"""
    columns_map = {
        # 时间信息
        'Announced Date': '融资公告日期',
        # 企业基本信息
        'Organization Name': '企业名称',
        'Organization Website': '企业官网',
        'Organization Location': '企业所在地',
        # 业务概况
        'Organization Description': '企业描述',
        'Organization Industries': '所属行业',
        # 财务信息
        'Money Raised (in USD)': '本轮融资金额(美元)',
        'Total Funding Amount (in USD)': '累计融资总额(美元)',
        'Organization Revenue Range': '收入范围',
        'Funding Status': '融资状态'
    }
    try:
        extract_company_info(data_path,columns_map,save_path) #对crunchbase数据进行清洗重构
    except Exception as e:
        logger.error(f"重构crunchbase数据失败: {e}")
        raise e

def build_base_vector_store(vectot_path:str=Config.VECTOR_STORE_PATH,
                            base_name:str=Config.VECTTOR_BASE_COLLECTION_NAME,
                            data_path:str=Config.BASEDATA_RESTRUCTURE_PATH):

    restructure_crunchbase_data(save_path=data_path)
    #每一行作为一个document对象
    # Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    #                                           api_key=os.getenv("API_KEY"))
    Settings.embed_model = ZhipuAIEmbedding(model="embedding-2", api_key=os.getenv("ZHIPU_API_KEY"))


    reader = MyJSONReader(path=data_path)
    try:
        documents = reader.load()
        print(f"总共加载出 {len(documents)} 条文档数据")
        print(repr(documents[0].text))
        if not os.path.exists(vectot_path):
            Path(vectot_path).mkdir(parents=True, exist_ok=True)
        chroma_client = PersistentClient(path=vectot_path)
        collection=chroma_client.get_or_create_collection(base_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents=documents,storage_context=storage_context)
        index.storage_context.persist(persist_dir=vectot_path)
    except Exception as e:
        logger.error(f"构建基础向量存储失败: {e}")
        raise e

if __name__ == "__main__":
    build_base_vector_store()
    # restructure_crunchbase_data()
    # if  os.path.exists(Config.BASEDATA_RESTRUCTURE_PATH):
    #     print(f"{Config.BASEDATA_RESTRUCTURE_PATH}路径存在")

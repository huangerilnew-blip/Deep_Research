#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 检索前处理模块
负责在基础向量数据库上添加textnodes
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from concurrent_log_handler import ConcurrentRotatingFileHandler

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.embeddings.zhipuai import ZhipuAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseEmbedding, BaseNode
from chromadb import PersistentClient
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from core.config import Config

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []

handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=Config.MAX_BYTES,
    backupCount=Config.BACKUP_COUNT
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


class VectorStoreManager:
    """RAG 检索前处理模块
    
    管理 Chroma 向量数据库，支持基础数据加载和增量更新
    使用本地 vllm 部署的 Embedding 模型
    """
    
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        persist_dir: str = Config.VECTOR_STORE_PATH,
        collection_name: str = Config.VECTTOR_BASE_COLLECTION_NAME
    ):
        """初始化向量存储管理器
        
        Args:
            persist_dir: 向量库持久化路径
            collection_name: 集合名称
            embedding_model: 本地 vllm 部署的 Embedding 模型（可选）
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model or self._get_local_embedding_model()
        # self.embedding_model = embedding_model
        self.chroma_client = None
        self.vector_store = None
        self.index = None
        
        logger.info(f"初始化 VectorStoreManager: persist_dir={persist_dir}, collection={collection_name}")
    
    def _get_local_embedding_model(self):
        """获取本地 vllm 部署的 Embedding 模型
        
        根据配置选择合适的 Embedding 模型：
        1. 如果配置了 VLLM_BASE_URL，尝试使用 TextEmbeddingsInference 连接本地服务
        2. 否则使用 ZhipuAI Embedding（与现有代码保持一致）
        3. 如果都不可用，回退到本地 HuggingFace 模型
        """
        logger.info(f"尝试初始化 Embedding 模型: {Config.EMBEDDING_MODEL_NAME}")
        # 优先使用本地 vllm/TEI 服务
        if Config.VLLM_BASE_URL :
            try:
                
                
                logger.info(f"尝试连接本地 vllm 服务: {Config.VLLM_BASE_URL}")
                embedding_model = OpenAILikeEmbedding(
                    base_url=Config.VLLM_BASE_URL,
                    model_name=Config.EMBEDDING_MODEL_NAME,
                    timeout=30,
                    embed_batch_size=10
                )
                
                Settings.embed_model = embedding_model ## 这里将会构造一个全局的embedding（llamaindex的框架代码）
                logger.info("本地 vllm Embedding 服务连接成功")
                return embedding_model
                
            except Exception as e:
                logger.warning(f"连接本地 vllm 服务失败: {e}")


        # 其次使用 ZhipuAI（与现有代码保持一致）
        # try:
        #     zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        #     if zhipu_api_key:
        #         logger.info("使用 ZhipuAI Embedding 模型")
        #         embedding_model = ZhipuAIEmbedding(
        #             model="embedding-2",
        #             api_key=zhipu_api_key
        #         )
        #         Settings.embed_model = embedding_model
        #         logger.info("ZhipuAI Embedding 模型初始化完成")
        #         return embedding_model
        # except Exception as e:
        #     logger.warning(f"ZhipuAI Embedding 初始化失败: {e}")
        

        
        # # 回退到本地 HuggingFace 模型
        # logger.info(f"使用本地 HuggingFace 模型: {Config.EMBEDDING_MODEL_NAME}")
        # embedding_model = HuggingFaceEmbedding(
        #     model_name=Config.EMBEDDING_MODEL_NAME
        # )
        
        # Settings.embed_model = embedding_model
        # logger.info("HuggingFace Embedding 模型初始化完成")
        
        # return embedding_model
    
    def _load_json_documents(self, json_path: str) -> List[Document]:
        """从 JSON 文件加载文档
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            Document 列表
        """
        documents = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            if isinstance(json_data, list):
                for idx, item in enumerate(json_data):
                    # 将每个 JSON 对象转换为 Document
                    # 保留原始 JSON 作为文本内容，同时添加元数据
                    doc_text = json.dumps(item, ensure_ascii=False, indent=2)
                    
                    # 提取元数据（如果存在）
                    metadata = {
                        "source": "base_data",
                        "doc_index": idx,
                        "json_path": json_path
                    }
                    
                    # 如果 JSON 对象有特定字段，添加到元数据
                    if isinstance(item, dict):
                        if "name" in item:
                            metadata["name"] = item["name"]
                        if "id" in item:
                            metadata["id"] = str(item["id"])
                    
                    documents.append(Document(
                        text=doc_text,
                        metadata=metadata
                    ))
            else:
                logger.warning(f"JSON 文件格式不是列表: {json_path}")
                
            logger.info(f"从 {json_path} 加载了 {len(documents)} 个文档")
            
        except Exception as e:
            logger.error(f"加载 JSON 文件失败: {e}")
            raise e
        
        return documents
    
    def built_base_vector_store(
        self,
        base_data_path: str = Config.BASEDATA_RESTRUCTURE_PATH,
        force_reload: bool = False
    ) -> VectorStoreIndex:
        """加载基础向量库（公司信息）
        
        Args:
            base_data_path: 基础数据 JSON 文件路径
            force_reload: 是否强制重新加载（默认 False，如果已存在则直接加载）
            
        Returns:
            VectorStoreIndex 实例
        """
        logger.info(f"开始加载基础向量库: {base_data_path}")
        
        # 检查基础数据文件是否存在
        if not os.path.exists(base_data_path):
            logger.error(f"基础数据文件不存在: {base_data_path}")
            raise FileNotFoundError(f"基础数据文件不存在: {base_data_path}")
        
        # 创建持久化目录
        if not os.path.exists(self.persist_dir):
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"创建向量存储目录: {self.persist_dir}")
        
        # 初始化 Chroma 客户端
        self.chroma_client = PersistentClient(path=self.persist_dir)
        collection = self.chroma_client.get_or_create_collection(self.collection_name)
        
        # 创建向量存储
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 检查集合是否已有数据
        collection_count = collection.count()
        logger.info(f"当前集合中已有 {collection_count} 个文档")
        
        if collection_count > 0 and not force_reload:
            # 如果已有数据且不强制重新加载，直接从现有存储加载
            logger.info("检测到已有向量数据，直接加载现有索引")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_context
            )
        else:
            # 否则从基础数据构建新索引
            if force_reload and collection_count > 0:
                logger.warning("强制重新加载，将清空现有数据")
                # 清空集合
                self.chroma_client.delete_collection(self.collection_name)
                collection = self.chroma_client.create_collection(self.collection_name)
                self.vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # 加载文档
            documents = self._load_json_documents(base_data_path)
            
            if not documents:
                logger.warning("没有加载到任何文档")
                return None
            
            # 构建索引
            logger.info(f"开始构建向量索引，文档数量: {len(documents)}")
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context
            )
            
            # 持久化
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info(f"基础向量库加载完成，已持久化到: {self.persist_dir}")
        
        return self.index
    
    def add_nodes(
        self,
        nodes: List[BaseNode]
    ) -> VectorStoreIndex:
        """向现有向量库添加新文档
        
        Args:
             nodes: LlamaIndex TextNode 列表
            
        Returns:
            更新后的 VectorStoreIndex
        """
        if not  nodes:
            logger.warning("没有文档需要添加")
            return self.index
        
        logger.info(f"开始添加 {len(nodes)} 个新文档到向量库")
        
        # 如果索引未初始化，先尝试加载或构建向量库
        if self.index is None:
            logger.info("索引未初始化，尝试加载或构建向量库")
            self._load_or_build_index()
        
        # 添加文档到索引
            self.index.insert_nodes(nodes)
        
        # 持久化
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info(f"成功添加 {len( nodes)} 个node并持久化")
        
        return self.index
    
    def _load_or_build_index(self) -> VectorStoreIndex:
        """智能加载或构建向量索引
        
        判断本地是否已存在向量库：
        - 如果存在且有数据，则直接加载
        - 如果不存在或为空，则构建基础向量库
        
        Returns:
            VectorStoreIndex 实例
        """
        logger.info("检查本地向量库是否存在")
        
        # 检查持久化目录和 Chroma 数据库文件是否存在
        vector_db_exists = os.path.exists(self.persist_dir)
        chroma_db_file = os.path.join(self.persist_dir, "chroma.sqlite3")
        has_chroma_db = os.path.exists(chroma_db_file)
        
        logger.info(f"向量库目录存在: {vector_db_exists}, Chroma 数据库存在: {has_chroma_db}")
        
        # 如果目录和数据库文件都存在，尝试加载现有数据
        if vector_db_exists and has_chroma_db:
            try:
                logger.info("检测到本地向量库，尝试加载")
                
                # 初始化 Chroma 客户端
                self.chroma_client = PersistentClient(path=self.persist_dir)
                
                # 尝试获取集合
                try:
                    collection = self.chroma_client.get_collection(self.collection_name)
                    collection_count = collection.count()
                    logger.info(f"找到集合 '{self.collection_name}'，包含 {collection_count} 个文档")
                    
                    # 如果集合有数据，直接加载
                    if collection_count > 0:
                        logger.info("从现有向量库加载索引")
                        self.vector_store = ChromaVectorStore(chroma_collection=collection)
                        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                        self.index = VectorStoreIndex.from_vector_store(
                            vector_store=self.vector_store,
                            storage_context=storage_context
                        )
                        logger.info("成功从本地加载向量索引")
                        return self.index
                    else:
                        logger.info("集合为空，需要构建基础向量库")
                except Exception as e:
                    logger.info(f"集合 '{self.collection_name}' 不存在: {e}，需要构建基础向量库")
                    
            except Exception as e:
                logger.warning(f"加载本地向量库失败: {e}，将尝试构建基础向量库")
        else:
            logger.info("本地向量库不存在，需要构建基础向量库")
        
        # 如果无法加载现有数据，则构建基础向量库
        logger.info("开始构建基础向量库")
        self.built_base_vector_store()
        
        return self.index
    
    def get_retriever(self, top_k: int = Config.TOP_K):
        """获取检索器
        
        Args:
            top_k: 检索返回的文档数量
            
        Returns:
            检索器实例
        """
        if self.index is None:
            logger.error("索引未初始化，无法创建检索器")
            raise ValueError("索引未初始化，请先调用 built_base_vector_store()")
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        logger.info(f"创建检索器，top_k={top_k}")
        
        return retriever
    
    def get_stats(self) -> dict:
        """获取向量库统计信息
        
        Returns:
            包含统计信息的字典
        """
        stats = {
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "embedding_model": Config.EMBEDDING_MODEL_NAME,
            "vllm_base_url": Config.VLLM_BASE_URL,
            "index_initialized": self.index is not None,
            "document_count": 0
        }
        
        if self.chroma_client and self.collection_name:
            try:
                collection = self.chroma_client.get_collection(self.collection_name)
                stats["document_count"] = collection.count()
            except Exception as e:
                logger.warning(f"无法获取集合统计信息: {e}")
        
        return stats


# 测试代码
if __name__ == "__main__":
    # 测试向量存储管理器
    manager = VectorStoreManager()
    
    # 加载基础向量库
    try:
        print("=" * 60)
        print("测试 VectorStoreManager")
        print("=" * 60)
        
        # 显示初始统计信息
        stats = manager.get_stats()
        print(f"\n初始统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 加载基础向量库
        print(f"\n正在加载基础向量库...")
        index = manager.built_base_vector_store()
        print(f"✓ 基础向量库加载成功")
        
        # 显示加载后的统计信息
        stats = manager.get_stats()
        print(f"\n加载后统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试检索器
        retriever = manager.get_retriever(top_k=3)
        print(f"\n✓ 检索器创建成功 (top_k=3)")
        
        # 测试添加文档
        print(f"\n正在添加测试文档...")
        test_docs = [
            Document(
                text="这是一个测试文档1：关于人工智能的发展",
                metadata={"source": "test", "doc_id": "test_1"}
            ),
            Document(
                text="这是一个测试文档2：关于机器学习的应用",
                metadata={"source": "test", "doc_id": "test_2"}
            )
        ]
        manager.add_documents(test_docs)
        print(f"✓ 成功添加 {len(test_docs)} 个文档")
        
        # 显示最终统计信息
        stats = manager.get_stats()
        print(f"\n最终统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiAgent 协调器
负责整体流程编排，协调各个组件
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from typing import Dict, Any, List
from psycopg_pool import AsyncConnectionPool
from concurrent_log_handler import ConcurrentRotatingFileHandler

from agents.agents import PlannerAgent
from agents.executor_pool import ExecutorAgentPool
from core.rag_preprocess_module import VectorStoreManager
from core.document_processor import DocumentProcessor
from core.rag_module import RAGModule
from core.models import QuestionsPool
from core.config import Config
from core.llms import get_llm
from core.reranker import BGEReranker

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


class MultiAgent:
    """MultiAgent 协调器
    
    整体流程编排，协调各个组件
    """
    
    def __init__(
        self,
        pool: AsyncConnectionPool,
        executor_pool_size: int = Config.EXECUTOR_POOL_SIZE,
        planner_model: str = Config.LLM_PLANNER,
        executor_model: str = Config.LLM_EXECUTOR
    ):
        """初始化 MultiAgent
        
        Args:
            pool: PostgreSQL 连接池（用于短期记忆）
            executor_pool_size: ExecutorAgent 池大小
            planner_model: PlannerAgent 使用的模型
            executor_model: ExecutorAgent 使用的模型
        """
        self.pool = pool
        self.executor_pool_size = executor_pool_size
        self.planner_model = planner_model
        self.executor_model = executor_model
        
        # 初始化组件
        self.planner_agent = PlannerAgent(pool, planner_model)
        self.executor_pool = ExecutorAgentPool(pool, executor_pool_size, executor_model)
        self.vector_store_manager = VectorStoreManager()
        
        # LLM 和 Reranker
        self.llm = get_llm(executor_model)[0]
        self.reranker = BGEReranker()
        
        # 文档处理器
        self.document_processor = DocumentProcessor(
            embedding_model=self.vector_store_manager.embedding_model,
            llm=self.llm
        )
        
        # 向量存储索引（延迟初始化）
        self.vector_store_index = None
        
        logger.info(f"初始化 MultiAgent: executor_pool_size={executor_pool_size}")

    async def process_query(self, user_query: str, thread_id: str) -> Dict[str, Any]:
        """处理用户查询的完整流程
        
        Args:
            user_query: 用户输入的查询
            thread_id: 会话线程 ID
            
        Returns:
            包含最终回答和元数据的字典
        """
        logger.info(f"开始处理用户查询: {user_query}")
        
        try:
            # 1. 初始化向量存储
            if self.vector_store_index is None:
                await self._initialize_vector_store()
            
            # 2. 调用 PlannerAgent 拆解查询
            sub_questions = await self._plan_query(user_query, thread_id)
            logger.info(f"查询拆解完成，生成 {len(sub_questions)} 个子问题")
            
            # 3. 初始化 Questions Pool
            questions_pool = QuestionsPool()
            questions_pool.add_original_questions(sub_questions)
            
            # 4. 调用 ExecutorAgent Pool 并发执行
            executor_results = await self.executor_pool.execute_questions(
                sub_questions,
                thread_id
            )
            logger.info(f"ExecutorAgent Pool 执行完成，获得 {len(executor_results)} 个结果")
            
            # 5. 收集所有下载的文档
            all_documents = []
            for result in executor_results:
                downloaded_papers = result.get('downloaded_papers', [])
                all_documents.extend(downloaded_papers)
            
            logger.info(f"共收集到 {len(all_documents)} 个文档")
            
            # 6. 处理文档（PDF 转 Markdown、切割、问题改写）
            if all_documents:
                llama_docs, rewritten_questions = await self.document_processor.get_nodes(
                    all_documents
                )
                logger.info(f"文档处理完成: {len(llama_docs)} 个片段, {len(rewritten_questions)} 个改写问题")
                
                # 7. 更新 Questions Pool
                questions_pool.add_rewritten_questions(rewritten_questions)
                
                # 8. 向量化并入库
                if llama_docs:
                    self.vector_store_manager.add_documents(llama_docs)
                    logger.info(f"成功添加 {len(llama_docs)} 个文档到向量库")
            
            # 9. RAG 检索和生成
            rag_module = RAGModule(
                vector_store=self.vector_store_index,
                reranker=self.reranker,
                llm=self.llm
            )
            
            result = await rag_module.retrieve(
                questions_pool=questions_pool.get_all_questions(),
                original_query=user_query
            )
            
            # 10. 构建返回结果
            final_result = {
                'query': user_query,
                'sub_questions': sub_questions,
                'rewritten_questions_count': len(questions_pool.rewritten_questions),
                'total_questions': len(questions_pool),
                'documents_processed': len(all_documents),
                'answer': result['answer'],
                'metadata': {
                    'retrieved_count': result['retrieved_count'],
                    'unique_count': result['unique_count'],
                    'reranked_count': result['reranked_count']
                }
            }
            
            logger.info("查询处理完成")
            return final_result
            
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'query': user_query,
                'error': str(e),
                'answer': f"抱歉，处理您的查询时出现错误: {str(e)}"
            }

    async def _initialize_vector_store(self):
        """初始化向量存储，加载基础公司信息数据"""
        logger.info("开始初始化向量存储")
        
        try:
            self.vector_store_index = self.vector_store_manager.load_base_vector_store()
            logger.info("向量存储初始化完成")
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            raise
    
    async def _plan_query(self, user_query: str, thread_id: str) -> List[str]:
        """调用 PlannerAgent 拆解查询
        
        Args:
            user_query: 用户查询
            thread_id: 线程 ID
            
        Returns:
            子问题列表
        """
        try:
            # 调用 PlannerAgent
            config = {"configurable": {"thread_id": f"{thread_id}_planner"}}
            initial_state = {
                "planner_messages": [user_query],
                "planner_result": None,
                "epoch": 0
            }
            
            result = await self.planner_agent.graph.ainvoke(initial_state, config)
            
            # 解析结果
            planner_result = result.get('planner_result')
            if planner_result:
                content = planner_result.content
                try:
                    data = json.loads(content)
                    tasks = data.get('tasks', [])
                    
                    if isinstance(tasks, list) and len(tasks) >= 3:
                        return tasks
                    else:
                        logger.warning(f"子问题数量不足: {len(tasks)}")
                        return tasks if tasks else [user_query]
                except json.JSONDecodeError:
                    logger.error(f"解析 JSON 失败: {content}")
                    return [user_query]
            else:
                logger.error("PlannerAgent 未返回结果")
                return [user_query]
                
        except Exception as e:
            logger.error(f"查询拆解失败: {e}")
            # 降级：返回原始查询
            return [user_query]
    
    async def _cleanup(self):
        """清理资源"""
        logger.info("开始清理 MultiAgent 资源")
        
        try:
            # 清理 PlannerAgent
            await self.planner_agent._clean()
            
            # 清理 ExecutorAgent Pool
            await self.executor_pool.cleanup()
            
            logger.info("MultiAgent 资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("MultiAgent 测试")
    print("=" * 60)
    print("✓ MultiAgent 模块导入成功")
    print("\n注意：完整测试需要 PostgreSQL 连接池")

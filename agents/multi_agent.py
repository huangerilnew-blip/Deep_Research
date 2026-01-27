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
from core.rag.rag_preprocess_module import VectorStoreManager
from core.rag.document_processor import DocumentProcessor
from core.rag.rag_postprocess_module import RAGPostProcessModule as RAGModule
from core.rag.models import QuestionsPool, BGERerankNodePostprocessor
from core.config.config import Config
from core.llms import get_llm
from core.rag.reranker import BGEReranker

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

        # 创建 BGE Reranker 节点后处理器
        self.node_postprocessor = BGERerankNodePostprocessor(
            reranker=self.reranker,
            top_n=Config.RERANK_TOP_N,
            score_threshold=Config.RERANK_THRESHOLD
        )
        
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
                llama_docs = await self.document_processor.get_nodes(all_documents)
                logger.info(f"文档处理完成: {len(llama_docs)} 个片段")

                # 向量化并入库
                if llama_docs:
                    self.vector_store_manager.add_nodes(llama_docs)
                    logger.info(f"成功添加 {len(llama_docs)} 个文档到向量库")

            # 7. RAG 检索（只用 planner 子问题进行检索）
            retriever = self.vector_store_index.as_retriever(similarity_top_k=Config.TOP_K)
            rag_module = RAGModule(
                vector_store=self.vector_store_index,
                retriever=retriever,
                node_postprocessor=self.node_postprocessor,
                top_k=Config.TOP_K
            )

            # 执行检索，获取去重后的节点
            retrieved_nodes = await rag_module.retrieve_postprecess(
                planner_questions=sub_questions
            )

            # 8. 从检索结果的 nodes 中提取 questions，构建 question_pool
            retrieved_questions = self._extract_questions_from_nodes(retrieved_nodes)
            logger.info(f"从检索结果中提取到 {len(retrieved_questions)} 个问题")

            # question_pool = planner 子问题 + 检索结果中的 questions
            questions_pool = QuestionsPool()
            questions_pool.add_original_questions(sub_questions)
            questions_pool.add_rewritten_questions(retrieved_questions)
            logger.info(f"Question Pool 构建完成，共 {len(questions_pool)} 个问题")

            # 10. 基于检索结果生成最终答案
            answer = await self._generate_answer(
                user_query=user_query,
                sub_questions=sub_questions,
                question_pool=question_pool,
                retrieved_nodes=retrieved_nodes
            )

            # 11. 构建返回结果
            final_result = {
                'query': user_query,
                'sub_questions': sub_questions,
                'rewritten_questions_count': len(questions_pool.rewritten_questions),
                'total_questions': len(question_pool),
                'documents_processed': len(all_documents),
                'answer': answer,
                'metadata': {
                    'retrieved_count': len(retrieved_nodes),
                    'unique_count': len(retrieved_nodes),
                    'reranked_count': len(retrieved_nodes)
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
            # 将 user_query 包装为 HumanMessage，与 agents.py 中的数据类型预期一致
            from langchain_core.messages import HumanMessage
            config = {"configurable": {"thread_id": f"{thread_id}_planner"}}
            initial_state = {
                "planner_messages": [HumanMessage(content=user_query)],
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

    def _extract_questions_from_nodes(self, retrieved_nodes: list) -> List[str]:
        """从检索结果的 nodes 中提取 metadata 中的 questions

        Args:
            retrieved_nodes: 检索到的 NodeWithScore 列表

        Returns:
            提取的问题列表（去重后）
        """
        questions_set = set()

        for node_with_score in retrieved_nodes:
            try:
                metadata = node_with_score.node.metadata
                if metadata and "questions" in metadata:
                    questions = metadata["questions"]
                    # questions 可能是字符串列表或单个字符串
                    if isinstance(questions, list):
                        for q in questions:
                            if q and isinstance(q, str):
                                questions_set.add(q.strip())
                    elif isinstance(questions, str):
                        questions_set.add(questions.strip())
            except Exception as e:
                logger.warning(f"提取节点问题时出错: {e}")
                continue

        return list(questions_set)

    async def _generate_answer(
        self,
        user_query: str,
        sub_questions: List[str],
        question_pool: List[str],
        retrieved_nodes: list
    ) -> str:
        """基于检索结果生成最终答案

        Args:
            user_query: 用户原始查询
            sub_questions: Planner 生成的子问题
            question_pool: 完整的问题池（包含改写问题）
            retrieved_nodes: 检索到的节点列表

        Returns:
            生成的答案
        """
        try:
            # 构建检索上下文
            retrieved_contexts = []
            for node_with_score in retrieved_nodes:
                node = node_with_score.node
                metadata = node.metadata
                source = metadata.get('file_name', metadata.get('source', 'Unknown'))
                content = node.get_content()

                # 截断过长内容
                if len(content) > 500:
                    content = content[:500] + "..."

                retrieved_contexts.append(f"来源: {source}\n内容: {content}\n")

            contexts_str = "\n".join(retrieved_contexts)
            questions_str = "\n".join([f"- {q}" for q in question_pool])

            # 构建提示词
            prompt = f"""你是一个专业的研究助手。基于以下信息回答用户的问题。

用户问题：
{user_query}

相关子问题：
{questions_str}

检索到的相关信息：
{contexts_str}

要求：
1. 回答应该全面且准确
2. 引用具体的来源信息
3. 如果信息不足，明确说明
4. 使用清晰的结构组织回答

回答："""

            # 调用 LLM 生成答案
            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"答案生成完成，长度: {len(answer)} 字符")
            return answer

        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，生成答案时出现错误: {str(e)}"
    
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

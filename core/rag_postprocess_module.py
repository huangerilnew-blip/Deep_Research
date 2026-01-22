#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 检索后处理模块--负责从index->retirever->重排序->过滤->去重->list[NodeWithScore]
构建好问题池
"""

import hashlib
import logging
from typing import List, Dict, Any
from concurrent_log_handler import ConcurrentRotatingFileHandler

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from core.config import Config
from core.models import BGERerankNodePostprocessor
from llama_index.core.base.base_retriever import BaseRetriever
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


# 答案生成提示词
ANSWER_GENERATION_PROMPT = """
你是一个专业的研究助手。基于以下信息回答用户的问题。

用户问题：
{original_query}

相关子问题：
{planner_questions}

检索到的相关信息：
{retrieved_contexts}

要求：
1. 回答应该全面且准确
2. 引用具体的来源信息
3. 如果信息不足，明确说明
4. 使用清晰的结构组织回答

回答：
"""


class RAGPostProcessModule:
    """RAG 检索后处理模块
    
    负责向量检索、去重、重排序、答案生成
    """
    
    def __init__(
        self,
        vector_store: VectorStoreIndex,
        retriever:BaseRetriever,
        node_postprocessor: BGERerankNodePostprocessor,
        top_k: int = Config.TOP_K
    ):
        """初始化 RAG 模块
        
        Args:
            vector_store: 向量存储索引
            node_postprocessor: BGE Reranker 节点后处理器
            llm: 用于生成回答的 LLM（已废弃，保留兼容性）
            top_k: 每个问题检索的文档数
        """
        self.vector_store = vector_store
        self.node_postprocessor = node_postprocessor
        self.top_k = top_k
        # 预构建 retriever，避免每个问题重复创建
        if retriever:
            self.retriever = retriever
        elif self.vector_store:
            self.retriever = self.vector_store.as_retriever(similarity_top_k=self.top_k)
        else:
             raise ValueError("Retriever 为空，请提供 Retriever 或 VectorStore")
        # 问题池，用于存储所有相关问题（去重后）
        self.question_pool = []
        
        logger.info(f"初始化 RAGModule: top_k={top_k}")
    
    async def retrieve_postprecess(
        self,
        planner_questions: List[str],
    ) -> List[NodeWithScore]:
        """检索相关语料
        
        Args:
            planner_questions: planner 的问题列表
            original_query: 用户的原始查询（保留兼容性，未使用）
            
        Returns:
            去重后的 NodeWithScore 列表
        """
        logger.info(f"开始 RAG 检索，planner问题数: {len(planner_questions)}")
        
        # 1. 对每个问题单独检索、重排序、过滤
        all_reranked_nodes = []
        for i, question in enumerate(planner_questions, 1):
            try:
                # 初步检索
                nodes = await self._retrieve_single_question(question)
                logger.debug(f"问题 {i}/{len(planner_questions)} 初步检索到 {len(nodes)} 个节点")
                
                # 使用 BGERerankNodePostprocessor 进行重排序和过滤
                reranked_nodes = await self._rerank_and_filter(question, nodes)
                logger.debug(f"问题 {i}/{len(planner_questions)} 重排序+过滤后保留 {len(reranked_nodes)} 个节点")
                
                all_reranked_nodes.extend(reranked_nodes)
            except Exception as e:
                logger.error(f"处理问题 {i} 失败: {e}")
                continue
        
        logger.info(f"所有问题检索并重排序完成，共 {len(all_reranked_nodes)} 个节点")
        
        # 2. 去重（根据 Node.text，保留最大分数）
        unique_nodes = self._deduplicate_by_text(all_reranked_nodes)
        logger.info(f"去重后保留 {len(unique_nodes)} 个节点")
        
        # 3. 构建 question_pool
        self._build_question_pool(planner_questions, unique_nodes)
        logger.info(f"构建问题池完成，共 {len(self.question_pool)} 个问题")
        
        return unique_nodes
    
    async def _retrieve_single_question(
        self,
        question: str
    ) -> List[NodeWithScore]:
        """从向量库检索单个问题的相关语料
        
        Args:
            question: 单个问题
            
        Returns:
            检索到的节点列表
            
        Raises:
            Exception: 检索失败时抛出异常
        """
        # 使用在 __init__ 中构建好的 retriever，避免重复创建
        nodes = self.retriever.retrieve(question)
        return nodes
    
    async def _rerank_and_filter(
        self,
        query: str,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """使用 BGERerankNodePostprocessor 进行重排序和过滤
        
        Args:
            query: 查询问题
            nodes: 初步检索的节点列表
            
        Returns:
            重排序并过滤后的节点列表
            
        Raises:
            Exception: 重排序失败时抛出异常
        """
        if not nodes:
            raise ValueError(f"过滤时，目标node为空")
        
        # 构造 QueryBundle
        query_bundle = QueryBundle(query_str=query)
        
        # 调用 node_postprocessor 进行重排序和过滤
        # BGERerankNodePostprocessor 会自动根据 score_threshold 过滤
        reranked_nodes = await self.node_postprocessor._async_postprocess_nodes(
            nodes, query_bundle
        )
        
        return reranked_nodes
    
    def _deduplicate_by_text(
        self,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """根据 Node.text 去重，保留最大分数
        
        Args:
            nodes: 节点列表
            
        Returns:
            去重后的节点列表
        """
        if not nodes:
            return []
        
        # 使用字典存储，key 为节点文本，value 为节点
        unique_dict = {}
        
        for node_with_score in nodes:
            # 获取节点文本内容
            text = node_with_score.node.get_content()
            
            # 如果文本不存在，或者当前节点分数更高，则保留
            if text not in unique_dict or node_with_score.score > unique_dict[text].score:
                unique_dict[text] = node_with_score
        
        # 转换为列表并按分数排序
        unique_nodes = list(unique_dict.values())
        unique_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return unique_nodes
    
    def _build_question_pool(
        self,
        planner_questions: List[str],
        nodes: List[NodeWithScore]
    ) -> None:
        """构建问题池，由 planner_questions 和节点 metadata 中的 questions 组合而成
        
        Args:
            planner_questions: planner 生成的问题列表
            nodes: 过滤后的节点列表
        """
        question_set = set()
        
        # 1. 添加 planner_questions
        for question in planner_questions:
            if question and isinstance(question, str):
                question_set.add(question.strip())
        
        # 2. 从节点 metadata 中提取 questions
        for node_with_score in nodes:
            try:
                metadata = node_with_score.node.metadata
                if metadata and "questions" in metadata:
                    questions = metadata["questions"]
                    # questions 可能是字符串列表或单个字符串
                    if isinstance(questions, list):
                        for q in questions:
                            if q and isinstance(q, str):
                                question_set.add(q.strip())
                    elif isinstance(questions, str):
                        question_set.add(questions.strip())
            except Exception as e:
                logger.warning(f"提取节点问题时出错: {e}")
                continue
        
        # 3. 转换为列表并存储到 self.question_pool
        self.question_pool = list(question_set)
        logger.debug(f"问题池构建完成，包含 {len(self.question_pool)} 个去重后的问题")
    def get_question_pool(self) -> List[str]:
        """获取问题池
        Returns:
            问题池列表
        """
        if not self.question_pool:
            raise ValueError("question_pool为空")
        return self.question_pool



# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("RAGModule 测试")
    print("=" * 60)
    print("✓ RAGModule 模块导入成功")
    print("\n注意：完整测试需要初始化向量存储和 LLM")

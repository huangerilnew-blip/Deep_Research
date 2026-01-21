#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 模块
负责向量检索、去重、重排序、答案生成
"""

import hashlib
import logging
from typing import List, Dict, Any
from concurrent_log_handler import ConcurrentRotatingFileHandler

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms.llm import LLM

from core.config import Config
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


# 答案生成提示词
ANSWER_GENERATION_PROMPT = """
你是一个专业的研究助手。基于以下信息回答用户的问题。

用户问题：
{original_query}

相关子问题：
{questions_pool}

检索到的相关信息：
{retrieved_contexts}

要求：
1. 回答应该全面且准确
2. 引用具体的来源信息
3. 如果信息不足，明确说明
4. 使用清晰的结构组织回答

回答：
"""


class RAGModule:
    """RAG 模块
    
    负责向量检索、去重、重排序、答案生成
    """
    
    def __init__(
        self,
        vector_store: VectorStoreIndex,
        reranker: BGEReranker,
        llm: LLM,
        top_k: int = Config.TOP_K,
        rerank_top_n: int = Config.RERANK_TOP_N
    ):
        """初始化 RAG 模块
        
        Args:
            vector_store: 向量存储索引
            reranker: BGE Reranker 实例
            llm: 用于生成回答的 LLM
            top_k: 每个问题检索的文档数
            rerank_top_n: Rerank 后保留的文档数
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        
        logger.info(f"初始化 RAGModule: top_k={top_k}, rerank_top_n={rerank_top_n}")
    
    async def retrieve_and_generate(
        self,
        questions_pool: List[str],
        original_query: str
    ) -> Dict[str, Any]:
        """检索相关语料并生成回答
        
        Args:
            questions_pool: 所有问题（原始 + 改写）
            original_query: 用户的原始查询
            
        Returns:
            包含回答和检索元数据的字典
        """
        logger.info(f"开始 RAG 检索和生成，问题数: {len(questions_pool)}")
        
        # 1. 检索相关语料
        contexts = await self._retrieve_contexts(questions_pool)
        logger.info(f"检索到 {len(contexts)} 个相关节点")
        
        # 2. 去重
        unique_contexts = self._deduplicate_contexts(contexts)
        logger.info(f"去重后剩余 {len(unique_contexts)} 个节点")
        
        # 3. 重排序
        reranked_contexts = await self._rerank_contexts(original_query, unique_contexts)
        logger.info(f"重排序后保留 {len(reranked_contexts)} 个节点")
        
        # 4. 生成回答
        answer = await self._generate_answer(original_query, reranked_contexts, questions_pool)
        
        # 构建返回结果
        result = {
            'answer': answer,
            'retrieved_count': len(contexts),
            'unique_count': len(unique_contexts),
            'reranked_count': len(reranked_contexts),
            'contexts': [
                {
                    'content': node.node.get_content()[:200],  # 只返回前200字符
                    'score': node.score,
                    'metadata': node.node.metadata
                }
                for node in reranked_contexts[:5]  # 只返回前5个
            ]
        }
        
        logger.info("RAG 检索和生成完成")
        return result
    
    async def _retrieve_contexts(
        self,
        questions: List[str]
    ) -> List[NodeWithScore]:
        """从向量库检索相关语料
        
        Args:
            questions: 问题列表
            
        Returns:
            检索到的节点列表
        """
        all_contexts = []
        
        # 获取检索器
        retriever = self.vector_store.as_retriever(similarity_top_k=self.top_k)
        
        # 对每个问题进行检索
        for i, question in enumerate(questions):
            try:
                # 检索
                nodes = retriever.retrieve(question)
                all_contexts.extend(nodes)
                logger.debug(f"问题 {i+1} 检索到 {len(nodes)} 个节点")
            except Exception as e:
                logger.error(f"检索问题 {i+1} 失败: {e}")
                continue
        
        return all_contexts
    
    def _deduplicate_contexts(
        self,
        contexts: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """去重检索结果
        
        基于内容哈希去重，保留相似度最高的节点
        
        Args:
            contexts: 检索结果列表
            
        Returns:
            去重后的结果列表
        """
        if not contexts:
            return []
        
        # 使用字典存储，key 为内容哈希，value 为节点
        unique_dict = {}
        
        for node_with_score in contexts:
            # 计算内容哈希
            content = node_with_score.node.get_content()
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # 如果哈希不存在，或者当前节点分数更高，则保留
            if content_hash not in unique_dict or node_with_score.score > unique_dict[content_hash].score:
                unique_dict[content_hash] = node_with_score
        
        # 转换为列表并按分数排序
        unique_contexts = list(unique_dict.values())
        unique_contexts.sort(key=lambda x: x.score, reverse=True)
        
        return unique_contexts
    
    async def _rerank_contexts(
        self,
        query: str,
        contexts: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """使用 Reranker 重排序
        
        Args:
            query: 原始查询
            contexts: 去重后的检索结果
            
        Returns:
            重排序后的结果列表
        """
        if not contexts:
            return []
        
        try:
            # 提取文档内容
            documents = [node.node.get_content() for node in contexts]
            
            # 调用 Reranker
            rerank_results = await self.reranker.rerank_async(query, documents)
            
            # 根据 rerank 结果重新排序
            reranked_contexts = []
            for item in rerank_results:
                idx = item['index']
                score = item['score']
                
                # 过滤低分
                if score >= Config.RERANK_THRESHOLD:
                    node_with_score = contexts[idx]
                    # 更新分数为 rerank 分数
                    node_with_score.score = score
                    reranked_contexts.append(node_with_score)
            
            # 限制数量
            reranked_contexts = reranked_contexts[:self.rerank_top_n]
            
            return reranked_contexts
            
        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            # 降级：返回原始结果
            return contexts[:self.rerank_top_n]
    
    async def _generate_answer(
        self,
        query: str,
        contexts: List[NodeWithScore],
        questions_pool: List[str]
    ) -> str:
        """生成最终回答
        
        Args:
            query: 原始查询
            contexts: 检索到的相关语料
            questions_pool: 问题池
            
        Returns:
            生成的回答
        """
        if not contexts:
            return "抱歉，没有找到相关信息来回答您的问题。"
        
        try:
            # 构建上下文文本
            context_texts = []
            for i, node_with_score in enumerate(contexts[:10], 1):  # 最多使用前10个
                content = node_with_score.node.get_content()
                metadata = node_with_score.node.metadata
                source = metadata.get('source', 'unknown')
                
                context_texts.append(f"[{i}] 来源: {source}\n内容: {content[:300]}...")
            
            retrieved_contexts = "\n\n".join(context_texts)
            
            # 构建问题池文本
            questions_text = "\n".join([f"- {q}" for q in questions_pool[:10]])  # 最多显示前10个
            
            # 构建提示词
            prompt = ANSWER_GENERATION_PROMPT.format(
                original_query=query,
                questions_pool=questions_text,
                retrieved_contexts=retrieved_contexts
            )
            
            # 调用 LLM 生成回答
            response = await self.llm.acomplete(prompt)
            answer = response.text.strip()
            
            logger.info(f"生成回答，长度: {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 降级：返回检索结果摘要
            summary = "抱歉，无法生成完整的回答。以下是检索到的相关信息：\n\n"
            for i, node_with_score in enumerate(contexts[:3], 1):
                content = node_with_score.node.get_content()
                summary += f"{i}. {content[:200]}...\n\n"
            
            return summary


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("RAGModule 测试")
    print("=" * 60)
    print("✓ RAGModule 模块导入成功")
    print("\n注意：完整测试需要初始化向量存储和 LLM")

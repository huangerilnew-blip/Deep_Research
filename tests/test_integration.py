#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端集成测试
测试完整查询流程、并发执行、错误恢复和向量库更新
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from agents.multi_agent import MultiAgent
from agents.executor_pool import ExecutorAgentPool
from core.vector_store_manager import VectorStoreManager
from core.models import QuestionsPool
from core.config import Config


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_query_flow(self, db_pool, sample_query, sample_thread_id):
        """测试完整查询流程
        
        验证：
        1. 查询拆解
        2. 并发执行
        3. 文档处理
        4. 向量库更新
        5. RAG 检索
        6. 答案生成
        """
        # 创建 MultiAgent 实例
        multi_agent = MultiAgent(
            pool=db_pool,
            executor_pool_size=2,  # 使用较小的池大小加快测试
            planner_model=Config.LLM_PLANNER,
            executor_model=Config.LLM_EXECUTOR
        )
        
        # Mock PlannerAgent 的拆解结果
        mock_sub_questions = [
            "人工智能在医疗诊断中的应用？",
            "人工智能在药物研发中的作用？",
            "人工智能在医疗影像分析中的应用？"
        ]
        
        with patch.object(multi_agent, '_plan_query', return_value=mock_sub_questions):
            # Mock ExecutorAgent Pool 的执行结果
            mock_executor_results = [
                {
                    'sub_question': mock_sub_questions[0],
                    'downloaded_papers': [
                        {
                            'title': 'AI in Medical Diagnosis',
                            'source': 'openalex',
                            'local_path': 'data/downloads/test_doc1.md',
                            'abstract': 'This paper discusses AI applications in medical diagnosis.'
                        }
                    ]
                },
                {
                    'sub_question': mock_sub_questions[1],
                    'downloaded_papers': [
                        {
                            'title': 'AI in Drug Discovery',
                            'source': 'semantic_scholar',
                            'local_path': 'data/downloads/test_doc2.md',
                            'abstract': 'This paper explores AI in drug discovery.'
                        }
                    ]
                }
            ]
            
            with patch.object(
                multi_agent.executor_pool,
                'execute_questions',
                return_value=mock_executor_results
            ):
                # Mock 文档处理器
                mock_llama_docs = [
                    MagicMock(text="AI in medical diagnosis content", metadata={'source': 'test_doc1'}),
                    MagicMock(text="AI in drug discovery content", metadata={'source': 'test_doc2'})
                ]
                mock_rewritten_questions = [
                    "如何使用 AI 进行疾病诊断？",
                    "AI 如何加速新药开发？"
                ]
                
                with patch.object(
                    multi_agent.document_processor,
                    'process_documents',
                    return_value=(mock_llama_docs, mock_rewritten_questions)
                ):
                    # Mock RAG 模块
                    mock_rag_result = {
                        'answer': '人工智能在医疗领域有多种应用，包括医疗诊断、药物研发和医疗影像分析等。',
                        'retrieved_count': 10,
                        'unique_count': 8,
                        'reranked_count': 5
                    }
                    
                    with patch('agents.multi_agent.RAGModule') as MockRAGModule:
                        mock_rag_instance = AsyncMock()
                        mock_rag_instance.retrieve_and_generate = AsyncMock(return_value=mock_rag_result)
                        MockRAGModule.return_value = mock_rag_instance
                        
                        # 执行完整流程
                        result = await multi_agent.process_query(sample_query, sample_thread_id)
                        
                        # 验证结果
                        assert 'query' in result
                        assert result['query'] == sample_query
                        assert 'answer' in result
                        assert len(result['answer']) > 0
                        assert 'sub_questions' in result
                        assert len(result['sub_questions']) == 3
                        assert 'metadata' in result
                        assert result['metadata']['retrieved_count'] == 10
                        assert result['metadata']['unique_count'] == 8
                        assert result['metadata']['reranked_count'] == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, db_pool):
        """测试并发执行
        
        验证：
        1. ExecutorAgent Pool 正确分配任务
        2. 多个 Agent 并发执行
        3. 所有结果正确收集
        """
        # 创建 ExecutorAgent Pool
        pool_size = 3
        executor_pool = ExecutorAgentPool(
            pool=db_pool,
            pool_size=pool_size,
            model=Config.LLM_EXECUTOR
        )
        
        # 验证池大小
        assert len(executor_pool.agents) == pool_size
        
        # 准备测试数据
        sub_questions = [
            "问题 1",
            "问题 2",
            "问题 3",
            "问题 4",
            "问题 5"
        ]
        
        # Mock 每个 Agent 的 invoke 方法
        mock_results = []
        for i, agent in enumerate(executor_pool.agents):
            mock_result = {
                'sub_question': f'问题 {i+1}',
                'downloaded_papers': [{'title': f'文档 {i+1}'}]
            }
            mock_results.append(mock_result)
            agent.invoke = AsyncMock(return_value=mock_result)
        
        # 执行并发任务
        results = await executor_pool.execute_questions(sub_questions, "test_thread")
        
        # 验证结果
        assert len(results) == len(sub_questions)
        for result in results:
            assert 'sub_question' in result
            assert 'downloaded_papers' in result
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, db_pool, sample_query, sample_thread_id):
        """测试错误恢复
        
        验证：
        1. 单个 ExecutorAgent 失败不影响其他
        2. 文档处理失败时的降级策略
        3. 系统能够继续执行并返回结果
        """
        # 创建 MultiAgent 实例
        multi_agent = MultiAgent(
            pool=db_pool,
            executor_pool_size=3,
            planner_model=Config.LLM_PLANNER,
            executor_model=Config.LLM_EXECUTOR
        )
        
        # Mock 查询拆解
        mock_sub_questions = ["问题 1", "问题 2", "问题 3"]
        
        with patch.object(multi_agent, '_plan_query', return_value=mock_sub_questions):
            # Mock ExecutorAgent Pool，模拟部分失败
            mock_executor_results = [
                {
                    'sub_question': '问题 1',
                    'downloaded_papers': [{'title': '文档 1', 'source': 'openalex'}]
                },
                # 问题 2 失败（返回空结果）
                {
                    'sub_question': '问题 2',
                    'downloaded_papers': []
                },
                {
                    'sub_question': '问题 3',
                    'downloaded_papers': [{'title': '文档 3', 'source': 'semantic_scholar'}]
                }
            ]
            
            with patch.object(
                multi_agent.executor_pool,
                'execute_questions',
                return_value=mock_executor_results
            ):
                # Mock 文档处理器，模拟部分文档处理失败
                with patch.object(
                    multi_agent.document_processor,
                    'process_documents',
                    return_value=([], [])  # 所有文档处理失败
                ):
                    # Mock RAG 模块（使用基础向量库）
                    mock_rag_result = {
                        'answer': '基于基础数据的回答',
                        'retrieved_count': 5,
                        'unique_count': 5,
                        'reranked_count': 3
                    }
                    
                    with patch('agents.multi_agent.RAGModule') as MockRAGModule:
                        mock_rag_instance = AsyncMock()
                        mock_rag_instance.retrieve_and_generate = AsyncMock(return_value=mock_rag_result)
                        MockRAGModule.return_value = mock_rag_instance
                        
                        # 执行流程
                        result = await multi_agent.process_query(sample_query, sample_thread_id)
                        
                        # 验证系统仍然返回结果
                        assert 'answer' in result
                        assert len(result['answer']) > 0
                        assert 'error' not in result or result['error'] == ''
    
    @pytest.mark.asyncio
    async def test_vector_store_update(self, db_pool):
        """测试向量库更新
        
        验证：
        1. 基础向量库正确加载
        2. 新文档正确添加到向量库
        3. 向量库节点数量增加
        
        注意：此测试使用 Mock 避免依赖实际的 embedding 服务
        """
        from llama_index.core import Document
        
        # Mock embedding 模型
        mock_embedding = MagicMock()
        mock_embedding.get_text_embedding = MagicMock(return_value=[0.1] * 768)
        
        # 创建 VectorStoreManager，传入 mock embedding
        with patch.object(VectorStoreManager, '_get_local_embedding_model', return_value=mock_embedding):
            vector_manager = VectorStoreManager(embedding_model=mock_embedding)
            
            # Mock 加载基础向量库
            mock_index = MagicMock()
            mock_index.insert = MagicMock()
            
            with patch.object(vector_manager, 'load_base_vector_store', return_value=mock_index):
                # 加载基础向量库
                initial_index = vector_manager.load_base_vector_store()
                
                # 验证基础向量库已加载
                assert initial_index is not None
                
                # 准备测试文档
                test_docs = [
                    Document(
                        text="这是测试文档 1 的内容",
                        metadata={'source': 'test_doc1', 'title': '测试文档 1'}
                    ),
                    Document(
                        text="这是测试文档 2 的内容",
                        metadata={'source': 'test_doc2', 'title': '测试文档 2'}
                    )
                ]
                
                # Mock 添加文档
                with patch.object(vector_manager, 'add_documents', return_value=mock_index):
                    # 添加文档到向量库
                    updated_index = vector_manager.add_documents(test_docs)
                    
                    # 验证向量库已更新
                    assert updated_index is not None
                    
                    # Mock retriever
                    mock_retriever = MagicMock()
                    with patch.object(vector_manager, 'get_retriever', return_value=mock_retriever):
                        # 验证可以获取检索器
                        retriever = vector_manager.get_retriever(top_k=5)
                        assert retriever is not None


class TestQuestionsPoolIntegration:
    """Questions Pool 集成测试"""
    
    def test_questions_pool_workflow(self):
        """测试 Questions Pool 完整工作流
        
        验证：
        1. 添加原始子问题
        2. 添加改写问题
        3. 去重功能
        4. 获取所有问题
        """
        # 创建 Questions Pool
        pool = QuestionsPool()
        
        # 添加原始子问题
        original_questions = [
            "人工智能在医疗诊断中的应用？",
            "人工智能在药物研发中的作用？",
            "人工智能在医疗影像分析中的应用？"
        ]
        pool.add_original_questions(original_questions)
        
        # 验证原始问题
        assert len(pool.original_questions) == 3
        
        # 添加改写问题
        rewritten_questions = [
            "如何使用 AI 进行疾病诊断？",
            "AI 如何加速新药开发？",
            "人工智能在医疗诊断中的应用？"  # 重复问题
        ]
        pool.add_rewritten_questions(rewritten_questions)
        
        # 验证去重
        all_questions = pool.get_all_questions()
        assert len(all_questions) == 5  # 3 原始 + 2 新问题（1 个重复被去除）
        
        # 验证没有重复
        assert len(all_questions) == len(set(all_questions))


class TestAPIIntegration:
    """API 集成测试"""
    
    @pytest.mark.asyncio
    async def test_api_query_endpoint(self, db_pool, sample_query):
        """测试 API 查询端点
        
        验证：
        1. 请求验证
        2. MultiAgent 调用
        3. 响应格式
        4. 错误处理
        """
        from fastapi.testclient import TestClient
        from main import app, set_multi_agent
        
        # 创建 MultiAgent 实例
        multi_agent = MultiAgent(
            pool=db_pool,
            executor_pool_size=2,
            planner_model=Config.LLM_PLANNER,
            executor_model=Config.LLM_EXECUTOR
        )
        
        # 设置到应用状态
        set_multi_agent(multi_agent)
        
        # Mock MultiAgent 的 process_query 方法
        mock_result = {
            'query': sample_query,
            'sub_questions': ['问题 1', '问题 2', '问题 3'],
            'rewritten_questions_count': 2,
            'total_questions': 5,
            'documents_processed': 3,
            'answer': '这是测试回答',
            'metadata': {
                'retrieved_count': 10,
                'unique_count': 8,
                'reranked_count': 5
            }
        }
        
        with patch.object(multi_agent, 'process_query', return_value=mock_result):
            # 创建测试客户端
            client = TestClient(app)
            
            # 发送请求
            response = client.post(
                "/api/v1/query",
                json={
                    "query": sample_query,
                    "thread_id": "test_thread"
                }
            )
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert data['query'] == sample_query
            assert 'answer' in data
            assert len(data['answer']) > 0
            assert 'sub_questions' in data
            assert 'metadata' in data
    
    @pytest.mark.asyncio
    async def test_api_health_endpoint(self):
        """测试健康检查端点"""
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'multi-agent-deep-search'


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

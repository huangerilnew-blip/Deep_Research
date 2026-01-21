#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终验证测试
验证所有核心模块的基本功能，不依赖外部服务
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_module():
    """测试配置模块"""
    print("\n1. 测试配置模块...")
    try:
        from core.config import Config
        
        # 验证关键配置项存在
        assert hasattr(Config, 'EXECUTOR_POOL_SIZE')
        assert hasattr(Config, 'MAX_CHUNK_SIZE')
        assert hasattr(Config, 'MINERU_BASE_URL')
        assert hasattr(Config, 'VLLM_BASE_URL')
        assert hasattr(Config, 'EMBEDDING_MODEL_NAME')
        
        # 验证配置值类型
        assert isinstance(Config.EXECUTOR_POOL_SIZE, int)
        assert isinstance(Config.MAX_CHUNK_SIZE, int)
        assert Config.EXECUTOR_POOL_SIZE > 0
        assert Config.MAX_CHUNK_SIZE > 0
        
        print(f"   ✓ Config 模块验证通过")
        print(f"     - EXECUTOR_POOL_SIZE: {Config.EXECUTOR_POOL_SIZE}")
        print(f"     - MAX_CHUNK_SIZE: {Config.MAX_CHUNK_SIZE}")
        return True
    except Exception as e:
        print(f"   ✗ Config 模块验证失败: {e}")
        return False


def test_questions_pool():
    """测试 Questions Pool 数据模型"""
    print("\n2. 测试 Questions Pool...")
    try:
        from core.models import QuestionsPool
        
        # 创建 Questions Pool
        pool = QuestionsPool()
        
        # 测试添加原始问题
        original_questions = ["问题1？", "问题2？", "问题3？"]
        pool.add_original_questions(original_questions)
        assert len(pool.original_questions) == 3
        
        # 测试添加改写问题
        rewritten_questions = ["改写问题1？", "改写问题2？"]
        pool.add_rewritten_questions(rewritten_questions)
        assert len(pool.rewritten_questions) == 2
        
        # 测试获取所有问题
        all_questions = pool.get_all_questions()
        assert len(all_questions) == 5
        
        # 测试去重
        pool.add_rewritten_questions(["问题1？"])  # 重复问题
        all_questions = pool.get_all_questions()
        assert len(all_questions) == 5  # 应该还是 5 个
        
        print(f"   ✓ Questions Pool 验证通过")
        print(f"     - 原始问题数: {len(pool.original_questions)}")
        print(f"     - 改写问题数: {len(pool.rewritten_questions)}")
        print(f"     - 总问题数: {len(all_questions)}")
        return True
    except Exception as e:
        print(f"   ✗ Questions Pool 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_metadata():
    """测试文档元数据模型"""
    print("\n3. 测试文档元数据模型...")
    try:
        from core.models import DocumentMetadata
        from datetime import datetime
        
        # 创建文档元数据
        metadata = DocumentMetadata(
            source="openalex",
            title="测试文档",
            abstract="这是一个测试摘要",
            url="https://example.com/doc",
            local_path="/path/to/doc.pdf",
            rerank_score=0.95,
            download_time=datetime.now()
        )
        
        # 验证字段
        assert metadata.source == "openalex"
        assert metadata.title == "测试文档"
        assert metadata.rerank_score == 0.95
        
        # 测试转换为字典
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict['source'] == "openalex"
        
        print(f"   ✓ DocumentMetadata 验证通过")
        return True
    except Exception as e:
        print(f"   ✗ DocumentMetadata 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_chunk():
    """测试 Markdown Chunk 模型"""
    print("\n4. 测试 Markdown Chunk 模型...")
    try:
        from core.models import MarkdownChunk
        
        # 创建 Markdown Chunk
        chunk = MarkdownChunk(
            content="这是一段测试内容",
            doc_title="测试文档",
            source="test_source",
            header_level=2,
            chunk_index=0
        )
        
        # 验证字段
        assert chunk.content == "这是一段测试内容"
        assert chunk.header_level == 2
        assert chunk.chunk_index == 0
        
        # 测试转换为字典
        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict['header_level'] == 2
        
        print(f"   ✓ MarkdownChunk 验证通过")
        return True
    except Exception as e:
        print(f"   ✗ MarkdownChunk 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieved_context():
    """测试检索上下文模型"""
    print("\n5. 测试 Retrieved Context 模型...")
    try:
        from core.models import RetrievedContext
        
        # 创建检索上下文
        context = RetrievedContext(
            content="这是检索到的内容",
            source="base_data",
            score=0.88,
            metadata={'key': 'value'}
        )
        
        # 验证字段
        assert context.content == "这是检索到的内容"
        assert context.score == 0.88
        assert context.metadata['key'] == 'value'
        
        print(f"   ✓ RetrievedContext 验证通过")
        return True
    except Exception as e:
        print(f"   ✗ RetrievedContext 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_imports():
    """测试所有核心模块导入"""
    print("\n6. 测试核心模块导入...")
    
    modules = [
        ('core.vector_store_manager', 'VectorStoreManager'),
        ('core.document_processor', 'DocumentProcessor'),
        ('core.rag_module', 'RAGModule'),
        ('agents.executor_pool', 'ExecutorAgentPool'),
        ('agents.multi_agent', 'MultiAgent'),
        ('api.routes', 'router'),
    ]
    
    success_count = 0
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            assert hasattr(module, class_name)
            print(f"   ✓ {module_name}.{class_name} 导入成功")
            success_count += 1
        except Exception as e:
            print(f"   ✗ {module_name}.{class_name} 导入失败: {e}")
    
    return success_count == len(modules)


def test_executor_pool_structure():
    """测试 ExecutorAgent Pool 结构"""
    print("\n7. 测试 ExecutorAgent Pool 结构...")
    try:
        from agents.executor_pool import ExecutorAgentPool
        import inspect
        
        # 验证类方法存在
        assert hasattr(ExecutorAgentPool, '__init__')
        assert hasattr(ExecutorAgentPool, '_initialize_agents')
        assert hasattr(ExecutorAgentPool, 'execute_questions')
        assert hasattr(ExecutorAgentPool, 'cleanup')
        
        # 验证方法签名
        init_sig = inspect.signature(ExecutorAgentPool.__init__)
        assert 'pool' in init_sig.parameters
        assert 'pool_size' in init_sig.parameters
        assert 'model' in init_sig.parameters
        
        print(f"   ✓ ExecutorAgentPool 结构验证通过")
        return True
    except Exception as e:
        print(f"   ✗ ExecutorAgentPool 结构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_agent_structure():
    """测试 MultiAgent 结构"""
    print("\n8. 测试 MultiAgent 结构...")
    try:
        from agents.multi_agent import MultiAgent
        import inspect
        
        # 验证类方法存在
        assert hasattr(MultiAgent, '__init__')
        assert hasattr(MultiAgent, 'process_query')
        assert hasattr(MultiAgent, '_initialize_vector_store')
        assert hasattr(MultiAgent, '_plan_query')
        assert hasattr(MultiAgent, '_cleanup')
        
        # 验证 process_query 方法签名
        process_sig = inspect.signature(MultiAgent.process_query)
        assert 'user_query' in process_sig.parameters
        assert 'thread_id' in process_sig.parameters
        
        print(f"   ✓ MultiAgent 结构验证通过")
        return True
    except Exception as e:
        print(f"   ✗ MultiAgent 结构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_module_structure():
    """测试 RAG Module 结构"""
    print("\n9. 测试 RAG Module 结构...")
    try:
        from core.rag_module import RAGModule
        import inspect
        
        # 验证类方法存在
        assert hasattr(RAGModule, '__init__')
        assert hasattr(RAGModule, 'retrieve_and_generate')
        assert hasattr(RAGModule, '_retrieve_contexts')
        assert hasattr(RAGModule, '_deduplicate_contexts')
        assert hasattr(RAGModule, '_rerank_contexts')
        assert hasattr(RAGModule, '_generate_answer')
        
        print(f"   ✓ RAGModule 结构验证通过")
        return True
    except Exception as e:
        print(f"   ✗ RAGModule 结构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_processor_structure():
    """测试 Document Processor 结构"""
    print("\n10. 测试 Document Processor 结构...")
    try:
        from core.document_processor import DocumentProcessor
        import inspect
        
        # 验证类方法存在
        assert hasattr(DocumentProcessor, '__init__')
        assert hasattr(DocumentProcessor, 'process_documents')
        assert hasattr(DocumentProcessor, '_pdf_to_markdown')
        assert hasattr(DocumentProcessor, '_split_by_headers')
        assert hasattr(DocumentProcessor, '_split_by_sentences')
        assert hasattr(DocumentProcessor, '_rewrite_to_questions')
        
        print(f"   ✓ DocumentProcessor 结构验证通过")
        return True
    except Exception as e:
        print(f"   ✗ DocumentProcessor 结构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_routes():
    """测试 API 路由"""
    print("\n11. 测试 API 路由...")
    try:
        from api.routes import router
        
        # 验证路由存在
        assert router is not None
        
        # 验证路由包含必要的端点
        routes = [route.path for route in router.routes]
        assert '/query' in routes or any('/query' in r for r in routes)
        
        print(f"   ✓ API 路由验证通过")
        print(f"     - 路由数量: {len(routes)}")
        return True
    except Exception as e:
        print(f"   ✗ API 路由验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证测试"""
    print("=" * 70)
    print("Multi-Agent 深度搜索系统 - 最终验证测试")
    print("=" * 70)
    
    tests = [
        test_config_module,
        test_questions_pool,
        test_document_metadata,
        test_markdown_chunk,
        test_retrieved_context,
        test_module_imports,
        test_executor_pool_structure,
        test_multi_agent_structure,
        test_rag_module_structure,
        test_document_processor_structure,
        test_api_routes,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n   ✗ 测试执行异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    print(f"失败: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ 所有验证测试通过！系统已准备就绪。")
        return 0
    else:
        print(f"\n❌ 有 {total - passed} 个测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

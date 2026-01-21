#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试所有核心模块"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Multi-Agent 深度搜索系统 - 快速测试")
print("=" * 70)

# 测试配置
print("\n1. 测试配置模块...")
try:
    from core.config import Config
    print(f"   ✓ Config 加载成功")
    print(f"     - EXECUTOR_POOL_SIZE: {Config.EXECUTOR_POOL_SIZE}")
    print(f"     - MAX_CHUNK_SIZE: {Config.MAX_CHUNK_SIZE}")
    print(f"     - VLLM_BASE_URL: {Config.VLLM_BASE_URL}")
except Exception as e:
    print(f"   ✗ Config 加载失败: {e}")
    sys.exit(1)

# 测试数据模型
print("\n2. 测试数据模型...")
try:
    from core.models import QuestionsPool, DocumentMetadata, MarkdownChunk, RetrievedContext
    pool = QuestionsPool()
    pool.add_original_questions(["问题1？", "问题2？"])
    print(f"   ✓ 数据模型导入成功")
    print(f"     - QuestionsPool: {pool}")
except Exception as e:
    print(f"   ✗ 数据模型导入失败: {e}")
    sys.exit(1)

# 测试向量存储管理器
print("\n3. 测试向量存储管理器...")
try:
    from core.vector_store_manager import VectorStoreManager
    print(f"   ✓ VectorStoreManager 导入成功")
except Exception as e:
    print(f"   ✗ VectorStoreManager 导入失败: {e}")
    sys.exit(1)

# 测试文档处理器
print("\n4. 测试文档处理器...")
try:
    from core.document_processor import DocumentProcessor
    print(f"   ✓ DocumentProcessor 导入成功")
except Exception as e:
    print(f"   ✗ DocumentProcessor 导入失败: {e}")
    sys.exit(1)

# 测试 RAG 模块
print("\n5. 测试 RAG 模块...")
try:
    from core.rag_module import RAGModule
    print(f"   ✓ RAGModule 导入成功")
except Exception as e:
    print(f"   ✗ RAGModule 导入失败: {e}")
    sys.exit(1)

# 测试 ExecutorAgent Pool
print("\n6. 测试 ExecutorAgent Pool...")
try:
    from agents.executor_pool import ExecutorAgentPool
    print(f"   ✓ ExecutorAgentPool 导入成功")
except Exception as e:
    print(f"   ✗ ExecutorAgentPool 导入失败: {e}")
    sys.exit(1)

# 测试 MultiAgent
print("\n7. 测试 MultiAgent 协调器...")
try:
    from agents.multi_agent import MultiAgent
    print(f"   ✓ MultiAgent 导入成功")
except Exception as e:
    print(f"   ✗ MultiAgent 导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有核心模块测试通过！")
print("=" * 70)
print("\n系统已准备就绪，可以开始使用。")
print("\n下一步：")
print("  1. 确保所有依赖服务运行（PostgreSQL, vllm, TEI, MinerU）")
print("  2. 创建 API 接口（api/routes.py）")
print("  3. 更新主程序（main.py）")
print("  4. 运行完整的端到端测试")

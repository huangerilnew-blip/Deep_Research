#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BGERerankNodePostprocessor 使用示例

演示如何使用 BGERerankNodePostprocessor 对检索结果进行重排序
"""

import asyncio
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import QueryBundle

from core.models import BGERerankNodePostprocessor
from core.reranker import BGEReranker
from core.config import Config


async def example_basic_usage():
    """基础使用示例：在查询引擎中集成 Reranker"""
    print("=" * 60)
    print("示例 1: 基础使用 - 在查询引擎中集成 Reranker")
    print("=" * 60)
    
    # 1. 准备测试文档
    documents = [
        Document(text="人工智能技术在医疗诊断中发挥着重要作用", metadata={"source": "doc1"}),
        Document(text="机器学习算法可以帮助医生更准确地诊断疾病", metadata={"source": "doc2"}),
        Document(text="今天天气很好，适合出去玩", metadata={"source": "doc3"}),
        Document(text="深度学习在图像识别领域取得了突破性进展", metadata={"source": "doc4"}),
        Document(text="AI 辅助医疗影像分析提高了诊断准确率", metadata={"source": "doc5"}),
    ]
    
    # 2. 创建向量索引（这里简化，实际应使用配置的 Embedding 模型）
    # index = VectorStoreIndex.from_documents(documents)
    print(f"创建了包含 {len(documents)} 个文档的向量索引")
    
    # 3. 初始化 BGEReranker
    reranker = BGEReranker(
        base_url=Config.RERANK_BASE_URL
    )
    print(f"初始化 BGEReranker: {Config.RERANK_BASE_URL}")
    
    # 4. 创建 BGERerankNodePostprocessor
    rerank_postprocessor = BGERerankNodePostprocessor(
        reranker=reranker,
        top_n=3,  # 保留前 3 个结果
        score_threshold=0.5  # 分数阈值
    )
    print(f"创建 BGERerankNodePostprocessor (top_n=3, threshold=0.5)")
    
    # 5. 使用方式 1: 作为查询引擎的 postprocessor
    # query_engine = index.as_query_engine(
    #     node_postprocessors=[rerank_postprocessor]
    # )
    # response = query_engine.query("人工智能在医疗领域的应用")
    
    print("\n✓ BGERerankNodePostprocessor 创建成功！")
    print("  可以将其传递给 query_engine 的 node_postprocessors 参数")
    
    await reranker.close()


async def example_direct_postprocessing():
    """直接使用 postprocessor 对节点进行重排序"""
    print("\n" + "=" * 60)
    print("示例 2: 直接使用 postprocessor 对节点进行重排序")
    print("=" * 60)
    
    from llama_index.core.schema import NodeWithScore, TextNode
    
    # 1. 创建模拟的检索结果
    nodes = [
        NodeWithScore(
            node=TextNode(text="人工智能技术在医疗诊断中发挥着重要作用", metadata={"source": "doc1"}),
            score=0.75
        ),
        NodeWithScore(
            node=TextNode(text="今天天气很好，适合出去玩", metadata={"source": "doc3"}),
            score=0.70
        ),
        NodeWithScore(
            node=TextNode(text="AI 辅助医疗影像分析提高了诊断准确率", metadata={"source": "doc5"}),
            score=0.68
        ),
        NodeWithScore(
            node=TextNode(text="深度学习在图像识别领域取得了突破性进展", metadata={"source": "doc4"}),
            score=0.65
        ),
    ]
    
    print(f"原始检索结果（{len(nodes)} 个节点）:")
    for i, node in enumerate(nodes, 1):
        print(f"  {i}. [score={node.score:.2f}] {node.node.get_content()[:40]}...")
    
    # 2. 初始化 Reranker 和 Postprocessor
    reranker = BGEReranker(base_url=Config.RERANK_BASE_URL)
    rerank_postprocessor = BGERerankNodePostprocessor(
        reranker=reranker,
        top_n=3,
        score_threshold=0.0
    )
    
    # 3. 创建查询
    query = "人工智能在医疗领域的应用"
    query_bundle = QueryBundle(query_str=query)
    
    # 4. 使用 postprocessor 进行重排序
    print(f"\n使用 BGEReranker 对查询 '{query}' 进行重排序...")
    reranked_nodes = await rerank_postprocessor._async_postprocess_nodes(
        nodes, query_bundle
    )
    
    print(f"\nRerank 后的结果（{len(reranked_nodes)} 个节点）:")
    for i, node in enumerate(reranked_nodes, 1):
        print(f"  {i}. [rerank_score={node.score:.4f}] {node.node.get_content()[:40]}...")
    
    await reranker.close()


async def example_with_retriever():
    """在检索器中使用 postprocessor"""
    print("\n" + "=" * 60)
    print("示例 3: 在检索器中使用 postprocessor")
    print("=" * 60)
    
    # 这个示例展示了如何在实际的 RAG 流程中使用
    print("""
使用方式：

from core.models import BGERerankNodePostprocessor
from core.reranker import BGEReranker

# 1. 创建 Reranker
reranker = BGEReranker()

# 2. 创建 Postprocessor
postprocessor = BGERerankNodePostprocessor(
    reranker=reranker,
    top_n=5,
    score_threshold=0.3
)

# 3. 在查询引擎中使用
query_engine = vector_index.as_query_engine(
    node_postprocessors=[postprocessor],
    similarity_top_k=20  # 先检索更多结果，再用 rerank 筛选
)

# 4. 执行查询
response = query_engine.query("你的查询问题")
    """)


async def main():
    """运行所有示例"""
    await example_basic_usage()
    await example_direct_postprocessing()
    await example_with_retriever()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

"""
测试 ExecutorAgent 的 LangGraph 流程
"""
import asyncio
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from agents import ExecutorAgent
from psycopg_pool import AsyncConnectionPool
from config import Config


async def test_executor():
    """测试 ExecutorAgent 的完整流程"""
    
    # 创建数据库连接池
    pool = AsyncConnectionPool(
        conninfo=Config.DB_URI,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE
    )
    
    try:
        # 初始化 ExecutorAgent
        print("=" * 60)
        print("初始化 ExecutorAgent")
        print("=" * 60)
        executor = ExecutorAgent(pool)
        print("✓ ExecutorAgent 初始化成功\n")
        
        # 测试查询
        test_queries = [
            "深度求索公司的基本情况",  # 应该调用 akshare
            "Tesla Inc. 的财务状况",  # 应该调用 sec_edgar
            "人工智能的最新研究进展",  # 不应该调用可选工具
            "Tesla 和比亚迪的对比分析",  # 应该调用 sec_edgar 和 akshare（测试循环）
        ]
        
        for i, query in enumerate(test_queries, 1):
            print("=" * 60)
            print(f"测试 {i}: {query}")
            print("=" * 60)
            
            try:
                result = await executor.invoke(
                    query=query,
                    thread_id=f"test_thread_{i}"
                )
                
                print(f"\n✓ 查询完成")
                print(f"  - 搜索结果数: {result['statistics']['searched']}")
                print(f"  - Rerank 后: {result['statistics']['after_rerank']}")
                print(f"  - 已下载: {result['statistics']['downloaded']}")
                print(f"  - 数据源分布: {result['papers_by_source']}")
                print(f"  - Top 论文数: {len(result['top_papers'])}")
                
                if result['top_papers']:
                    print(f"\n  Top 1 论文:")
                    top_paper = result['top_papers'][0]
                    print(f"    标题: {top_paper['title'][:50]}...")
                    print(f"    来源: {top_paper['source']}")
                    print(f"    分数: {top_paper['rerank_score']:.4f}")
                
                print()
            
            except Exception as e:
                print(f"✗ 查询失败: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        # 清理资源
        await executor._clean()
        print("✓ ExecutorAgent 资源清理完成")
    
    finally:
        await pool.close()
        print("✓ 数据库连接池关闭")


if __name__ == "__main__":
    asyncio.run(test_executor())

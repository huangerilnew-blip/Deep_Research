"""
可视化 ExecutorAgent 的 LangGraph 流程图
"""
import asyncio
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from agents import ExecutorAgent
from psycopg_pool import AsyncConnectionPool
from config import Config


async def visualize_executor_graph():
    """可视化 ExecutorAgent 的流程图"""
    
    # 创建数据库连接池
    pool = AsyncConnectionPool(
        conninfo=Config.DB_URI,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE
    )
    
    try:
        # 初始化 ExecutorAgent
        print("初始化 ExecutorAgent...")
        executor = ExecutorAgent(pool)
        
        # 获取图结构
        graph = executor.graph
        
        print("\n" + "=" * 60)
        print("ExecutorAgent 流程图结构")
        print("=" * 60)
        
        # 打印节点
        print("\n节点列表:")
        nodes = graph.nodes
        for i, node in enumerate(nodes, 1):
            print(f"  {i}. {node}")
        
        # 打印边
        print("\n边列表:")
        edges = graph.edges
        for i, (source, target) in enumerate(edges, 1):
            print(f"  {i}. {source} → {target}")
        
        # 打印条件边
        print("\n条件边:")
        print("  llm_decision → [_should_call_optional_tools]")
        print("    - 如果有 tool_calls → optional_tool_node")
        print("    - 如果没有 tool_calls → search")
        
        print("\n" + "=" * 60)
        print("完整流程")
        print("=" * 60)
        print("""
START
  ↓
llm_decision (Thought + Action)
  ↓
[条件边: _should_call_optional_tools]
  ↓                              ↓
optional_tool_node            search (完成决策)
(执行可选工具)                   ↓
  ↓                         clean_and_rerank
llm_decision                      ↓
(Observation + 再次决策)        download
  ↓                              ↓
[条件边: 继续循环?]            summarize
  ↓           ↓                  ↓
  YES        NO                 END
  ↑           ↓
  └─循环─→ search (合并所有结果)
              ↓
         clean_and_rerank
              ↓
         download
              ↓
         summarize
              ↓
            END

循环机制说明：
- LLM 可以多次决策是否调用可选工具（ReAct 模式）
- 每次调用可选工具后，会回到 llm_decision
- LLM 可以看到已有的可选工具结果（Observation），决定是否需要调用更多工具
- 当 LLM 认为已经收集了足够的信息时，会选择进入 search
- 一旦进入 search，就不再返回循环，直接执行后续节点
        """)
        
        # 打印工具信息
        print("=" * 60)
        print("工具配置")
        print("=" * 60)
        
        print("\n必需工具 (每个子问题都调用):")
        required_tools = executor._get_required_tools()
        for i, tool in enumerate(required_tools, 1):
            print(f"  {i}. {tool}")
        
        print("\n可选工具 (由 LLM 决定是否调用):")
        optional_tools = executor._get_optional_tools()
        for i, tool in enumerate(optional_tools, 1):
            print(f"  {i}. {tool.name}: {tool.description[:50]}...")
        
        print("\n下载工具 (延迟加载):")
        print("  - wikipedia_download")
        print("  - openalex_download")
        print("  - semantic_scholar_download")
        print("  - tavily_download")
        print("  - sec_edgar_download")
        print("  - akshare_download")
        
        # 清理资源
        await executor._clean()
        print("\n✓ 可视化完成")
    
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(visualize_executor_graph())

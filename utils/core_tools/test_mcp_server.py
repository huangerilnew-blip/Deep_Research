"""
测试 MCP Server 的功能
"""

import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient


async def test_mcp_server():
    """测试 MCP Server 的所有工具"""
    
    print("=" * 60)
    print("测试 Paper Search MCP Server")
    print("=" * 60)
    
    # 创建 MCP 客户端
    client = MultiServerMCPClient({
        "paper-search": {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "cwd": ".",  # 当前目录
            "transport": "stdio"
        }
    })
    
    try:
        # 获取所有工具
        print("\n【步骤 1】获取所有可用工具")
        print("-" * 60)
        tools = await client.get_tools()
        print(f"共找到 {len(tools)} 个工具：")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool.name}: {tool.description[:50]}...")
        
        # 测试 Wikipedia 搜索
        print("\n【步骤 2】测试 Wikipedia 搜索")
        print("-" * 60)
        wiki_tool = next((t for t in tools if t.name == "wikipedia_search"), None)
        if wiki_tool:
            result = await wiki_tool.ainvoke({
                "query": "人工智能",
                "limit": 2,
                "language": "zh"
            })
            print(f"Wikipedia 搜索结果：")
            papers = json.loads(result)
            for paper in papers[:1]:  # 只显示第一个结果
                print(f"  - 标题: {paper['title']}")
                print(f"  - 来源: {paper['source']}")
                print(f"  - URL: {paper['url']}")
                print(f"  - 摘要长度: {len(paper['abstract'])} 字符")
        
        # 测试 Tavily 搜索
        print("\n【步骤 3】测试 Tavily 搜索")
        print("-" * 60)
        tavily_tool = next((t for t in tools if t.name == "tavily_search"), None)
        if tavily_tool:
            result = await tavily_tool.ainvoke({
                "query": "latest AI research 2024",
                "limit": 2
            })
            print(f"Tavily 搜索结果：")
            papers = json.loads(result)
            for paper in papers[:1]:  # 只显示第一个结果
                print(f"  - 标题: {paper['title']}")
                print(f"  - 来源: {paper['source']}")
                print(f"  - URL: {paper['url']}")
                print(f"  - 评分: {paper['extra'].get('score', 'N/A')}")
                print(f"  - 摘要长度: {len(paper['abstract'])} 字符")
                if 'raw_content' in paper['extra']:
                    print(f"  - 完整内容长度: {len(paper['extra']['raw_content'])} 字符")
        
        # 测试 OpenAlex 搜索
        print("\n【步骤 4】测试 OpenAlex 搜索")
        print("-" * 60)
        openalex_tool = next((t for t in tools if t.name == "openalex_search"), None)
        if openalex_tool:
            result = await openalex_tool.ainvoke({
                "query": "machine learning",
                "limit": 2
            })
            print(f"OpenAlex 搜索结果：")
            papers = json.loads(result)
            for paper in papers[:1]:  # 只显示第一个结果
                print(f"  - 标题: {paper['title']}")
                print(f"  - 来源: {paper['source']}")
                print(f"  - 作者: {', '.join(paper['authors'][:3])}...")
                print(f"  - DOI: {paper['doi']}")
                print(f"  - 引用数: {paper['citations']}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

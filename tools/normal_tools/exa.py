import asyncio
import os
from typing import List

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from .paper import Paper

load_dotenv()


async def search(query: str, num_results: int = 3) -> List[Paper]:
    """
    执行Exa搜索并返回Paper列表

    Args:
        query: 搜索关键词
        num_results: 结果数量，默认3

    Returns:
        List[Paper]: 解析后的Paper对象列表
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY未配置，请在环境变量中设置EXA_API_KEY")

    # 使用 mcp-remote 适配器连接 Exa 托管 MCP 服务器
    # API Key 通过环境变量传递
    env = os.environ.copy()
    env["EXA_API_KEY"] = api_key

    mcp_client = MultiServerMCPClient(
        {
            "exa": {
                "command": "npx",
                "args": ["-y", "mcp-remote", "https://mcp.exa.ai/mcp"],
                "transport": "stdio",
                "env": env
            }
        }
    )

    print(f"正在使用Exa搜索: {query}...")

    try:
        # get_tools() 返回 List[BaseTool]
        tools = await mcp_client.get_tools()
        print(f"可用工具: {[t.name for t in tools]}")

        # 通过遍历找到 web_search_exa 工具
        web_search_exa = None
        for tool in tools:
            if tool.name == "web_search_exa":
                web_search_exa = tool
                break

        if not web_search_exa:
            print("未找到web_search_exa工具")
            raise ValueError("未找到web_search_exa工具")

        print(f"\n📤 发送搜索请求参数:")
        print(f"   query: {query}")
        print(f"   numResults: {num_results}")
        print(f"   contextMaxCharacters: {num_results * 10000}")

        result = await web_search_exa.ainvoke({
            "query": query,
            "numResults": num_results,
            "contextMaxCharacters": num_results * 10000  # 每个结果约10000字符
        })
        print("*********************************************************")
        print(f"结果的类型为{type(result)}")

        print(f"结果的长度为{len(result)}")
        print(f"*******************解析结果****************")
        index_0=result[0]
        print(repr(index_0))
        # 简单返回空列表，实际使用时需要解析结果
        return []

    except Exception as e:
        print(f"Exa搜索失败: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(search(query="大模型的能力边界"))

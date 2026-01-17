"""
MCP Server 入口点
允许使用 python -m core_tools 启动服务器
"""

from mcp_server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Paper Search MCP Server
整合所有论文搜索工具的 MCP Server (stdio 模式)
"""

import asyncio
import json
import sys
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 导入所有搜索器
from openalex import OpenAlexSearcher
from semantic_scholar import SemanticScholarSearcher
from wikipedia_searcher import WikipediaSearcher
from tavily import TavilySearcher
from sec_edgar import SECEdgarSearcher
from akshare_searcher import AkShareSearcher
from paper import Paper

# 创建 MCP Server 实例
app = Server("paper-search-server")

# 初始化所有搜索器
openalex_searcher = OpenAlexSearcher()
semantic_searcher = SemanticScholarSearcher()
wiki_searcher = WikipediaSearcher()
tavily_searcher = TavilySearcher()
sec_edgar_searcher = SECEdgarSearcher()
akshare_searcher = AkShareSearcher()


def paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """将 Paper 对象转换为字典"""
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "doi": paper.doi,
        "published_date": paper.published_date.isoformat() if paper.published_date else None,
        "pdf_url": paper.pdf_url,
        "url": paper.url,
        "source": paper.source,
        "categories": paper.categories,
        "keywords": paper.keywords,
        "citations": paper.citations,
        "references": paper.references,
        "extra": paper.extra
    }


@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出所有可用的工具"""
    return [
        # OpenAlex 工具
        Tool(
            name="openalex_search",
            description="搜索 OpenAlex 学术数据库。每个子问题都必须调用此工具。返回学术论文的基本信息和摘要。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="openalex_download",
            description="下载 OpenAlex 搜索结果的 PDF 文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "Paper 对象列表（JSON 格式）",
                        "items": {"type": "object"}
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存路径（可选）"
                    }
                },
                "required": ["papers"]
            }
        ),
        
        # Semantic Scholar 工具
        Tool(
            name="semantic_scholar_search",
            description="搜索 Semantic Scholar 学术数据库。每个子问题都必须调用此工具。返回学术论文的基本信息和摘要（可能为空）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="semantic_scholar_download",
            description="下载 Semantic Scholar 搜索结果的 PDF 文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "Paper 对象列表（JSON 格式）",
                        "items": {"type": "object"}
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存路径（可选）"
                    }
                },
                "required": ["papers"]
            }
        ),
        
        # Wikipedia 工具
        Tool(
            name="wikipedia_search",
            description="搜索 Wikipedia 百科。每个子问题都必须调用此工具。返回维基百科词条的完整内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 3
                    },
                    "language": {
                        "type": "string",
                        "description": "语言代码（en/zh）",
                        "default": "en"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="wikipedia_download",
            description="下载 Wikipedia 搜索结果为 Markdown 文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "Paper 对象列表（JSON 格式）",
                        "items": {"type": "object"}
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存路径（可选）"
                    }
                },
                "required": ["papers"]
            }
        ),
        
        # Tavily 工具
        Tool(
            name="tavily_search",
            description="使用 Tavily 进行网页搜索。每个子问题都必须调用此工具。返回高质量网页搜索结果（score > 0.8），包含摘要和完整原始内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="tavily_download",
            description="下载 Tavily 搜索结果为 Markdown 文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "Paper 对象列表（JSON 格式）",
                        "items": {"type": "object"}
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存路径（可选）"
                    }
                },
                "required": ["papers"]
            }
        ),
        
        # SEC EDGAR 工具
        Tool(
            name="sec_edgar_search",
            description="搜索美国证券交易委员会（SEC EDGAR）数据库。仅当需要查询在美国证券市场（如纽交所 NYSE、纳斯达克 NASDAQ）进行融资、挂牌或交易的实体时使用。返回公司证券信息。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "公司名称或股票代码"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="sec_edgar_download",
            description="下载 SEC EDGAR 搜索结果为 Markdown 文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "Paper 对象列表（JSON 格式）",
                        "items": {"type": "object"}
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存路径（可选）"
                    }
                },
                "required": ["papers"]
            }
        ),
        
        # AkShare 工具
        Tool(
            name="akshare_search",
            description="搜索中国股市数据（AkShare）。仅当需要了解在中国上市公司的基本情况时使用。返回公司基本情况作为摘要。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "公司名称或股票代码"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="akshare_download",
            description="下载 AkShare 搜索结果为 Markdown 文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "Paper 对象列表（JSON 格式）",
                        "items": {"type": "object"}
                    },
                    "save_path": {
                        "type": "string",
                        "description": "保存路径（可选）"
                    }
                },
                "required": ["papers"]
            }
        ),
    ]


# 工具处理函数映射
async def handle_search(searcher, arguments: Dict) -> List[Dict]:
    """通用搜索处理函数"""
    query = arguments["query"]
    limit = arguments.get("limit", 10)
    language = arguments.get("language")  # 仅 Wikipedia 使用
    
    # 只有 WikipediaSearcher 支持 language 参数
    if language and isinstance(searcher, WikipediaSearcher):
        papers = await searcher.search(query, limit, language)
    else:
        papers = await searcher.search(query, limit)
    
    return [paper_to_dict(p) for p in papers]


async def handle_download(searcher, arguments: Dict) -> List[Dict]:
    """通用下载处理函数"""
    papers_data = arguments["papers"]
    save_path = arguments.get("save_path")
    
    # 将字典转换回 Paper 对象
    papers = [Paper(**p) for p in papers_data]
    downloaded = await searcher.download(papers, save_path) if save_path else await searcher.download(papers)
    
    return [paper_to_dict(p) for p in downloaded]


# 工具映射字典：工具名 -> (处理函数, 搜索器实例)
TOOL_HANDLERS = {
    # OpenAlex
    "openalex_search": (handle_search, openalex_searcher),
    "openalex_download": (handle_download, openalex_searcher),
    
    # Semantic Scholar
    "semantic_scholar_search": (handle_search, semantic_searcher),
    "semantic_scholar_download": (handle_download, semantic_searcher),
    
    # Wikipedia
    "wikipedia_search": (handle_search, wiki_searcher),
    "wikipedia_download": (handle_download, wiki_searcher),
    
    # Tavily
    "tavily_search": (handle_search, tavily_searcher),
    "tavily_download": (handle_download, tavily_searcher),
    
    # SEC EDGAR
    "sec_edgar_search": (handle_search, sec_edgar_searcher),
    "sec_edgar_download": (handle_download, sec_edgar_searcher),
    
    # AkShare
    "akshare_search": (handle_search, akshare_searcher),
    "akshare_download": (handle_download, akshare_searcher),
}


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """处理工具调用 - 使用字典映射方式"""
    
    try:
        # 从映射字典中获取处理函数和搜索器
        if name in TOOL_HANDLERS:
            handler_func, searcher = TOOL_HANDLERS[name]
            results = await handler_func(searcher, arguments)
            return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        import traceback
        error_detail = traceback.format_exc()
        return [TextContent(type="text", text=f"{error_msg}\n\nDetails:\n{error_detail}")]


async def main():
    """主函数：启动 MCP Server (stdio 模式)"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

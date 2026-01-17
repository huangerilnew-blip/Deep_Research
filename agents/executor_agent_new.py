"""
ExecutorAgent 新版本 - 使用 LangGraph 标准模式处理可选工具
"""

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from config import Config
from llms import get_llm
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import logging, json, asyncio
from concurrent_log_handler import ConcurrentRotatingFileHandler

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.openai import OpenAIEmbedding

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []
handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=Config.MAX_BYTES,
    backupCount=Config.BACKUP_COUNT
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


class ExecutorState(TypedDict):
    executor_messages: Annotated[list[AnyMessage], add_messages]
    current_query: str  # 当前处理的子问题
    optional_search_results: List[Dict]  # 可选工具的搜索结果
    search_results: List[Dict]  # 所有搜索结果（必需工具 + 可选工具）
    reranked_results: List[Dict]  # Rerank后的结果
    downloaded_papers: List[Dict]  # 已下载的论文
    executor_result: Dict  # 最终结果摘要


class ExecutorAgent:
    """
    ExecutorAgent: 负责执行单个子问题的完整处理流程
    
    处理流程：
    START → llm_decision_node → [条件边] → optional_tool_node → search_node → clean_and_rerank → download → summarize → END
                                   ↓
                                search_node (直接跳过)
    """
    
    def __init__(self, pool: AsyncConnectionPool, modelname: ChatOpenAI = Config.LLM_EXECUTOR):
        self.chat_llm = get_llm(modelname)[0]
        # 只获取搜索工具
        self.search_tools: list[BaseTool] = self._get_search_tools()
        # 下载工具单独获取
        self.download_tools: list[BaseTool] = None  # 延迟加载
        self.memory = AsyncPostgresSaver(pool)
        
        # 初始化 LlamaIndex Reranker
        self.reranker = SentenceTransformerRerank(
            model=Config.RERANK_MODEL,
            top_n=Config.RERANK_TOP_N
        )
        
        self.graph = self._build_graph()
    
    def _get_search_tools(self) -> list[BaseTool]:
        """获取搜索工具（只包含 search 类工具）"""
        from tools import get_tools
        return asyncio.run(get_tools(tool_type="search"))
    
    async def _get_download_tools(self) -> list[BaseTool]:
        """获取下载工具（只包含 download 类工具）"""
        if self.download_tools is None:
            from tools import get_tools
            self.download_tools = await get_tools(tool_type="download")
        return self.download_tools
    
    def _get_required_tools(self) -> List[str]:
        """获取必需调用的工具列表"""
        return ["wikipedia_search", "openalex_search", "semantic_scholar_search", "tavily_search"]
    
    def _get_optional_tools(self) -> List[BaseTool]:
        """获取可选工具列表（由 LLM 决定是否调用）"""
        optional_tool_names = ["sec_edgar_search", "akshare_search"]
        return [t for t in self.search_tools if any(opt in t.name.lower() for opt in optional_tool_names)]
    
    async def _llm_decision_node(self, state: ExecutorState) -> Dict:
        """LLM 决策节点：让 LLM 决定是否需要调用可选工具"""
        query = state["current_query"]
        optional_tools = self._get_optional_tools()
        
        if not optional_tools:
            logger.info("没有可选工具，跳过 LLM 决策")
            return {"executor_messages": [AIMessage(content="不需要额外工具")]}
        
        logger.info(f"让 LLM 决定是否调用可选工具: {[t.name for t in optional_tools]}")
        
        try:
            # 绑定可选工具到 LLM
            llm_with_tools = self.chat_llm.bind_tools(optional_tools)
            
            # 构建提示
            prompt = f"""
            子问题: {query}
            
            请分析这个子问题，决定是否需要调用以下工具：
            - sec_edgar_search: 查询美国证券市场（NYSE、NASDAQ）的公司信息
            - akshare_search: 查询中国上市公司的基本情况
            
            如果需要，请调用相应的工具。如果不需要，直接回复"不需要额外工具"。
            """
            
            response = await llm_with_tools.ainvoke(prompt)
            
            logger.info(f"LLM 决策结果: 是否有工具调用={hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
            return {"executor_messages": [response]}
        
        except Exception as e:
            logger.error(f"LLM 决策出错: {e}")
            import traceback
            traceback.print_exc()
            return {"executor_messages": [AIMessage(content="决策出错，跳过可选工具")]}
    
    def _should_call_optional_tools(self, state: ExecutorState) -> str:
        """条件路由：判断是否需要调用可选工具"""
        last_message = state["executor_messages"][-1]
        
        if isinstance(last_message, AIMessage):
            # 检查是否有工具调用
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"LLM 决定调用 {len(last_message.tool_calls)} 个可选工具")
                return "optional_tool_node"
            else:
                logger.info("LLM 决定不调用可选工具")
                return "search_node"
        else:
            logger.warning(f"意外的消息类型: {type(last_message)}")
            return "search_node"
    
    async def _optional_tool_node(self, state: ExecutorState) -> Dict:
        """可选工具节点：执行 LLM 决定调用的可选工具"""
        last_message = state["executor_messages"][-1]
        optional_search_results = []
        
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls'):
            logger.warning("没有工具调用信息")
            return {"optional_search_results": []}
        
        optional_tools = self._get_optional_tools()
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            
            logger.info(f"执行可选工具: {tool_name}, 参数: {tool_args}")
            
            try:
                # 查找工具
                tool = next((t for t in optional_tools if t.name == tool_name), None)
                if tool:
                    result = await tool.ainvoke(tool_args)
                    
                    if result:
                        # MCP 工具返回的是 JSON 字符串，需要解析
                        if isinstance(result, str):
                            try:
                                papers = json.loads(result)
                            except json.JSONDecodeError as e:
                                logger.error(f"解析 JSON 失败: {e}")
                                continue
                        elif isinstance(result, list):
                            papers = result
                        else:
                            logger.warning(f"未知的结果类型: {type(result)}")
                            continue
                        
                        if isinstance(papers, list):
                            optional_search_results.extend(papers)
                            logger.info(f"工具 {tool_name} 返回 {len(papers)} 条结果")
                else:
                    logger.warning(f"未找到工具: {tool_name}")
            
            except Exception as e:
                logger.error(f"执行可选工具 {tool_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"可选工具搜索完成，共获得 {len(optional_search_results)} 条结果")
        return {"optional_search_results": optional_search_results}
    
    async def _search_node(self, state: ExecutorState) -> Dict:
        """搜索节点：并行调用必需的搜索工具"""
        query = state["current_query"]
        required_tools = self._get_required_tools()
        
        search_results = []
        
        # 并行调用必需工具
        logger.info(f"开始并行调用必需工具: {required_tools}")
        for tool_name in required_tools:
            try:
                tool = next((t for t in self.search_tools if tool_name in t.name.lower()), None)
                if tool:
                    logger.info(f"调用工具 {tool.name} 搜索: {query}")
                    result = await tool.ainvoke({"query": query})
                    
                    if result:
                        # MCP 工具返回的是 JSON 字符串，需要解析
                        if isinstance(result, str):
                            try:
                                papers = json.loads(result)
                            except json.JSONDecodeError as e:
                                logger.error(f"解析 JSON 失败: {e}, 原始结果: {result[:200]}")
                                continue
                        elif isinstance(result, list):
                            papers = result
                        else:
                            logger.warning(f"未知的结果类型: {type(result)}")
                            continue
                        
                        if isinstance(papers, list):
                            search_results.extend(papers)
                            logger.info(f"工具 {tool.name} 返回 {len(papers)} 条结果")
                else:
                    logger.warning(f"未找到工具: {tool_name}")
            except Exception as e:
                logger.error(f"调用工具 {tool_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 合并可选工具的搜索结果
        optional_results = state.get("optional_search_results", [])
        if optional_results:
            logger.info(f"合并 {len(optional_results)} 条可选工具搜索结果")
            search_results.extend(optional_results)
        
        logger.info(f"搜索完成，共获得 {len(search_results)} 条结果")
        return {"search_results": search_results}
    
    def _extract_abstract(self, paper: Dict) -> str:
        """根据数据源提取完整摘要"""
        source = paper.get("source", "").lower()
        
        if source == "sec_edgar":
            extra = paper.get("extra", {})
            parts = []
            if extra.get("company_info"):
                parts.append(f"公司基本信息:\n{extra['company_info']}")
            if extra.get("risk_factors"):
                parts.append(f"\n\n风险情况:\n{extra['risk_factors']}")
            if extra.get("company_assessment"):
                parts.append(f"\n\n公司评估:\n{extra['company_assessment']}")
            return "\n".join(parts) if parts else paper.get("abstract", "")
        else:
            return paper.get("abstract", "")
    
    def _should_clean(self, paper: Dict) -> bool:
        """判断是否需要清洗（用于 Rerank）"""
        source = paper.get("source", "").lower()
        abstract = paper.get("abstract", "")
        
        if source == "openalex":
            return True
        elif source == "semantic_scholar":
            return bool(abstract)
        elif source == "tavily":
            return True
        else:
            return False
    
    async def _clean_and_rerank_node(self, state: ExecutorState) -> Dict:
        """清洗和Rerank节点（使用 LlamaIndex）"""
        query = state["current_query"]
        search_results = state["search_results"]
        
        if not search_results:
            logger.warning("没有搜索结果需要rerank")
            return {"reranked_results": []}
        
        # 准备需要 rerank 的文档
        papers_to_rerank = []
        paper_indices = []
        
        for i, paper in enumerate(search_results):
            if self._should_clean(paper):
                abstract = self._extract_abstract(paper)
                if abstract:
                    papers_to_rerank.append(abstract)
                    paper_indices.append(i)
        
        logger.info(f"准备 rerank {len(papers_to_rerank)} 篇文档")
        
        if not papers_to_rerank:
            logger.warning("没有需要 rerank 的文档")
            return {"reranked_results": search_results}
        
        try:
            # 使用 LlamaIndex 创建文档
            documents = [
                Document(text=abstract, metadata={"index": idx})
                for abstract, idx in zip(papers_to_rerank, paper_indices)
            ]
            
            # 创建临时索引
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=OpenAIEmbedding()
            )
            
            # 使用 reranker 进行查询
            query_engine = index.as_query_engine(
                similarity_top_k=len(documents),
                node_postprocessors=[self.reranker]
            )
            
            response = query_engine.query(query)
            
            # 提取 rerank 后的结果
            reranked_results = []
            for node in response.source_nodes:
                original_idx = node.metadata.get("index")
                score = node.score if hasattr(node, 'score') else 0.0
                
                if score >= Config.RERANK_THRESHOLD:
                    paper = search_results[original_idx].copy()
                    paper["rerank_score"] = score
                    reranked_results.append(paper)
            
            reranked_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            logger.info(f"Rerank 完成，保留 {len(reranked_results)} 篇相关文档")
            
            # 添加未参与 rerank 的文档
            for i, paper in enumerate(search_results):
                if i not in paper_indices:
                    paper_copy = paper.copy()
                    paper_copy["rerank_score"] = 0.0
                    reranked_results.append(paper_copy)
            
            return {"reranked_results": reranked_results}
        
        except Exception as e:
            logger.error(f"Rerank 过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {"reranked_results": search_results}
    
    async def _download_node(self, state: ExecutorState) -> Dict:
        """下载节点：根据 paper.source 按检索器类型进行下载"""
        reranked_results = state["reranked_results"]
        
        if not reranked_results:
            logger.warning("没有需要下载的文档")
            return {"downloaded_papers": []}
        
        download_tools = await self._get_download_tools()
        downloaded_papers = []
        
        # 按 source 分组
        papers_by_source = {}
        for paper in reranked_results:
            source = paper.get("source", "unknown")
            if source not in papers_by_source:
                papers_by_source[source] = []
            papers_by_source[source].append(paper)
        
        # 按 source 调用相应的下载工具
        for source, papers in papers_by_source.items():
            try:
                logger.info(f"开始下载 {source} 的 {len(papers)} 篇文档到 {Config.DOC_SAVE_PATH}")
                
                download_tool_name = f"{source}_download"
                download_tool = next((t for t in download_tools if download_tool_name in t.name.lower()), None)
                
                if download_tool:
                    result = await download_tool.ainvoke({
                        "papers": papers,
                        "save_path": Config.DOC_SAVE_PATH
                    })
                    
                    if result:
                        if isinstance(result, str):
                            try:
                                downloaded = json.loads(result)
                            except json.JSONDecodeError as e:
                                logger.error(f"解析下载结果 JSON 失败: {e}")
                                continue
                        elif isinstance(result, list):
                            downloaded = result
                        else:
                            logger.warning(f"未知的下载结果类型: {type(result)}")
                            continue
                        
                        if isinstance(downloaded, list):
                            downloaded_papers.extend(downloaded)
                            logger.info(f"成功下载 {len(downloaded)} 篇 {source} 文档")
                else:
                    logger.warning(f"未找到 {source} 的下载工具")
            
            except Exception as e:
                logger.error(f"下载 {source} 文档时出错: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"下载完成，共下载 {len(downloaded_papers)} 篇文档")
        return {"downloaded_papers": downloaded_papers}
    
    async def _summarize_node(self, state: ExecutorState) -> Dict:
        """摘要节点：生成完整摘要"""
        downloaded_papers = state["downloaded_papers"]
        query = state["current_query"]
        
        summary = {
            "query": query,
            "total_papers": len(downloaded_papers),
            "papers_by_source": {},
            "top_papers": [],
            "statistics": {
                "searched": len(state.get("search_results", [])),
                "after_rerank": len(state.get("reranked_results", [])),
                "downloaded": len(downloaded_papers)
            }
        }
        
        # 按 source 统计
        for paper in downloaded_papers:
            source = paper.get("source", "unknown")
            if source not in summary["papers_by_source"]:
                summary["papers_by_source"][source] = 0
            summary["papers_by_source"][source] += 1
        
        # 提取 top papers
        sorted_papers = sorted(
            downloaded_papers,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True
        )[:5]
        
        for paper in sorted_papers:
            full_abstract = self._extract_abstract(paper)
            
            summary["top_papers"].append({
                "title": paper.get("title", ""),
                "source": paper.get("source", ""),
                "rerank_score": paper.get("rerank_score", 0),
                "url": paper.get("url", ""),
                "saved_path": paper.get("extra", {}).get("saved_path", ""),
                "abstract": full_abstract,
                "authors": paper.get("authors", []),
                "published_date": paper.get("published_date", ""),
                "doi": paper.get("doi", "")
            })
        
        logger.info(f"生成完整摘要完成: {summary['statistics']}")
        return {"executor_result": summary}
    
    def _build_graph(self):
        """构建 ExecutorAgent 的处理流程图"""
        builder = StateGraph(ExecutorState)
        
        # 添加节点
        builder.add_node("llm_decision", self._llm_decision_node)
        builder.add_node("optional_tool_node", self._optional_tool_node)
        builder.add_node("search", self._search_node)
        builder.add_node("clean_and_rerank", self._clean_and_rerank_node)
        builder.add_node("download", self._download_node)
        builder.add_node("summarize", self._summarize_node)
        
        # 添加边
        builder.add_edge(START, "llm_decision")
        builder.add_conditional_edges(
            "llm_decision",
            self._should_call_optional_tools,
            {
                "optional_tool_node": "optional_tool_node",
                "search_node": "search"
            }
        )
        builder.add_edge("optional_tool_node", "search")
        builder.add_edge("search", "clean_and_rerank")
        builder.add_edge("clean_and_rerank", "download")
        builder.add_edge("download", "summarize")
        builder.add_edge("summarize", END)
        
        graph = builder.compile(checkpointer=self.memory)
        logger.info("完成 executor_graph 的初始化构造")
        return graph
    
    async def invoke(self, query: str, thread_id: str) -> Dict:
        """执行单个子问题的完整处理流程"""
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "executor_messages": [],
            "current_query": query,
            "optional_search_results": [],
            "search_results": [],
            "reranked_results": [],
            "downloaded_papers": [],
            "executor_result": {}
        }
        
        try:
            result = await self.graph.ainvoke(initial_state, config)
            logger.info(f"executor 完成子问题 '{query}' 的处理")
            return result["executor_result"]
        except Exception as e:
            logger.error(f"executor 处理子问题 '{query}' 时出错: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    async def _clean(self):
        """清理资源"""
        if self.memory:
            try:
                await self.memory.aclose()
                logger.info("对实例化的 ExecutorAgent,完成对短期记忆连接池的断开处理")
            except Exception as e:
                logger.info(f"尝试对实例化的 ExecutorAgent 与短期记忆连接池断开，出现错误：{e}")

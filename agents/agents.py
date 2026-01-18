import asyncio

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from core.config import Config
from core.llms import get_llm
from core.reranker import BGEReranker
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import add_messages, StateGraph, START, END, state
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage,ToolMessage
from langchain_core.messages import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import logging, json, asyncio
from concurrent_log_handler import ConcurrentRotatingFileHandler

# 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    Config.LOG_FILE,
    maxBytes = Config.MAX_BYTES,
    backupCount = Config.BACKUP_COUNT
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

class PlannerState(TypedDict):
    planner_messages: Annotated[list[AnyMessage], add_messages]
    planner_result: AIMessage
    epoch: int

class ExecutorState(TypedDict):
    executor_messages: Annotated[list[AnyMessage], add_messages]
    current_query: str  # 当前处理的子问题
    optional_search_results:Annotated[list[ToolMessage], add_messages]  # 可选工具的搜索结果
    search_results: List[Dict]  # 所有搜索结果（必需工具 + 可选工具）
    reranked_results: List[Dict]  # Rerank后的结果
    downloaded_papers: List[Dict]  # 已下载的论文
    executor_result: Dict  # 最终结果摘要

class PlannerAgent:
    def __init__(self, pool: AsyncConnectionPool, modelname: ChatOpenAI = Config.LLM_PLANNER):
        self.chat_llm = get_llm(modelname)[0]
        self.memory = AsyncPostgresSaver(pool)
        self.graph = self._build_graph()

    def _json_node(self, state: PlannerState, planner_epoch=Config.PLANNER_EPOCH) -> dict:
        prompt = """请根据用户提供的主要问题{query}，从多个维度拆解成子问题列表。在回答中，你需要遵循以下要求：

        1. **拆解主问题：** 根据问题的多个层次和方面进行拆解，确保每个子问题具体且明确。
        2. **生成子问题列表：** 将拆解后的子问题组织成一个简单的列表，每个子问题是一个字符串。
        3. **结构化输出：** 使用以下JSON格式来输出子问题列表：

        ```
        {{
          "tasks": [
            "子问题1",
            "子问题2",
            "子问题3"
          ]
        }}
        ```

        **详细规则：**

        - "tasks"是一个字符串数组，每个元素是一个具体的子问题。
        - json的样例中虽然只放了三个子问题，但是你一定不能受到三个子问题数量的限制。而是要思考主问题，去拆解分析，打破欧式距离限制，帮助用户深层次的去了解问题。
        - 子问题之间不需要考虑依赖关系，只需要列出所有相关的子问题即可。
        """
        query = state["planner_messages"][0]
        template = ChatPromptTemplate.from_messages([
            {"role": "system", "content": prompt}],
        )
        try:
            chain = {"query": RunnablePassthrough()} | template | self.chat_llm
            if state["epoch"] < planner_epoch:
                result = chain.invoke(query)
                state["epoch"] += 1
                return {"planner_messages": [result]}
        except Exception as e:
            logger.error(f"planner_agent_node1 分析用户的问题时，出现错误:{e}")
            raise e

    def _condition_router(self, state: PlannerState, planner_epoch=Config.PLANNER_EPOCH):
        result = state["planner_messages"][-1]
        if isinstance(result, AIMessage):
            if state["epoch"] < planner_epoch:
                try:
                    _ = json.loads(result.content)
                    state["planner_result"] = result
                    return "END"
                except Exception:
                    return "json_node"
            state["planner_result"] = AIMessage(content=str({"tasks": "error"}))
            logger.warning(f"planner_agent 达到最大迭代次数{planner_epoch}，仍未能生成有效的json结构，结束planner流程")
            return "END"
        logger.error(f"planner_agent 条件路由器收到非AIMessage类型的消息，类型为:{type(result)}，内容为:{result.content}")
        raise TypeError(f"planner_agent 条件路由器收到非AIMessage类型的消息，类型为:{type(result)}")
    
    def _build_graph(self):
        builder = StateGraph(PlannerState)
        builder.add_node("json_node", self._json_node)
        builder.add_edge(START, "json_node")
        builder.add_conditional_edges("json_node", self._condition_router, {"END": END, "json_node": "json_node"})
        builder.compile(checkpointer=self.memory)
        logger.info(f"完成planner_graph的初始化构造")
        return builder

    async def invoke(self, thread_id: str):
        query = state["planner_messages"][0]
        config = {"configurable": {"thread_id": thread_id}}
        try:
            response = await self.chat_llm.ainvoke(query, config)
            logger.info(f"planner_chatmodel对于用户的{query} 返回:{response}")
            return response
        except Exception as e:
            logger.error(f"planner_chatmodel对于用户的{query} 出现错误：{e}")
            raise e

    async def _clean(self):
        if self.memory:
            try:
                await self.memory.aclose()
                logger.info("对实例化的PlannerAgent,完成对短期记忆连接池的断开处理")
            except Exception as e:
                logger.info(f"尝试对实例化的PlannerAgent与短期记忆连接池断开，出现错误：{e}")


class ExecutorAgent:
    """
    ExecutorAgent: 负责执行单个子问题的完整处理流程
    
    处理流程：
    START → llm_decision_node → [条件边] → optional_tool_node → llm_decision_node (循环) → search_node → clean_and_rerank → download → summarize → END
                                   ↓
                                search_node (直接跳过循环)
    
    LLM 可以多次决策是否调用可选工具，形成循环，直到 LLM 认为已经收集了足够的信息。
    """
    
    def __init__(self, pool: AsyncConnectionPool, modelname: ChatOpenAI = Config.LLM_EXECUTOR):
        self.chat_llm = get_llm(modelname)[0]
        # 只获取搜索工具
        self.search_tools: list[BaseTool] = self._get_search_tools()
        # 下载工具单独获取
        self.download_tools: list[BaseTool] = None  # 延迟加载
        self.memory = AsyncPostgresSaver(pool)
        
        # 初始化 BGE Reranker（使用 TEI 部署）
        self.reranker = BGEReranker(
            batch_size=Config.RERANK_BATCH_SIZE,
            max_concurrent=Config.RERANK_MAX_CONCURRENT
        )
        
        self.graph = self._build_graph()
    
    def _get_search_tools(self) -> list[BaseTool]:
        """获取搜索工具（只包含 search 类工具）"""
        from core.tools import get_tools
        return asyncio.run(get_tools(tool_type="search"))
    
    async def _get_download_tools(self) -> list[BaseTool]:
        """获取下载工具（只包含 download 类工具）"""
        if self.download_tools is None:
            from core.tools import get_tools
            self.download_tools = await get_tools(tool_type="download")
        return self.download_tools
    
    def _get_required_tools(self) -> List[str]:
        """获取必需调用的工具列表"""
        return ["wikipedia_search", "openalex_search", "semantic_scholar_search", "tavily_search"]
    
    def _get_optional_tools(self) -> List[BaseTool]:
        """获取可选工具列表（由 LLM 决定是否调用）"""
        optional_tool_names = ["sec_edgar_search", "akshare_search"]
        return [t for t in self.search_tools if any(opt in t.name.lower() for opt in optional_tool_names)]
    
    async def _invoke_single_tool(self, tool: BaseTool, query: str)->list[dict]:
        """单个工具调用的异步函数
        
        Args:
            tool: 已验证存在的工具对象
            query: 搜索查询
        """
        logger.info(f"调用工具 {tool.name} 搜索: {query}")
        
        try:
            result = await tool.ainvoke({"query": query})
        except Exception as e:
            logger.error(f"工具 {tool.name} 调用失败: {e}")
            raise RuntimeError(f"工具 {tool.name} 调用失败: {e}")
        
        # MultiServerMCPClient 返回的是 str 类型（从 TextContent.text 提取）
        if not result or not isinstance(result, str):
            logger.error(f"工具 {tool.name} 返回结果异常: type={type(result)}")
            raise TypeError(f"工具 {tool.name} 返回结果类型错误")
        
        # 根据 mcp_server.py 的三种返回情况判断：
        # 1. 正常返回：JSON 格式结果
        # 2. 工具名不存在："Unknown tool: {name}"
        # 3. 意外错误："Error executing {name}: ..."
        
        if result.startswith("Unknown tool:"):
            logger.error(f"MCP 工具名不存在: {result}")
            raise NameError(f"MCP 工具名不存在: {tool.name}")
        
        if result.startswith("Error executing"):
            logger.error(f"MCP 工具执行错误: {result[:200]}")
            raise RuntimeError(f"MCP 工具执行错误: {tool.name}")
        
        # 正常情况：解析 JSON
        try:
            papers = json.loads(result)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}, 原始结果: {result[:200]}")
            raise ValueError(f"工具 {tool.name} 返回的 JSON 格式错误")
        
        if not isinstance(papers, list):
            logger.error(f"工具 {tool.name} 返回的数据不是列表: {type(papers)}")
            raise TypeError(f"工具 {tool.name} 返回数据格式错误")
        
        logger.info(f"工具 {tool.name} 返回 {len(papers)} 条结果")
        return papers
    
    def _crop_observation(self, tool_messages: List[ToolMessage]) -> List[ToolMessage]:
        """
        裁剪工具执行结果为观察信息（Observation）
        裁剪 ToolMessage 内容，只保留关键信息用于 LLM 决策
        目的：减少 Token 消耗，加快 LLM 判断速度
        
        Args:
            tool_messages: 原始的 ToolMessage 列表

        Returns:
            格式化后的 ToolMessage 列表，内容精简但包含关键信息
        """
        if not tool_messages:
            logger.error(f"ExecutorAgent的optional_search_results状态为空，无法进行格式化处理")
            return []
        
        formatted_messages = []
        
        for tool_msg in tool_messages:
            content = tool_msg.content
            
            try:
                # 解析 JSON 内容
                if isinstance(content, str):
                    papers = json.loads(content)
                elif isinstance(content, list):
                    papers = content
                else:
                    formatted_messages.append(tool_msg)
                    continue
                
                # 提取关键信息摘要
                observations = []
                for paper in papers:
                    source = paper.get("source", "unknown")
                    title = paper.get("title", "无标题")
                    
                    if source == "sec_edgar":
                        # SEC EDGAR: 显示公司名和简短预览
                        company_name = paper.get("extra", {}).get("company_name", title)
                        abstract = paper.get("abstract", "")
                        preview = abstract[:150] + "..." if len(abstract) > 150 else abstract
                        observations.append(
                            f"✓ {company_name}\n"
                            f"  来源: SEC EDGAR\n"
                            f"  预览: {preview}"
                        )
                    elif source == "akshare":
                        # AkShare: 显示公司名和简短预览
                        abstract = paper.get("abstract", "")
                        preview = abstract[:150] + "..." if len(abstract) > 150 else abstract
                        observations.append(
                            f"✓ {title}\n"
                            f"  来源: AkShare\n"
                            f"  预览: {preview}"
                        )
                    else:
                        # 其他来源：只显示标题和来源
                        observations.append(f"✓ {title} (来源: {source})")
                
                # 创建精简的 ToolMessage
                if observations:
                    formatted_content = "\n\n".join(observations)
                    formatted_msg = ToolMessage(
                        content=formatted_content,
                        tool_call_id=tool_msg.tool_call_id,
                        name=tool_msg.name
                    )
                    formatted_messages.append(formatted_msg)
                else:
                    formatted_messages.append(tool_msg)
            
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"解析 ToolMessage 内容失败: {e}")
                formatted_messages.append(tool_msg)
                continue
        
        return formatted_messages
    async def _llm_decision_node(self, state: ExecutorState) -> Dict:
        """
        LLM 决策节点：按照 ReAct 模式让 LLM 进行推理和决策
        利用 LangGraph 的消息流机制，让 LLM 看到完整的对话历史自主决策
        """
        # 从 executor_messages 的第一条消息获取 query
        try:
            query=state["executor_messages"].content
        except Exception as e:
            logger.error(f"ExctorAgent-llm_decision_node 状态错误{e}，无法正确提取用户查询")
            raise e
        # 缓存到 current_query 中，方便后续节点使用
        if not state.get("current_query"):
            state["current_query"] = query
        
        optional_tools = self._get_optional_tools()
        optional_search_results = state.get("optional_search_results", [])
        
        if not optional_tools:
            logger.info("没有可选工具，跳过 LLM 决策")
            return {"executor_messages": [AIMessage(content="Thought: 不需要额外的专业工具，可以直接进行搜索。")]}
        
        logger.info(f"让 LLM 决定是否调用可选工具: {[t.name for t in optional_tools]}")
        
        try:
            llm_with_tools = self.chat_llm.bind_tools(optional_tools)
            
            # 构建消息列表：利用 LangGraph 的消息流
            if optional_search_results:
                # 有工具结果，使用裁剪后的 ToolMessage
                formatted_messages = self._crop_observation(optional_search_results)
                
                # 构建完整消息流：原始问题 + 之前的 AI 响应 + 工具结果
                messages =[]+ state["executor_messages"][:2] + formatted_messages
            else:
                # 第一次决策，只有原始问题
                messages = state["executor_messages"]
            
            # LLM 会看到完整的对话历史，自主决策
            response = await llm_with_tools.ainvoke(messages)
            
            # 记录决策结果
            has_tool_calls = hasattr(response, 'tool_calls') and bool(response.tool_calls)
            logger.info(f"LLM 决策结果: 是否调用工具={has_tool_calls}")
            if hasattr(response, 'content') and response.content:
                logger.info(f"LLM 思考: {response.content[:150]}...")
            
            return {"executor_messages": [response]}
        
        except Exception as e:
            logger.error(f"ExctorAgent-llm_decision_node 状态错误{e}")
            return {"executor_messages": [AIMessage(content="Thought: 决策过程出错，跳过可选工具，直接进行搜索。")]}
    
    def _should_call_optional_tools(self, state: ExecutorState) -> str:
        """条件路由：判断是否需要调用可选工具"""
        last_message = state["executor_messages"][-1]
        
        if isinstance(last_message, AIMessage):
            # 检查是否有工具调用
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"LLM 决定调用 {len(last_message.tool_calls)} 个可选工具")
                return "optional_tool_node"
            else:
                # LLM 决定不再调用可选工具，进入搜索阶段
                logger.info("LLM 决定不调用可选工具，进入搜索阶段")
                return "search"
        else:
            logger.warning(f"意外的消息类型: {type(last_message)}")
            return "search"
    
    async def _optional_tool_node(self, state: ExecutorState) -> Dict:
        """
        可选工具节点：使用 LangGraph 的 ToolNode 执行工具调用
        按照 ReAct 模式，执行 Action 并返回 Observation (ToolMessage)
        """
        # 检查最后一条消息是否有 tool_calls
        last_message = state["executor_messages"][-1] if state["executor_messages"] else None
        
        if not last_message or not isinstance(last_message, AIMessage):
            logger.warning("最后一条消息不是 AIMessage")
            return {"optional_search_results": []}
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("AIMessage 中没有 tool_calls")
            return {"optional_search_results": []}
        
        optional_tools = self._get_optional_tools()
        tool_node = ToolNode(optional_tools)
        
        try:
            logger.info(f"准备执行 {len(last_message.tool_calls)} 个可选工具调用")
            
            # ToolNode.ainvoke() 接收 AIMessage，返回 ToolMessage 列表
            tool_messages = await tool_node.ainvoke(last_message)
            
            # 确保返回的是列表
            if not isinstance(tool_messages, list):
                tool_messages = [tool_messages]
            
            # 记录日志
            logger.info(f"可选工具执行完成，返回 {len(tool_messages)} 条 ToolMessage")
            for tool_msg in tool_messages:
                try:
                    content = json.loads(tool_msg.content) if isinstance(tool_msg.content, str) else tool_msg.content
                    if isinstance(content, list):
                        logger.info(f"  - 工具 {tool_msg.name} 返回 {len(content)} 条结果")
                except:
                    logger.info(f"  - 工具 {tool_msg.name} 执行完成")
            
            return {"optional_search_results": [tool_messages],"executor_messages":[tool_messages]}
        
        except Exception as e:
            logger.error(f"执行可选工具时出错: {e}")
            raise e
    
    async def _search_node(self, state: ExecutorState) -> Dict:
        """搜索节点：并行调用必需的搜索工具"""
        query = state["current_query"]
        required_tool_names = self._get_required_tools()
        
        search_results = []
        
        # 并行调用必需工具
        logger.info(f"开始并行调用必需工具: {required_tool_names}")
        
        # 提前验证并获取工具对象
        tools_to_invoke = []
        for tool_name in required_tool_names:
            tool = next((t for t in self.search_tools if tool_name in t.name.lower()), None)
            if not tool:
                logger.error(f"未找到必需工具: {tool_name}")
                raise ValueError(f"未找到必需工具: {tool_name}")
            tools_to_invoke.append(tool)
        
        # 使用 asyncio.gather 并发调用所有工具
        tasks = [self._invoke_single_tool(tool, query) for tool in tools_to_invoke]
        #[Exception，[paper]] 将正常结果进行判断
        results = await asyncio.gather(*tasks, return_exceptions=True) #有异常也不终端
        
        # 处理结果，遇到异常记录，继续执行 
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"工具 {tools_to_invoke[i].name} 执行失败: {result}")
                # 不中断流程，继续处理其他工具的结果
            elif isinstance(result, list):# list[paper]
                search_results.extend(result)
        
        logger.info(f"搜索完成，共获得 {len(search_results)} 条结果")
        return {"search_results": search_results} #先不将工具检索的list[paper] 保存到"executor_messages"中
    
    def _extract_abstract(self, paper: Dict) -> str:
        """根据数据源提取摘要
        
        只提取配置中指定的数据源的摘要用于 rerank
        """
        source = paper.get("source", "").lower()
        
        # 只处理配置中指定的数据源
        if source in Config.PAPER_CLEAN:
            return paper.get("abstract", "")
        
        return ""
    
    def _should_clean(self, paper: Dict) -> bool:
        """判断是否需要清洗（用于Rerank）
        只对配置中指定的数据源进行清洗，且必须有摘要
        """
        source = paper.get("source", "").lower()
        abstract = paper.get("abstract", "")
        
        # 只清洗配置中指定的数据源，且必须有摘要
        if source in Config.PAPER_CLEAN:
            return bool(abstract)
        
        return False
    
    async def _clean_node(self, state: ExecutorState) -> Dict:
        """清洗和Rerank节点（使用 LlamaIndex）"""
        try:
            query = state["current_query"]
            search_results = state["search_results"]
        except Exception as e:
            logger.error(f"ExecutorAgent clean_node 获取状态失败: {e}")
            raise e

        if not search_results:
            logger.warning("没有搜索结果需要rerank")
            return {"reranked_results": []}
        
        # 准备需要 rerank 的文档（只处理 openalex 和 semantic_scholar）
        papers_to_rerank = [] #需要清洗的paper
        paper_indices = [] #记录需要清洗的索引
        
        for i, paper in enumerate(search_results):
            if self._should_clean(paper):
                abstract = self._extract_abstract(paper)
                if abstract:
                    papers_to_rerank.append(abstract)
                    paper_indices.append(i)
        
        logger.info(f"准备 rerank {len(papers_to_rerank)} 篇文档（来源：{', '.join(Config.PAPER_CLEAN)}）")
        
        if not papers_to_rerank:
            logger.warning("没有需要 rerank 的文档")
            return {"reranked_results": search_results}
        
        try:
            # 使用 TEI 部署的 bge-reranker 进行重排序[{"index":xx,"score":xx}] 这里的index 是list[paper]中的index
            rerank_results = await self._rerank_with_bge(query, papers_to_rerank, paper_indices)
            
            # 提取 rerank 后的结果
            reranked_results = []
            for item in rerank_results:
                original_idx = item["index"]
                score = item["score"]
                
                if score >= Config.RERANK_THRESHOLD:
                    paper = search_results[original_idx].copy()
                    paper["rerank_score"] = score
                    reranked_results.append(paper)
            
            reranked_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            # 只保留 top_n 个结果
            reranked_results = reranked_results[:Config.RERANK_TOP_N]
            
            logger.info(f"Rerank 完成，保留 {len(reranked_results)} 篇相关文档")
            
            # 添加未参与 rerank 的文档（其他来源的文档）
            for i, paper in enumerate(search_results):
                if i not in paper_indices:
                    paper_copy = paper.copy()
                    paper_copy["rerank_score"] = 0.0
                    reranked_results.append(paper_copy)
            
            return {"reranked_results": reranked_results} #list[dict(paper)]
        
        except Exception as e:
            logger.error(f"Rerank 过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {"reranked_results": search_results}
    
    async def _rerank_with_bge(self, query: str, documents: List[str], indices: List[int]) -> List[Dict]:
        """使用 BGEReranker 进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            indices: 文档在原始列表中的索引
            
        Returns:
            重排序结果列表，每个元素包含 {"index": int, "score": float}
        """
        try:
            # 调用 BGEReranker 的异步 rerank 方法
            # 返回结果是 [{"index": int, "score": float}] 降序
            rerank_results = await self.reranker.rerank_async(query, documents)
            
            # 将 rerank 返回的索引映射回原始索引
            mapped_results = []
            for item in rerank_results:
                doc_idx = item["index"]
                if doc_idx < len(indices):
                    mapped_results.append({
                        "index": indices[doc_idx],
                        "score": item["score"]
                    })
            
            return mapped_results
            
        except Exception as e:
            logger.error(f"调用 BGEReranker 失败: {e}")
            raise RuntimeError(f"Rerank papers 失败: {e}")
    
    async def _download_node(self, state: ExecutorState) -> Dict:
        """下载节点：下载 reranked_results 和 optional_search_results 中的文档"""
        reranked_results = state.get("reranked_results", [])
        optional_search_results = state.get("optional_search_results", [])
        
        # 合并所有需要下载的文档
        all_papers = []
        
        # 添加 reranked_results
        if reranked_results:
            all_papers.extend(reranked_results)
            logger.info(f"从 reranked_results 获取 {len(reranked_results)} 篇文档")
        
        # 从 optional_search_results (List[ToolMessage]) 中提取文档
        if optional_search_results:
            for tool_msg in optional_search_results:
                try:
                    # tool_msg 是 ToolMessage 对象
                    content = tool_msg.content
                    
                    if isinstance(content, str):
                        papers = json.loads(content)
                    elif isinstance(content, list):
                        papers = content
                    else:
                        continue
                    
                    if isinstance(papers, list):
                        all_papers.extend(papers)
                        logger.info(f"从 optional_search_results 的工具 {tool_msg.name} 获取 {len(papers)} 篇文档")
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"解析 optional_search_results 失败: {e}")
                    continue
        
        if not all_papers:
            logger.warning("没有需要下载的文档")
            return {"downloaded_papers": []}
        
        logger.info(f"总共需要下载 {len(all_papers)} 篇文档")
        
        download_tools = await self._get_download_tools()
        downloaded_papers = []
        
        # 按 source 分组
        papers_by_source = {}
        for paper in all_papers:
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
      
    def _build_graph(self):
        """构建 ExecutorAgent 的处理流程图"""
        builder = StateGraph(ExecutorState)
        
        # 添加节点
        builder.add_node("llm_decision", self._llm_decision_node)
        builder.add_node("optional_tool_node", self._optional_tool_node)
        builder.add_node("search", self._search_node)
        builder.add_node("clean_and_rerank", self._clean_node)
        builder.add_node("download", self._download_node)
        
        # 添加边
        builder.add_edge(START, "llm_decision")
        builder.add_conditional_edges(
            "llm_decision",
            self._should_call_optional_tools,
            {
                "optional_tool_node": "optional_tool_node",
                "search": "search"
            }
        )
        builder.add_edge("optional_tool_node", "llm_decision")
        builder.add_edge("search", "clean_and_rerank")
        builder.add_edge("clean_and_rerank", "download")
        builder.add_edge("download", END)
        
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

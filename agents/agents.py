from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from config import Config
from llms import get_llm
from typing import TypedDict, Annotated, List
from langgraph.graph import add_messages, StateGraph, START, END, state
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.messages import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import logging,json
from concurrent_log_handler import ConcurrentRotatingFileHandler
from tools import get_tools
# # 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

class PlannerState(TypedDict):
    planner_messages:Annotated[list[AnyMessage],add_messages]
    planner_result:AIMessage
    epoch:int

class ExecutorState(TypedDict):
    exector_messages:Annotated[list[AnyMessage],add_messages]
    exector_result:dict
class PlannerAgent:
    def __init__(self,pool:AsyncConnectionPool,modelname:ChatOpenAI=Config.LLM_PLANNER):
        self.chat_llm=get_llm(modelname)[0]
        self.memory=AsyncPostgresSaver(pool)
        self.graph=self._build_graph()

    def _json_node(self,state:PlannerState,planner_epoch=Config.PLANNER_EPOCH)->dict:
        prompt = """请根据用户提供的主要问题{query}，从多个维度拆解成子问题，并按DAG思想进行组织，确保子问题的依赖关系清晰。在回答中，你需要遵循以下要求：

        1. **拆解主问题：** 根据问题的多个层次和方面进行拆解，确保每个子问题具体且明确。
        2. **生成子问题ID和依赖关系：** 每个子问题应分配一个唯一的ID。子问题之间可能存在依赖关系（例如某个子问题需要其他子问题的答案才能解答），这种依赖关系需要通过“dep”字段体现，格式为数组。
        3. **结构化输出：** 使用以下JSON格式来输出每个任务及其依赖关系：

        ```
        {{
          "tasks": [
            {{"id": "T1", "query": "子问题1", "dep": []}},
            {{"id": "T2", "query": "子问题2", "dep": ["T1"]}},
            {{"id": "T3", "query": "子问题3", "dep": []}}
          ]
        }}
        ```

        **详细规则：**

        - 每个“任务”对应一个子问题。
        - “id”是子问题的唯一标识符。
        - “query”是子问题的具体内容。
        - “dep”是依赖数组，列出当前任务所依赖的其他任务ID。如果没有依赖，则为空数组“[]”
        - json的样例中虽然只放了三个子问题，但是你一定不能受到三个子问题数量的限制。而是要思考主问题，去拆解分析，打破欧式距离限制，帮助用户深层次的去了解问题
        """
        query=state["planner_messages"][0]
        template = ChatPromptTemplate.from_messages([
            {"role": "system", "content": prompt}],
        )
        try:
            chain = {"query": RunnablePassthrough()} | template | self.chat_llm
            if state["epoch"]<planner_epoch:
                result=chain.invoke(query)
                state["epoch"]+=1
                return {"planner_messages":[result]}
        except Exception as e:
            logger.error(f"planner_agent_node1 分析用户的问题时，出现错误:{e}")
            raise e

    def _condition_router(self,state:PlannerState,planner_epoch=Config.PLANNER_EPOCH):
        result=state["planner_messages"][-1]
        if isinstance(result,AIMessage):
            if state["epoch"]<planner_epoch:
                try:
                    _=json.loads(result.content)
                    state["planner_result"]=result
                    return "END"
                except Exception:
                    return "json_node"
            state["planner_result"]=AIMessage(content=str({"tasks":"error"}))
            logger.warning(f"planner_agent 达到最大迭代次数{planner_epoch}，仍未能生成有效的json结构，结束planner流程")
            return "END"
        logger.error(f"planner_agent 条件路由器收到非AIMessage类型的消息，类型为:{type(result)}，内容为:{result.content}")
        raise TypeError(f"planner_agent 条件路由器收到非AIMessage类型的消息，类型为:{type(result)}")
    def _build_graph(self):
        builder=StateGraph(PlannerState)
        builder.add_node("json_node",self._json_node)
        builder.add_edge(START, "json_node")
        builder.add_conditional_edges("json_node",self._condition_router,{"END":END,"json_node":"json_node"})
        builder.compile(checkpointer=self.memory)
        logger.info(f"完成planner_graph的初始化构造")
        return builder

    async def invoke(self,thread_id:str):
        query=state["planner_messages"][0]
        config = {"configurable": {"thread_id": thread_id}}
        try:
            response=await self.chat_llm.ainvoke(query,config)
            logger.info(f"planner_chatmodel对于用户的{query} 返回:{response}")
            return response
        except Exception as e:
            logger.error(f"planner_chatmodel对于用户的{query} 出现错误：{e}")
            raise e

    async def _clean(self):
        if self.memory:
            try:
                await  self.memory.aclose()
                logger.info("对实例化的PlannerAgent,完成对短期记忆连接池的断开处理")
            except Exception as e:
                logger.info(f"尝试对实例化的PlannerAgent与短期记忆连接池断开，出现错误：{e}")
class ExecutorAgent:
    def __init__(self,pool:AsyncConnectionPool,modelname:ChatOpenAI=Config.LLM_EXECUTOR):
        self.chat_llm = get_llm(modelname)[0]
        self.tools:list[BaseTool] =self._get_tools()
        self.memory = AsyncPostgresSaver(pool)
        self.graph = self._build_graph()
    def _get_tools(self)->list[BaseTool]:
        return get_tools()
    def _chat_model_node(self,state:ExecutorState):
        self.tools_llm=self.chat_llm.bind_tools(self.tools)
        query = state["exector_messages"][0].content


        ## 这里引导llm循环并行调用工具
        try:
            json_queries = json.loads(query)  # {"tasks":[{"id": "T1", "query": "子问题1", "dep": []}]}
            template = f"""{json_queries}这是一个多任务的DAG图，请根据任务的依赖关系，合理调用工具来完成每个子任务的回答。
            最终完成所有任务。你需要遵循以下要求：
            1. **任务不存在依赖关系：** 当不存在依赖关系时，按照DAG思想组织工具调用，并行执行这些不存在依赖的任务，从而节省时间。。
            2. **任务依赖关系存在时：**当任务存在依赖关系时，按照DAG思想，必须保证依赖的任务优先执行完成之后，再执行此任务。
            3. **工具调用：** 对于每个子任务，选择合适的工具进行回答。如果任务之间存在依赖关系，确保在调用工具时考虑这些依赖。
            4. **观察：**观察所有任务的搜索结果，问题与对应搜索结果是否对应齐全。如果齐全那结果是：{{"result":True,"missing":None}};如果不完整，例如T2和T4缺失，继续判断**是否存在依赖关系**。
            根据依赖情况，重新尝试调用1-2个工具。
            5. **轮次限制：**当同一个问题经过两次搜索都无法得到搜索结果时，标记该问题为无法回答，并尝试继续完成其他问题的结果搜索。
            6.**最终回答：**所有问题全部获得搜索结果：{{"result":True,"missing":None}}。如果问题缺失，就将缺失的问题id填入"missing"对应的value，例如T3问题两次调用工具都无法获取搜索结果，
            那最终的结果是：{{"result":False,"missing":["T3"]}}
            **样例：**
            {{
              "tasks": [
                {{"id": "T1", "query": "子问题1", "dep": []}},
                {{"id": "T2", "query": "子问题2", "dep": ["T1"]}},
                {{"id": "T3", "query": "子问题3", "dep": ["T1"]}}
                {{"id": "T4", "query": "子问题4", "dep": ["T2"]}}
              ]
            }}
            step1：T1没有依赖，可以直接调用工具执行，尝试调用工具获取"子问题1"的搜索结果；
            step2：T2和T3都依赖T1，必须等待T1完成。T1完成之后，为了节省时间，需要并行调用工具
            获取"子问题2"、"子问题3"的尝试搜索结果；
            step3：T4依赖T2，必须等待T2完成。T2完成之后，立即调用工具获取"子问题4"的搜索结果。
            step4-1：分析"T1"、"T2"、"T3"、"T4"的问题"子问题1"、"子问题2"、"子问题3"、"子问题4"是否都有各自对应搜索。如果都有，返回：{{"result":True,"missing":None}}。
            step4-2：观察窗到"T4"的"子问题4"没有对应的搜索结果。
            step5：对"T4"的"子问题4"尝试第二次调用工具获取搜索结果。如果仍然没有，标记为无法回答，继续完成其他问题的结果搜索。
            step6：观察到只有"T4"的"子问题4"没有搜索结果，并且"T4"的"子问题4"已经尝试两次调用工具都无法获取到搜索结果，最终返回：{{"result":False,"missing":["T4"]}}。
            """
            sys_template=template.format(query=query)
            prompt=ChatPromptTemplate.from_messages([{"role":"system","content":sys_template}])
            reponse=self.tools_llm.invoke(prompt)
            return {"exector_messages":[reponse]}
        except Exception as e:
            logger.error(f"executor_agent_node1 执行子任务时，出现错误:{e}")
            raise e

    def _tool_node(self,state:ExecutorState):
        try:
            result=ToolNode(self.tools).invoke(state["exector_messages"][-1])
            return {"exector_messages":[result]}
        except Exception as e:
            logger.error(f"executor_agent_toolnode 工具调用时，出现错误:{e}")
            raise e
    def _condition_router(self,state:ExecutorState):
        result=state["exector_messages"][-1]
        if isinstance(result,AIMessage):
            try:
                if result.tool_calls:
                    return "tool_node"
                elif result.content and not result.tool_calls:
                    return "END"
            except Exception as e:
                logger.error(f"executor_agent 条件路由器在解析AIMessage时，出现错误:{e}")
                raise e
        logger.error(f"executor_agent 条件路由器收到非AIMessage类型的消息，类型为:{type(result)}")
        raise TypeError(f"executor_agent 条件路由器收到非AIMessage类型的消息，类型为:{type(result)}")

    def _build_graph(self):
        builder=StateGraph(ExecutorState)
        builder.add_node("chat_model_node",self._chat_model_node)
        builder.add_node("tool_node",self._tool_node)
        builder.add_edge(START, "chat_model_node")
        builder.add_conditional_edges("chat_model_node",self._condition_router,{"tool_node":"tool_node","END":END})
        builder.compile(checkpointer=self.memory)
        logger.info(f"完成executor_graph的初始化构造")
        return builder
    async def _clean(self):
        if self.memory:
            try:
                await  self.memory.aclose()
                logger.info("对实例化的ExecutorAgent,完成对短期记忆连接池的断开处理")
            except Exception as e:
                logger.info(f"尝试对实例化的ExecutorAgent与短期记忆连接池断开，出现错误：{e}")
                raise e
    async def invoke(self,thread_id:str):
        query=state["exector_messages"][0]
        config = {"configurable": {"thread_id": thread_id}}
        try:
            response=await self.chat_llm.ainvoke(query,config)
            logger.info(f"executor_chatmodel对于用户的{query} 返回:{response}")
            return response
        except Exception as e:
            logger.error(f"executor_chatmodel对于用户的{query} 出现错误：{e}")
            raise e

from config import Config
from llms import get_llm
from typing import TypedDict,Annotated
from langgraph.graph import add_messages, StateGraph, START, END, state
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.messages import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import logging,json
from concurrent_log_handler import ConcurrentRotatingFileHandler
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
                return {"planner_messages":result}
        except Exception as e:
            logger.error(f"planner agent 分析用户的问题时，出现错误:{e}")
            raise e

    def _condition_router(self,state:PlannerState,planner_epoch=Config.PLANNER_EPOCH):
        result=state["planner_messages"][-1]
        if state["epoch"]<planner_epoch:
            try:
                _=json.loads(result.content)
                state["planner_result"]=result
                return "END"
            except Exception:
                return "json_node"
        state["planner_result"]=AIMessage(content=str({"tasks":"error"}))
        return "END"

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
    def __init__(self,pool:AsyncConnectionPool,modelname:ChatOpenAI=Config.LLM_PLANNER):
        self.chat_llm = get_llm(modelname)[0]
        self.tools=self._get_tools()
        self.memory = AsyncPostgresSaver(pool)
        self.graph = self._build_graph()
    def _get_tools(self):
        return
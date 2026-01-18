import logging
import asyncio
from typing import List, Dict
from tei_client import AsyncTeiClient
from core.config import Config
from concurrent_log_handler import ConcurrentRotatingFileHandler

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


class BGEReranker:
    """BGE Reranker 类，使用 tei_client 调用 TEI 部署的 bge-reranker"""
    
    def __init__(
        self,
        base_url: str = None,
        batch_size: int = None,
        max_concurrent: int = None
    ):
        """初始化 BGE Reranker
        
        Args:
            base_url: TEI API 地址
            batch_size: 批处理大小（保留参数以兼容，但 TEI 本身支持批量）
            max_concurrent: 最大并发请求数（保留参数以兼容）
        """
        self.base_url = base_url or Config.RERANK_BASE_URL
        
        # 创建 AsyncTeiClient
        self.client = AsyncTeiClient(base_url=self.base_url)
        
        logger.info(f"初始化 BGEReranker: base_url={self.base_url}")
    
    async def rerank_async(self, query: str, documents: List[str]) -> List[Dict[str, float]]:
        """异步对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            
        Returns:
            重排序结果列表，每个元素包含 {"index": int, "score": float}
            按分数从高到低排序
        """
        if not documents:
            logger.warning("文档列表为空，跳过 rerank")
            return []
        
        try:
            logger.info(f"开始异步 rerank，查询: {query[:50]}..., 文档数量: {len(documents)}")
            
            # 使用 tei_client 的 rerank 方法（TEI 本身支持批量处理）
            #返回的结果是[rerank(index:int=xx,score:int=xx)]
            results = await self.client.rerank(query=query, texts=documents)
            
            # 将结果转换为标准格式
            rerank_results = []
            for item in results:
                rerank_results.append({
                    "index": item.index,
                    "score": item.score
                })
            
            # 按分数排序(降序)
            rerank_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"异步 Rerank 完成，返回 {len(rerank_results)} 个结果")
            return rerank_results
            
        except Exception as e:
            logger.error(f"异步 Rerank 过程出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认分数
            return [{"index": i, "score": 0.5} for i in range(len(documents))]
    
    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, float]]:
        """同步接口：对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            
        Returns:
            重排序结果列表，每个元素包含 {"index": int, "score": float}
            按分数从高到低排序
        """
        # 在同步环境中运行异步方法
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.rerank_async(query, documents))
    
    async def close(self):
        """关闭 TEI 客户端"""
        await self.client.close()
        logger.info("BGEReranker TEI 客户端已关闭")
    
    def __del__(self):
        """析构函数，确保客户端关闭"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass


# 测试代码
if __name__ == "__main__":
    async def test_reranker():
        # 测试 BGEReranker
        reranker = BGEReranker()
        
        query = "人工智能在医疗领域的应用"
        documents = [
            "人工智能技术在医疗诊断中发挥着重要作用",
            "机器学习算法可以帮助医生更准确地诊断疾病",
            "今天天气很好，适合出去玩",
            "深度学习在图像识别领域取得了突破性进展",
            "AI 辅助医疗影像分析提高了诊断准确率",
            "自然语言处理技术在电子病历分析中的应用"
        ]
        
        results = await reranker.rerank_async(query, documents)
        
        print("Rerank 结果:")
        for item in results:
            print(f"  文档 {item['index']}: {documents[item['index']][:30]}... (分数: {item['score']:.4f})")
        
        await reranker.close()
    
    # 运行测试
    asyncio.run(test_reranker())

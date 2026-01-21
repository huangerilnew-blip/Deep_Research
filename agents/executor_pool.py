#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExecutorAgent Pool
管理多个 ExecutorAgent 实例，支持并发执行
"""

import asyncio
import logging
from typing import List, Dict
from psycopg_pool import AsyncConnectionPool
from concurrent_log_handler import ConcurrentRotatingFileHandler

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agents import ExecutorAgent
from core.config import Config

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


class ExecutorAgentPool:
    """ExecutorAgent 池
    
    管理多个 ExecutorAgent 实例，支持并发执行多个子问题
    """
    
    def __init__(
        self,
        pool: AsyncConnectionPool,
        pool_size: int,
        model: str = Config.LLM_EXECUTOR
    ):
        """初始化 ExecutorAgent Pool
        
        Args:
            pool: PostgreSQL 连接池
            pool_size: 池大小
            model: 使用的模型
        """
        self.pool = pool
        self.pool_size = pool_size
        self.model = model
        self.agents: List[ExecutorAgent] = []
        
        self._initialize_agents()
        
        logger.info(f"初始化 ExecutorAgentPool: pool_size={pool_size}, model={model}")
    
    def _initialize_agents(self):
        """创建指定数量的 ExecutorAgent 实例"""
        for i in range(self.pool_size):
            agent = ExecutorAgent(self.pool, self.model)
            self.agents.append(agent)
            logger.debug(f"创建 ExecutorAgent {i+1}/{self.pool_size}")
        
        logger.info(f"成功创建 {len(self.agents)} 个 ExecutorAgent 实例")
    
    async def execute_questions(
        self,
        questions: List[str],
        base_thread_id: str
    ) -> List[Dict]:
        """并发执行多个子问题
        
        Args:
            questions: 子问题列表
            base_thread_id: 基础线程 ID
            
        Returns:
            所有 ExecutorAgent 的结果列表
        """
        if not questions:
            logger.warning("没有子问题需要执行")
            return []
        
        logger.info(f"开始并发执行 {len(questions)} 个子问题")
        
        # 创建任务列表
        tasks = []
        for i, question in enumerate(questions):
            # 使用轮询方式分配 Agent
            agent = self.agents[i % self.pool_size]
            thread_id = f"{base_thread_id}_executor_{i}"
            
            # 创建异步任务
            task = agent.invoke(question, thread_id)
            tasks.append(task)
            
            logger.debug(f"分配子问题 {i+1} 到 Agent {i % self.pool_size}")
        
        # 并发执行所有任务，使用 return_exceptions=True 确保单个失败不影响其他
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"子问题 {i+1} 执行失败: {result}")
                failed_count += 1
            else:
                valid_results.append(result)
                logger.debug(f"子问题 {i+1} 执行成功")
        
        logger.info(f"并发执行完成: 成功 {len(valid_results)}/{len(questions)}, 失败 {failed_count}")
        
        return valid_results
    
    async def cleanup(self):
        """清理所有 Agent 资源"""
        logger.info("开始清理 ExecutorAgentPool 资源")
        
        cleanup_tasks = []
        for i, agent in enumerate(self.agents):
            task = agent._clean()
            cleanup_tasks.append(task)
        
        # 并发清理所有 Agent
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info(f"成功清理 {len(self.agents)} 个 ExecutorAgent")
    
    def __len__(self) -> int:
        """返回池中 Agent 的数量"""
        return len(self.agents)
    
    def __repr__(self) -> str:
        return f"ExecutorAgentPool(size={len(self.agents)}, model={self.model})"


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    async def test_executor_pool():
        print("=" * 60)
        print("ExecutorAgentPool 测试")
        print("=" * 60)
        
        # 创建模拟的连接池（实际使用时需要真实的 PostgreSQL 连接）
        from psycopg_pool import AsyncConnectionPool
        
        try:
            # 尝试创建连接池
            pool = AsyncConnectionPool(
                conninfo=Config.DB_URI,
                min_size=1,
                max_size=2,
                open=False  # 不立即打开连接
            )
            
            # 创建 ExecutorAgentPool
            executor_pool = ExecutorAgentPool(
                pool=pool,
                pool_size=3
            )
            
            print(f"\n✓ ExecutorAgentPool 创建成功")
            print(f"  - 池大小: {len(executor_pool)}")
            print(f"  - 模型: {executor_pool.model}")
            print(f"  - {executor_pool}")
            
            # 测试方法存在性
            print("\n✓ 测试方法存在性:")
            methods = ['_initialize_agents', 'execute_questions', 'cleanup']
            for method in methods:
                if hasattr(executor_pool, method):
                    print(f"  ✓ {method}")
                else:
                    print(f"  ✗ {method} 不存在")
            
            print("\n" + "=" * 60)
            print("基础测试通过！")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行测试
    asyncio.run(test_executor_pool())

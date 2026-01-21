#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent 深度搜索系统 - 主程序
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool
from concurrent_log_handler import ConcurrentRotatingFileHandler

from core.config import Config
from agents.multi_agent import MultiAgent
from api.routes import router

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

# 全局变量
_multi_agent = None
_db_pool = None


def get_multi_agent() -> MultiAgent:
    """获取 MultiAgent 实例"""
    return _multi_agent


def set_multi_agent(multi_agent: MultiAgent):
    """设置 MultiAgent 实例（用于测试）"""
    global _multi_agent
    _multi_agent = multi_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _multi_agent, _db_pool
    
    logger.info("=" * 70)
    logger.info("Multi-Agent 深度搜索系统启动中...")
    logger.info("=" * 70)
    
    try:
        # 1. 初始化 PostgreSQL 连接池
        logger.info("初始化 PostgreSQL 连接池...")
        _db_pool = AsyncConnectionPool(
            conninfo=Config.DB_URI,
            min_size=Config.MIN_SIZE,
            max_size=Config.MAX_SIZE
        )
        await _db_pool.open()
        logger.info(f"✓ PostgreSQL 连接池初始化成功: {Config.DB_URI}")
        
        # 2. 初始化 MultiAgent
        logger.info("初始化 MultiAgent...")
        _multi_agent = MultiAgent(
            pool=_db_pool,
            executor_pool_size=Config.EXECUTOR_POOL_SIZE,
            planner_model=Config.LLM_PLANNER,
            executor_model=Config.LLM_EXECUTOR
        )
        logger.info(f"✓ MultiAgent 初始化成功")
        
        logger.info("=" * 70)
        logger.info("系统启动完成！")
        logger.info(f"API 地址: http://{Config.HOST}:{Config.PORT}")
        logger.info(f"文档地址: http://{Config.HOST}:{Config.PORT}/docs")
        logger.info("=" * 70)
        
        yield
        
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 清理资源
        logger.info("=" * 70)
        logger.info("系统关闭中...")
        logger.info("=" * 70)
        
        if _multi_agent:
            try:
                await _multi_agent._cleanup()
                logger.info("✓ MultiAgent 资源清理完成")
            except Exception as e:
                logger.error(f"MultiAgent 清理失败: {e}")
        
        if _db_pool:
            try:
                await _db_pool.close()
                logger.info("✓ PostgreSQL 连接池关闭完成")
            except Exception as e:
                logger.error(f"连接池关闭失败: {e}")
        
        logger.info("系统已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="Multi-Agent 深度搜索系统",
    description="基于多智能体协作的深度文献检索和问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Multi-Agent 深度搜索系统",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Multi-Agent 深度搜索系统")
    print("=" * 70)
    print(f"启动服务器: http://{Config.HOST}:{Config.PORT}")
    print(f"API 文档: http://{Config.HOST}:{Config.PORT}/docs")
    print("=" * 70)
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,  # 生产环境设为 False
        log_level="info"
    )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest 配置文件
提供测试夹具和配置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import asyncio
from psycopg_pool import AsyncConnectionPool
from core.config import Config


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_pool():
    """创建数据库连接池"""
    pool = AsyncConnectionPool(
        conninfo=Config.DB_URI,
        min_size=Config.MIN_SIZE,
        max_size=Config.MAX_SIZE,
        open=False
    )
    
    await pool.open()
    yield pool
    await pool.close()


@pytest.fixture
def sample_query():
    """示例查询"""
    return "人工智能在医疗领域的应用有哪些？"


@pytest.fixture
def sample_thread_id():
    """示例线程 ID"""
    return "test_thread_001"

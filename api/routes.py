#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API 路由
提供 HTTP 接口
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from concurrent_log_handler import ConcurrentRotatingFileHandler

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

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["query"])


# 请求模型
class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="用户查询")
    thread_id: str = Field(default="default", description="会话线程 ID")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "人工智能在医疗领域的应用有哪些？",
                "thread_id": "user_123"
            }
        }


# 响应模型
class QueryResponse(BaseModel):
    """查询响应模型"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的回答")
    sub_questions: list = Field(default=[], description="拆解的子问题")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    error: str = Field(default="", description="错误信息")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "query": "人工智能在医疗领域的应用有哪些？",
                "answer": "人工智能在医疗领域有多种应用...",
                "sub_questions": ["AI 在医疗诊断中的应用？", "AI 在药物研发中的作用？"],
                "metadata": {
                    "retrieved_count": 50,
                    "unique_count": 30,
                    "reranked_count": 20
                },
                "error": ""
            }
        }


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_query(request: QueryRequest) -> QueryResponse:
    """处理用户查询
    
    Args:
        request: 查询请求
        
    Returns:
        查询响应
    """
    logger.info(f"收到查询请求: query='{request.query}', thread_id='{request.thread_id}'")
    
    try:
        # 获取 MultiAgent 实例（从应用状态中获取）
        from main import get_multi_agent
        multi_agent = get_multi_agent()
        
        if multi_agent is None:
            logger.error("MultiAgent 未初始化")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="服务未就绪，请稍后重试"
            )
        
        # 调用 MultiAgent 处理查询
        result = await multi_agent.process_query(
            user_query=request.query,
            thread_id=request.thread_id
        )
        
        # 构建响应
        if 'error' in result:
            logger.error(f"查询处理失败: {result['error']}")
            return QueryResponse(
                success=False,
                query=request.query,
                answer=result.get('answer', ''),
                error=result['error']
            )
        else:
            logger.info(f"查询处理成功: thread_id='{request.thread_id}'")
            return QueryResponse(
                success=True,
                query=result['query'],
                answer=result['answer'],
                sub_questions=result.get('sub_questions', []),
                metadata=result.get('metadata', {}),
                error=""
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理查询时发生异常: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务器内部错误: {str(e)}"
        )


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """健康检查
    
    Returns:
        健康状态
    """
    return {
        "status": "healthy",
        "service": "multi-agent-deep-search"
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("API Routes 测试")
    print("=" * 60)
    print("✓ API Routes 模块导入成功")
    print("\n可用端点:")
    print("  - POST /api/v1/query")
    print("  - GET  /api/v1/health")

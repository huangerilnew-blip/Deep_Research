#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据模型
定义系统中使用的核心数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class QuestionsPool:
    """问题池数据模型
    
    存储 PlannerAgent 生成的原始子问题以及从 Markdown 文档中改写的问题
    """
    original_questions: List[str] = field(default_factory=list)  # PlannerAgent 生成的原始子问题
    rewritten_questions: List[str] = field(default_factory=list)  # 从文档改写的问题
    
    def get_all_questions(self) -> List[str]:
        """获取所有问题
        
        Returns:
            所有问题的列表（原始 + 改写）
        """
        return self.original_questions + self.rewritten_questions
    
    def add_rewritten_questions(self, questions: List[str]):
        """添加改写的问题（自动去重）
        
        Args:
            questions: 新的改写问题列表
        """
        # 去重：只添加不存在的问题
        existing = set(self.get_all_questions())
        new_questions = [q for q in questions if q not in existing]
        self.rewritten_questions.extend(new_questions)
    
    def add_original_questions(self, questions: List[str]):
        """添加原始子问题
        
        Args:
            questions: 原始子问题列表
        """
        self.original_questions.extend(questions)
    
    def __len__(self) -> int:
        """返回问题总数"""
        return len(self.original_questions) + len(self.rewritten_questions)
    
    def __repr__(self) -> str:
        return f"QuestionsPool(original={len(self.original_questions)}, rewritten={len(self.rewritten_questions)}, total={len(self)})"


@dataclass
class DocumentMetadata:
    """文档元数据
    
    存储下载文档的元信息
    """
    source: str  # 数据源（openalex, semantic_scholar, wikipedia, tavily, sec_edgar, akshare）
    title: str  # 文档标题
    abstract: str = ""  # 摘要
    url: str = ""  # 原始 URL
    local_path: str = ""  # 本地存储路径
    rerank_score: float = 0.0  # Rerank 分数
    download_time: Optional[datetime] = None  # 下载时间
    extra: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'source': self.source,
            'title': self.title,
            'abstract': self.abstract,
            'url': self.url,
            'local_path': self.local_path,
            'rerank_score': self.rerank_score,
            'download_time': self.download_time.isoformat() if self.download_time else None,
            'extra': self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """从字典创建实例"""
        download_time = data.get('download_time')
        if download_time and isinstance(download_time, str):
            download_time = datetime.fromisoformat(download_time)
        
        return cls(
            source=data.get('source', ''),
            title=data.get('title', ''),
            abstract=data.get('abstract', ''),
            url=data.get('url', ''),
            local_path=data.get('local_path', ''),
            rerank_score=data.get('rerank_score', 0.0),
            download_time=download_time,
            extra=data.get('extra', {})
        )


@dataclass
class MarkdownChunk:
    """Markdown 切割片段
    
    存储切割后的 Markdown 文档片段信息
    """
    content: str  # 内容
    doc_title: str  # 原始文档标题
    source: str  # 来源
    header_level: int = 0  # 标题层级（0=无标题, 1=H1, 2=H2, 3=H3）
    chunk_index: int = 0  # 在文档中的位置
    embedding: Optional[List[float]] = None  # 向量（可选）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'content': self.content,
            'doc_title': self.doc_title,
            'source': self.source,
            'header_level': self.header_level,
            'chunk_index': self.chunk_index,
            'embedding': self.embedding,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarkdownChunk':
        """从字典创建实例"""
        return cls(
            content=data.get('content', ''),
            doc_title=data.get('doc_title', ''),
            source=data.get('source', ''),
            header_level=data.get('header_level', 0),
            chunk_index=data.get('chunk_index', 0),
            embedding=data.get('embedding'),
            metadata=data.get('metadata', {})
        )


@dataclass
class RetrievedContext:
    """检索到的语料
    
    存储从向量库检索到的相关语料信息
    """
    content: str  # 内容
    source: str  # 来源（base_data 或文档标题）
    score: float  # 相似度分数
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'content': self.content,
            'source': self.source,
            'score': self.score,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedContext':
        """从字典创建实例"""
        return cls(
            content=data.get('content', ''),
            source=data.get('source', ''),
            score=data.get('score', 0.0),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"RetrievedContext(source='{self.source}', score={self.score:.3f}, content='{content_preview}')"


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("数据模型测试")
    print("=" * 60)
    
    # 测试 QuestionsPool
    print("\n1. 测试 QuestionsPool:")
    pool = QuestionsPool()
    pool.add_original_questions(["问题1？", "问题2？", "问题3？"])
    print(f"   添加原始问题后: {pool}")
    
    pool.add_rewritten_questions(["改写问题1？", "改写问题2？"])
    print(f"   添加改写问题后: {pool}")
    
    # 测试去重
    pool.add_rewritten_questions(["改写问题1？", "新问题？"])
    print(f"   测试去重后: {pool}")
    
    all_questions = pool.get_all_questions()
    print(f"   所有问题: {all_questions}")
    print(f"   ✓ QuestionsPool 测试通过")
    
    # 测试 DocumentMetadata
    print("\n2. 测试 DocumentMetadata:")
    doc_meta = DocumentMetadata(
        source="openalex",
        title="测试文档",
        abstract="这是一个测试摘要",
        url="https://example.com/doc",
        local_path="/path/to/doc.pdf",
        rerank_score=0.85,
        download_time=datetime.now()
    )
    print(f"   创建文档元数据: {doc_meta.title}")
    
    # 测试序列化
    doc_dict = doc_meta.to_dict()
    doc_meta2 = DocumentMetadata.from_dict(doc_dict)
    assert doc_meta.title == doc_meta2.title
    print(f"   ✓ DocumentMetadata 序列化测试通过")
    
    # 测试 MarkdownChunk
    print("\n3. 测试 MarkdownChunk:")
    chunk = MarkdownChunk(
        content="这是一个测试片段",
        doc_title="测试文档",
        source="openalex",
        header_level=2,
        chunk_index=0
    )
    print(f"   创建 Markdown 片段: level={chunk.header_level}, index={chunk.chunk_index}")
    
    # 测试序列化
    chunk_dict = chunk.to_dict()
    chunk2 = MarkdownChunk.from_dict(chunk_dict)
    assert chunk.content == chunk2.content
    print(f"   ✓ MarkdownChunk 序列化测试通过")
    
    # 测试 RetrievedContext
    print("\n4. 测试 RetrievedContext:")
    context = RetrievedContext(
        content="这是检索到的相关内容",
        source="base_data",
        score=0.92,
        metadata={"type": "company_info"}
    )
    print(f"   {context}")
    
    # 测试序列化
    context_dict = context.to_dict()
    context2 = RetrievedContext.from_dict(context_dict)
    assert context.content == context2.content
    print(f"   ✓ RetrievedContext 序列化测试通过")
    
    print("\n" + "=" * 60)
    print("所有数据模型测试通过！")
    print("=" * 60)

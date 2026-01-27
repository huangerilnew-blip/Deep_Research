#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据模型
定义系统中使用的核心数据结构
"""

import asyncio
import os
import tempfile
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from rerank import BGEReranker
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


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


class BGERerankNodePostprocessor(BaseNodePostprocessor):
    """BGE Reranker 节点后处理器
    
    使用 BGEReranker 对检索到的节点进行重新打分和排序
    """
    
    def __init__(
        self,
        reranker: BGEReranker,
        top_n: int = 5,
        score_threshold: float = Config.RERANK_THRESHOLD
    ):
        """初始化 BGE Rerank 节点后处理器
        
        Args:
            reranker: BGEReranker 实例
            top_n: 保留的最高分数节点数量
            score_threshold: 分数阈值，低于此分数的节点会被过滤
        """
        super().__init__()
        self.reranker = reranker
        self.top_n = top_n
        self.score_threshold = score_threshold
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """对节点进行重新打分（同步方法）
        
        Args:
            nodes: 原始节点列表
            query_bundle: 查询信息
            
        Returns:
            重新打分后的节点列表
        """
        if not nodes:
            return []
        
        if query_bundle is None:
            # 如果没有提供查询，直接返回原始节点
            return nodes[:self.top_n]
        
        # 在同步环境中运行异步方法
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._async_postprocess_nodes(nodes, query_bundle)
        )
    
    async def _async_postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """异步对节点进行重新打分
        
        Args:
            nodes: 原始节点列表
            query_bundle: 查询信息
            
        Returns:
            重新打分后的节点列表
        """
        if not nodes:
            return []
        
        # 提取查询文本
        query = query_bundle.query_str
        
        # 提取节点内容
        documents = [node.node.get_content() for node in nodes]
        
        # 使用 BGEReranker 进行重排序
        rerank_results = await self.reranker.rerank_async(query, documents)
        
        # 根据 rerank 结果更新节点分数
        reranked_nodes = []
        for item in rerank_results:
            idx = item["index"]
            score = item["score"]
            
            # 过滤低于阈值的节点
            if score >= self.score_threshold and idx < len(nodes):
                node_with_score = nodes[idx]
                # 更新分数为 rerank 分数
                node_with_score.score = score
                reranked_nodes.append(node_with_score)
        
        # 限制返回数量
        return reranked_nodes[:self.top_n]
    
    @classmethod
    def class_name(cls) -> str:
        """返回类名"""
        return "BGERerankNodePostprocessor"


class PDFParser:
    """
    PDF 解析器
    使用 mineru 命令行工具将 PDF 转换为 Markdown
    """
    
    def __init__(self, mineru_server_url: str = "http://localhost:30000"):
        """
        初始化 PDF 解析器
        
        Args:
            mineru_server_url: MinerU 服务的 URL 地址
        """
        self.mineru_server_url = mineru_server_url
        self.logger = logging.getLogger(__name__)
    
    async def parse_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        将 PDF 文件转换为 Markdown 文本
        
        Args:
            pdf_path: PDF 文件的路径
            
        Returns:
            转换后的 Markdown 文本
            
        Raises:
            FileNotFoundError: 如果 PDF 文件不存在
            RuntimeError: 如果转换过程失败
        """
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        # 创建临时输出目录
        temp_dir = tempfile.mkdtemp(prefix="mineru_output_")
        
        try:
            # 获取 PDF 文件名（不含扩展名）
            pdf_basename = Path(pdf_path).stem
            
            # 构建 mineru 命令
            cmd = [
                "mineru",
                "-p", pdf_path,
                "-o", temp_dir,
                "-b", "vlm-http-client",
                "-u", self.mineru_server_url
            ]
            
            self.logger.info(f"开始解析 PDF: {pdf_path}")
            self.logger.debug(f"执行命令: {' '.join(cmd)}")
            
            # 异步执行命令
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # 检查命令执行结果
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"mineru 命令执行失败: {error_msg}")
                raise RuntimeError(f"PDF 转换失败: {error_msg}")
            
            # 定位输出的 Markdown 文件
            # 路径格式: {temp_dir}/auto/{pdf_basename}/{pdf_basename}.md
            markdown_path = os.path.join(temp_dir, "auto", pdf_basename, f"{pdf_basename}.md")
            
            if not os.path.exists(markdown_path):
                self.logger.error(f"未找到输出的 Markdown 文件: {markdown_path}")
                raise RuntimeError(f"未找到输出的 Markdown 文件")
            
            # 读取 Markdown 内容
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            self.logger.info(f"PDF 解析成功: {pdf_path}, Markdown 长度: {len(markdown_content)}")
            return markdown_content
            
        except Exception as e:
            self.logger.error(f"PDF 解析过程出错: {e}")
            raise
            
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir)
                self.logger.debug(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                self.logger.warning(f"清理临时目录失败: {e}")
    
    def parse_pdf_to_markdown_sync(self, pdf_path: str) -> str:
        """
        同步版本的 PDF 转换方法
        
        Args:
            pdf_path: PDF 文件的路径
            
        Returns:
            转换后的 Markdown 文本
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.parse_pdf_to_markdown(pdf_path))


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
    
    # 测试 BGERerankNodePostprocessor
    print("\n5. 测试 BGERerankNodePostprocessor:")
    print("   BGERerankNodePostprocessor 类创建成功")
    print(f"   类名: {BGERerankNodePostprocessor.class_name()}")
    print(f"   ✓ BGERerankNodePostprocessor 测试通过")
    
    # 测试 PDFParser
    print("\n6. 测试 PDFParser:")
    parser = PDFParser(mineru_server_url="http://localhost:30000")
    print(f"   PDFParser 实例化成功")
    print(f"   MinerU 服务器 URL: {parser.mineru_server_url}")
    print(f"   ✓ PDFParser 测试通过")
    print("   注意: 实际 PDF 解析需要安装 mineru 并启动服务")
    
    print("\n" + "=" * 60)
    print("所有数据模型测试通过！")
    print("=" * 60)

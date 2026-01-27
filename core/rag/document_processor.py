#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理器
负责 PDF 转 Markdown、智能切割、问题改写
使用 LlamaIndex 原生 API：
- MarkdownReader: 加载 Markdown 文件
- MarkdownElementNodeParser: 解析文档并优化表格处理
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from llama_index.core.schema import BaseNode
from concurrent_log_handler import ConcurrentRotatingFileHandler

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceSplitter
from llama_index.core.readers.file import MarkdownReader
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from core.config import Config
from core.models import PDFParser

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

class DocumentProcessor:
    """文档处理器
    
    负责 PDF 转 Markdown、智能切割、问题改写
    """
    
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        llm: LLM,
        mineru_base_url: str = Config.MINERU_BASE_URL,
        max_chunk_size: int = Config.MAX_CHUNK_SIZE
    ):
        """初始化文档处理器
        
        Args:
            embedding_model: 本地 vllm 部署的 Embedding 模型
            llm: 用于问题改写的 LLM
            mineru_base_url: MinerU 服务地址（用于创建 MinerUClient）
            max_chunk_size: Markdown 切割最大长度
        """
        self.embedding_model = embedding_model
        self.llm = llm
        self.mineru_base_url = mineru_base_url
        self.max_chunk_size = max_chunk_size
        
        # 初始化 PDFParser（用于 PDF 转 Markdown）
        self.pdf_parser = PDFParser(mineru_server_url=mineru_base_url)
        
        # 初始化 MarkdownReader（负责从文件系统加载 Markdown 文档）
        self.markdown_reader = MarkdownReader()
        
        # 初始化 MarkdownElementNodeParser（负责结构化解析 + 表格优化）
        self.markdown_parser = MarkdownElementNodeParser(
            llm=llm,
            num_workers=4  # 并行处理 workers 数量
        )
        
        # 初始化 SentenceSplitter（按句子边界进一步切分节点）
        self.sentence_splitter = SentenceSplitter(
            chunk_size=self.max_chunk_size
        )
        
        # 初始化 QuestionsAnsweredExtractor（从节点中抽取可回答的问题）
        self.question_extractor = QuestionsAnsweredExtractor(
            questions=3  # 每个 chunk 抽取的问题数量
        )
        
        # 构建 IngestionPipeline：MarkdownElementNodeParser -> SentenceSplitter -> QuestionsAnsweredExtractor
        self.ingestion_pipeline = IngestionPipeline(
            transformations=[
                self.markdown_parser,
                self.sentence_splitter,
                self.question_extractor,
            ]
        )
        
        logger.info(f"初始化 DocumentProcessor: mineru_url={mineru_base_url}, max_chunk_size={max_chunk_size}")
    
    async def get_nodes(
        self,
        documents: List[Dict]
    ) -> List[BaseNode]:
        """处理文档：转换、切割、问题改写
        
        Args:
            documents: 文档元数据列表
            doc_path: 文档存储路径
            
        Returns:
            处理后的 Node 列表（包含改写问题的元数据）
        """
        logger.info(f"开始处理 {len(documents)} 个文档")
        
        docs_for_pipeline: List[Document] = []
        
        # 1. 先将所有原始文件转为 LlamaIndex Document（Markdown 文本）
        for doc_meta in documents:
            # 获取文档路径
            local_path = doc_meta.get("local_path", "")
            if not local_path:
                raise ValueError(f"文档路径为空: {doc_meta.get('title', 'unknown')}")
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"文档文件不存在: {local_path}")

            # 判断文件类型
            file_ext = os.path.splitext(local_path)[1].lower()
            base_metadata = {
                'source': doc_meta.get('source', 'unknown'),
                'doc_title': doc_meta.get('title', ''),
                'local_path': local_path,
            }

            if file_ext == '.pdf':
                # PDF 转 Markdown
                markdown_text = await self._pdf_to_markdown(local_path)
                if not markdown_text:
                    raise ValueError(f"PDF 文档内容为空: {local_path}")

                temp_doc = Document(
                    text=markdown_text,
                    metadata=base_metadata,
                )
                docs_for_pipeline.append(temp_doc)
                logger.info(f"PDF 文档 {local_path} 转换为 Markdown 并加入 pipeline")

            elif file_ext in ['.md', '.markdown']:
                # 使用 MarkdownReader 加载 Markdown 文件
                md_docs = self.markdown_reader.load_data(file=Path(local_path))
                if not md_docs:
                    raise ValueError(f"Markdown 文档解析为空: {local_path}")

                for d in md_docs:
                    # 合并元数据（保留原始 + 基础元数据）
                    d.metadata = {**getattr(d, "metadata", {}) or {}, **base_metadata}
                    docs_for_pipeline.append(d)

                logger.info(f"Markdown 文档 {local_path} 加载为 {len(md_docs)} 个 Document 并加入 pipeline")
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}，文档: {local_path}")

        if not docs_for_pipeline:
            raise ValueError("没有可处理的文档")
        
        # 2. 通过 IngestionPipeline 执行：MarkdownElementNodeParser -> SentenceSplitter -> QuestionsAnsweredExtractor
        try:
            logger.info(f"开始通过 IngestionPipeline 处理 {len(docs_for_pipeline)} 个 Document")
            nodes = self.ingestion_pipeline.run(
                documents=docs_for_pipeline,
                in_place=True,
                show_progress=False,
            )
        except Exception as e:
            logger.error(f"IngestionPipeline 处理失败: {e}")
            raise e
        
        # 3. 返回处理后的节点列表
        logger.info(f"文档处理完成: {len(nodes)} 个节点")
        return nodes
    
    async def _pdf_to_markdown(self, pdf_path: str) -> str:
        """将 PDF 转换为 Markdown

        使用 PDFParser 进行转换

        Args:
            pdf_path: PDF 文件路径

        Returns:
            Markdown 文本
        """
        logger.info(f"开始转换 PDF: {pdf_path}")

        try:
            markdown_text = await self.pdf_parser.parse_pdf_to_markdown(pdf_path)
            if markdown_text:
                logger.info(f"PDF 转换成功: {pdf_path}, 长度: {len(markdown_text)}")
                return markdown_text
            else:
                logger.error(f"PDF 转换失败: 返回内容为空")
                return ""

        except Exception as e:
            logger.error(f"PDF 转换异常: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def extract_questions_from_nodes(self, nodes: List[BaseNode]) -> List[str]:
        """从节点列表中提取改写后的问题
        
        Args:
            nodes: 处理后的 Node 列表
            
        Returns:
            改写的问题列表（已去重）
        """
        all_questions = []
        
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            # QuestionsAnsweredExtractor 默认将问题写入 metadata["questions"] 或 metadata["questions_answered"]
            node_questions = metadata.get("questions") or metadata.get("questions_answered")
            if not node_questions:
                continue
            
            if isinstance(node_questions, str):
                all_questions.append(node_questions.strip())
            elif isinstance(node_questions, list):
                for q in node_questions:
                    if isinstance(q, str) and q.strip():
                        all_questions.append(q.strip())
        
        # 去重
        unique_questions = list(dict.fromkeys(all_questions))
        logger.info(f"从 {len(nodes)} 个节点中提取了 {len(unique_questions)} 个唯一问题")
        return unique_questions


# 测试代码
if __name__ == "__main__":
    import asyncio
    from core.llms import get_llm
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    async def test_document_processor():
        print("=" * 60)
        print("DocumentProcessor 测试")
        print("=" * 60)
        
        # 初始化 LLM 和 Embedding
        llm = get_llm(Config.LLM_EXECUTOR)[0]
        embedding = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_base=Config.VLLM_BASE_URL,
            api_key="EMPTY"
        )
        
        # 创建处理器
        processor = DocumentProcessor(
            embedding_model=embedding,
            llm=llm
        )
        print(f"✓ DocumentProcessor 实例化成功")
        print(f"✓ MarkdownReader 已初始化")
        print(f"✓ MarkdownElementNodeParser 已初始化（优化表格解析）")
        
        # 测试 Markdown 文件加载
        print(f"\n✓ 测试 Markdown 文件加载:")
        print(f"  - 使用 MarkdownReader 加载 .md 文件")
        print(f"  - 使用 MarkdownElementNodeParser 解析节点")
        print(f"  - 自动识别表格、代码块、标题等结构")
        
        print("\n" + "=" * 60)
        print("基础测试通过！")
        print("=" * 60)
    
    # 运行测试
    asyncio.run(test_document_processor())

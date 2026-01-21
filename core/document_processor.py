#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档处理器
负责 PDF 转 Markdown、智能切割、问题改写
"""

import os
import re
import json
import logging
import aiohttp
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent_log_handler import ConcurrentRotatingFileHandler

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM

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


# 问题改写提示词
QUESTION_REWRITE_PROMPT = """
分析以下文档片段，提取其中的关键信息，并将其改写为 1-3 个具体的问题。

文档片段：
{chunk}

要求：
1. 问题应该具体且明确
2. 问题应该能够通过文档内容回答
3. 避免过于宽泛或模糊的问题
4. 每个问题独立成行
5. 问题必须以问号结尾

改写的问题：
"""


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
            mineru_base_url: vllm 部署的 MinerU 服务地址
            max_chunk_size: Markdown 切割最大长度
        """
        self.embedding_model = embedding_model
        self.llm = llm
        self.mineru_base_url = mineru_base_url
        self.max_chunk_size = max_chunk_size
        
        logger.info(f"初始化 DocumentProcessor: mineru_url={mineru_base_url}, max_chunk_size={max_chunk_size}")
    
    async def process_documents(
        self,
        documents: List[Dict],
        doc_path: str = Config.DOC_SAVE_PATH
    ) -> Tuple[List[Document], List[str]]:
        """处理文档：转换、切割、问题改写
        
        Args:
            documents: 文档元数据列表
            doc_path: 文档存储路径
            
        Returns:
            (LlamaIndex Document 列表, 改写的问题列表)
        """
        logger.info(f"开始处理 {len(documents)} 个文档")
        
        all_llama_docs = []
        all_questions = []
        
        for doc_meta in documents:
            try:
                # 获取文档路径
                local_path = doc_meta.get("local_path", "")
                if not local_path or not os.path.exists(local_path):
                    logger.warning(f"文档文件不存在: {local_path}")
                    continue
                
                # 判断文件类型
                file_ext = os.path.splitext(local_path)[1].lower()
                
                if file_ext == '.pdf':
                    # PDF 转 Markdown
                    markdown_text = await self._pdf_to_markdown(local_path)
                elif file_ext in ['.md', '.markdown']:
                    # 直接读取 Markdown
                    with open(local_path, 'r', encoding='utf-8') as f:
                        markdown_text = f.read()
                else:
                    logger.warning(f"不支持的文件类型: {file_ext}")
                    continue
                
                if not markdown_text:
                    logger.warning(f"文档内容为空: {local_path}")
                    continue
                
                # 切割 Markdown
                chunks = self._split_markdown(markdown_text)
                logger.info(f"文档 {local_path} 切割为 {len(chunks)} 个片段")
                
                # 创建 LlamaIndex Document
                for i, chunk_data in enumerate(chunks):
                    llama_doc = Document(
                        text=chunk_data['content'],
                        metadata={
                            'source': doc_meta.get('source', 'unknown'),
                            'doc_title': doc_meta.get('title', ''),
                            'header_level': chunk_data['header_level'],
                            'chunk_index': i,
                            'local_path': local_path
                        }
                    )
                    all_llama_docs.append(llama_doc)
                
                # 问题改写（只对部分有代表性的 chunk 进行改写，避免过多）
                sample_chunks = chunks[:min(5, len(chunks))]  # 最多取前 5 个 chunk
                questions = await self._rewrite_to_questions(
                    [c['content'] for c in sample_chunks]
                )
                all_questions.extend(questions)
                
            except Exception as e:
                logger.error(f"处理文档失败: {doc_meta.get('title', 'unknown')}, 错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info(f"文档处理完成: {len(all_llama_docs)} 个片段, {len(all_questions)} 个改写问题")
        return all_llama_docs, all_questions
    
    async def _pdf_to_markdown(self, pdf_path: str) -> str:
        """将 PDF 转换为 Markdown
        
        使用 vllm 部署的 MinerU 进行转换
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            Markdown 文本
        """
        logger.info(f"开始转换 PDF: {pdf_path}")
        
        try:
            # 调用 MinerU API
            async with aiohttp.ClientSession() as session:
                # 读取 PDF 文件
                with open(pdf_path, 'rb') as f:
                    pdf_data = f.read()
                
                # 构建请求
                data = aiohttp.FormData()
                data.add_field('file',
                              pdf_data,
                              filename=os.path.basename(pdf_path),
                              content_type='application/pdf')
                
                # 发送请求
                url = f"{self.mineru_base_url}/convert"
                async with session.post(url, data=data, timeout=300) as response:
                    if response.status == 200:
                        result = await response.json()
                        markdown_text = result.get('markdown', '')
                        logger.info(f"PDF 转换成功: {pdf_path}, 长度: {len(markdown_text)}")
                        return markdown_text
                    else:
                        error_text = await response.text()
                        logger.error(f"PDF 转换失败: {response.status}, {error_text}")
                        return ""
        
        except Exception as e:
            logger.error(f"PDF 转换异常: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _split_markdown(self, markdown_text: str) -> List[Dict]:
        """切割 Markdown 文档
        
        按标题等级切割，超长文本按语句边界切割
        
        Args:
            markdown_text: Markdown 文本
            
        Returns:
            切割后的片段列表，每个元素包含 {'content': str, 'header_level': int}
        """
        # 先按标题切割
        chunks = self._split_by_headers(markdown_text)
        
        # 对超长 chunk 进行二次切割
        final_chunks = []
        for chunk_data in chunks:
            content = chunk_data['content']
            header_level = chunk_data['header_level']
            
            if len(content) <= self.max_chunk_size:
                final_chunks.append(chunk_data)
            else:
                # 按语句边界切割
                sub_chunks = self._split_by_sentences(content, self.max_chunk_size)
                for sub_content in sub_chunks:
                    final_chunks.append({
                        'content': sub_content,
                        'header_level': header_level
                    })
        
        return final_chunks
    
    def _split_by_headers(self, markdown_text: str) -> List[Dict]:
        """按标题等级切割 Markdown
        
        识别 H1 (#), H2 (##), H3 (###) 标题
        
        Args:
            markdown_text: Markdown 文本
            
        Returns:
            切割后的片段列表
        """
        chunks = []
        
        # 匹配标题的正则表达式
        header_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        
        # 找到所有标题位置
        matches = list(header_pattern.finditer(markdown_text))
        
        if not matches:
            # 没有标题，整个文档作为一个 chunk
            return [{'content': markdown_text.strip(), 'header_level': 0}]
        
        # 按标题切割
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
            
            content = markdown_text[start:end].strip()
            header_level = len(match.group(1))  # # 的数量
            
            if content:
                chunks.append({
                    'content': content,
                    'header_level': header_level
                })
        
        return chunks
    
    def _split_by_sentences(self, text: str, max_length: int) -> List[str]:
        """按语句边界切割超长文本
        
        Args:
            text: 文本
            max_length: 最大长度
            
        Returns:
            切割后的文本列表
        """
        if len(text) <= max_length:
            return [text]
        
        # 句子分隔符
        sentence_endings = r'[.!?。！？]+'
        
        # 按句子分割
        sentences = re.split(f'({sentence_endings})', text)
        
        # 重新组合句子和标点
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
            combined_sentences.append(sentence + punctuation)
        
        # 如果最后一个元素不是标点，也加入
        if len(sentences) % 2 == 1:
            combined_sentences.append(sentences[-1])
        
        # 按最大长度组合句子
        chunks = []
        current_chunk = ""
        
        for sentence in combined_sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _rewrite_to_questions(self, chunks: List[str]) -> List[str]:
        """从文档片段中提取关键信息并改写为问题
        
        Args:
            chunks: 文档片段列表
            
        Returns:
            改写的问题列表
        """
        all_questions = []
        
        for chunk in chunks:
            try:
                # 跳过太短的 chunk
                if len(chunk) < 50:
                    continue
                
                # 构建提示词
                prompt = QUESTION_REWRITE_PROMPT.format(chunk=chunk[:500])  # 限制长度
                
                # 调用 LLM
                response = await self.llm.acomplete(prompt)
                response_text = response.text.strip()
                
                # 解析问题（每行一个问题）
                questions = [
                    q.strip()
                    for q in response_text.split('\n')
                    if q.strip() and '?' in q or '？' in q
                ]
                
                all_questions.extend(questions)
                logger.debug(f"从 chunk 中提取了 {len(questions)} 个问题")
                
            except Exception as e:
                logger.error(f"问题改写失败: {e}")
                continue
        
        # 去重
        unique_questions = list(set(all_questions))
        logger.info(f"问题改写完成: {len(unique_questions)} 个唯一问题")
        
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
        
        # 测试 Markdown 切割
        test_markdown = """
# 标题 1

这是第一段内容。这是第一段内容的第二句话。

## 标题 2

这是第二段内容。这是第二段内容的第二句话。这是第二段内容的第三句话。

### 标题 3

这是第三段内容。
"""
        
        chunks = processor._split_markdown(test_markdown)
        print(f"\n✓ Markdown 切割测试:")
        print(f"  - 原始长度: {len(test_markdown)}")
        print(f"  - 切割片段数: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"  - Chunk {i+1}: level={chunk['header_level']}, length={len(chunk['content'])}")
        
        # 测试语句切割
        long_text = "这是一个很长的句子。" * 100
        sentence_chunks = processor._split_by_sentences(long_text, 200)
        print(f"\n✓ 语句切割测试:")
        print(f"  - 原始长度: {len(long_text)}")
        print(f"  - 切割片段数: {len(sentence_chunks)}")
        print(f"  - 最大片段长度: {max(len(c) for c in sentence_chunks)}")
        
        print("\n" + "=" * 60)
        print("基础测试通过！")
        print("=" * 60)
    
    # 运行测试
    asyncio.run(test_document_processor())

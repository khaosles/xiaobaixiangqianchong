"""
RAG工具模块
用于PDF问答的检索增强生成
"""
import os
from typing import List, Dict
from openai import AsyncOpenAI
import numpy as np


class SimpleRAG:
    """简单的RAG实现，用于PDF问答"""
    
    def __init__(self):
        # 如果配置了单独的嵌入服务，使用嵌入服务的配置，否则使用主模型服务
        embedding_base_url = os.getenv("SCI_EMBEDDING_BASE_URL")
        embedding_api_key = os.getenv("SCI_EMBEDDING_API_KEY")
        
        if embedding_base_url and embedding_api_key:
            self.client = AsyncOpenAI(
                base_url=embedding_base_url,
                api_key=embedding_api_key
            )
        else:
            # 回退到主模型服务
            self.client = AsyncOpenAI(
                base_url=os.getenv("SCI_MODEL_BASE_URL"),
                api_key=os.getenv("SCI_MODEL_API_KEY")
            )
        
        self.embedding_model = os.getenv("SCI_EMBEDDING_MODEL", "text-embedding-v4")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取嵌入向量错误: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        将文本分块
        
        Args:
            text: 输入文本
            chunk_size: 每块大小（字符数）
            overlap: 重叠大小
            
        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def retrieve_relevant_chunks(self, query: str, text: str, top_k: int = 5) -> List[str]:
        """
        检索与查询相关的文本块
        
        Args:
            query: 查询文本
            text: 文档文本
            top_k: 返回前k个最相关的块
            
        Returns:
            相关文本块列表
        """
        try:
            # 获取查询的嵌入向量
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                # 如果嵌入失败，使用简单的关键词匹配
                return self._simple_keyword_search(query, text, top_k)
            
            # 分块文本
            chunks = self.chunk_text(text, chunk_size=1000, overlap=200)
            
            # 计算每个块的相似度
            chunk_scores = []
            for chunk in chunks:
                chunk_embedding = await self.get_embedding(chunk)
                if chunk_embedding:
                    similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                    chunk_scores.append((similarity, chunk))
            
            # 按相似度排序并返回top_k
            chunk_scores.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in chunk_scores[:top_k]]
            
        except Exception as e:
            print(f"检索相关块错误: {e}，使用简单搜索")
            return self._simple_keyword_search(query, text, top_k)
    
    def _simple_keyword_search(self, query: str, text: str, top_k: int) -> List[str]:
        """简单的关键词搜索（fallback）"""
        query_words = set(query.lower().split())
        chunks = self.chunk_text(text, chunk_size=1000, overlap=200)
        
        chunk_scores = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            chunk_scores.append((score, chunk))
        
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in chunk_scores[:top_k]]


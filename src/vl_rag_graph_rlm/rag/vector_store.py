"""Milvus vector store with hybrid search (dense + keyword + RRF fusion).

Based on Paddle-ERNIE-RAG implementation with support for:
- Dense vector search
- Keyword search with jieba segmentation
- Reciprocal Rank Fusion (RRF)
"""

import os
import re
import random
import logging
from typing import List, Dict, Any, Optional

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

from vl_rag_graph_rlm.rag.ernie_client import ERNIEClient

logger = logging.getLogger("vl_rag_graph_rlm.vector_store")


class MilvusVectorStore:
    """Milvus vector store with hybrid search capabilities.
    
    Supports dense vector search, keyword search with RRF fusion.
    Works with both Milvus server and Milvus Lite (local .db files).
    
    Example:
        >>> store = MilvusVectorStore(
        ...     uri="./demo_data.db",
        ...     token="",
        ...     collection_name="my_docs"
        ... )
        >>> store.insert_documents([{"content": "...", "filename": "doc.pdf", "page": 1}])
        >>> results = store.search("query", top_k=10)
    """
    
    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: str,
        embedding_client: Optional[ERNIEClient] = None,
        embedding_service_url: Optional[str] = None,
        qianfan_api_key: Optional[str] = None,
        embedding_dim: int = 384
    ):
        if not HAS_MILVUS:
            raise ImportError("pymilvus not installed. Install with: pip install pymilvus")
        
        self.collection_name = collection_name
        self.uri = uri
        self.token = token
        self.embedding_dim = embedding_dim
        
        # Use provided client or create default
        if embedding_client:
            self.embedding_client = embedding_client
        else:
            self.embedding_client = ERNIEClient(
                embed_api_base=embedding_service_url,
                embed_api_key=qianfan_api_key
            )
        
        self._connect_milvus()
        self._init_collection()
    
    def _connect_milvus(self):
        """Connect to Milvus server or Lite."""
        try:
            if connections.has_connection("default"):
                return
            
            if self.uri.endswith(".db"):
                logger.info(f"Connecting to Milvus Lite: {self.uri}")
                connections.connect("default", uri=self.uri)
            else:
                logger.info(f"Connecting to Milvus server: {self.uri}")
                connections.connect("default", uri=self.uri, token=self.token)
        except Exception as e:
            logger.error(f"Milvus connection failed: {e}")
            # Fallback to local db
            if not self.uri.endswith(".db") and not connections.has_connection("default"):
                try:
                    connections.connect("default", uri="./demo_data.db")
                except:
                    pass
    
    def _init_collection(self):
        """Initialize collection with schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        schema = CollectionSchema(fields, "PDF QA Collection")
        
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name, schema)
            index_params = {
                "metric_type": "L2",
                "index_type": "FLAT",
                "params": {}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Created new collection (FLAT index): {self.collection_name}")
        else:
            self.collection = Collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        
        self.collection.load()
    
    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for texts."""
        if not texts:
            return []
        try:
            if hasattr(self.embedding_client, 'get_embeddings'):
                return self.embedding_client.get_embeddings(texts)
            else:
                return [self.embedding_client.get_embedding(t) for t in texts]
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int = 50, expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """Keyword search with jieba segmentation."""
        results = []
        try:
            stop_words = {
                "的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或",
                "一个", "没有", "我们", "你们", "他们", "它", "解释", "是什么",
                "含义", "文章", "图片", "这个", "篇", "请问", "以及", "什么",
                "如何", "怎么", "为什么", "分析", "介绍", "描述",
                "what", "is", "the", "of", "in", "and", "to", "a", "an", "are",
                "explain", "describe", "tell", "me", "about", "how", "why", "paper", "article"
            }
            
            keywords = []
            try:
                import jieba
                words = jieba.cut_for_search(query)
                for w in words:
                    w = w.strip()
                    if len(w) > 1 and w.lower() not in stop_words:
                        keywords.append(w)
            except ImportError:
                clean_query = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", query)
                keywords = [w for w in clean_query.split() if w.lower() not in stop_words and len(w) > 1]
            
            if not keywords:
                return []
            keywords = list(set(keywords))
            
            zh_keywords = []
            en_keywords = []
            
            for k in keywords:
                if any('\u4e00' <= char <= '\u9fff' for char in k):
                    zh_keywords.append(k)
                else:
                    en_keywords.append(k)
            
            final_parts = []
            for k in zh_keywords[:5]:
                final_parts.append(f'content like "%{k}%"')
            for k in en_keywords[:5]:
                final_parts.append(f'content like "%{k}%"')
            
            if not final_parts:
                return []
            
            keyword_expr = " || ".join(final_parts)
            
            if expr:
                final_milvus_expr = f"({expr}) and ({keyword_expr})"
            else:
                final_milvus_expr = keyword_expr
            
            res = self.collection.query(
                expr=final_milvus_expr,
                output_fields=["id", "filename", "page", "content", "chunk_id"],
                limit=top_k
            )
            
            for hit in res:
                results.append({
                    "content": hit.get("content"),
                    "filename": hit.get("filename"),
                    "page": hit.get("page"),
                    "chunk_id": hit.get("chunk_id"),
                    "semantic_score": 0.0,
                    "raw_score": 100.0,
                    "type": "keyword",
                    "id": hit.get("id")
                })
        except Exception as e:
            logger.warning(f"Keyword search skipped: {e}")
        
        return results
    
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Hybrid search: dense + keyword + RRF fusion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional options including 'expr' for filtering
            
        Returns:
            List of search results with content, metadata, and scores
        """
        expr = kwargs.get('expr', None)
        
        # Dense vector search
        dense_results = []
        try:
            query_vector = self.embedding_client.get_embedding(query)
            if query_vector:
                search_params = {"metric_type": "L2", "params": {}}
                
                milvus_res = self.collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k * 5,
                    expr=expr,
                    output_fields=["filename", "page", "content", "chunk_id"]
                )
                
                for hit in milvus_res[0]:
                    raw_score = 1.0 / (1.0 + hit.distance) * 100
                    dense_results.append({
                        "content": hit.entity.get("content"),
                        "filename": hit.entity.get("filename"),
                        "page": hit.entity.get("page"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "semantic_score": hit.distance,
                        "raw_score": raw_score,
                        "type": "dense",
                        "id": hit.id
                    })
        except Exception as e:
            logger.error(f"Vector search error: {e}")
        
        # Keyword search
        keyword_results = self._keyword_search(query, top_k=top_k * 5, expr=expr)
        
        # RRF fusion
        rank_dict = {}
        
        def apply_rrf(results_list, k=60, weight=1.0):
            for rank, item in enumerate(results_list):
                doc_id = item.get('id') or item.get('chunk_id')
                if doc_id not in rank_dict:
                    rank_dict[doc_id] = {"data": item, "score": 0.0}
                rank_dict[doc_id]["score"] += weight * (1.0 / (k + rank))
        
        apply_rrf(dense_results, weight=4.0)
        apply_rrf(keyword_results, weight=1.0)
        
        # Sort and return
        sorted_docs = sorted(rank_dict.values(), key=lambda x: x['score'], reverse=True)
        final_results = [item['data'] for item in sorted_docs[:top_k * 2]]
        
        logger.info(f"Hybrid search: dense={len(dense_results)}, keyword={len(keyword_results)}, fused={len(final_results)}")
        return final_results
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert documents with embeddings.
        
        Args:
            documents: List of dicts with 'content', 'filename', 'page', 'chunk_id'
        """
        if not documents:
            return
        
        logger.info(f"Requesting embeddings for {len(documents)} documents...")
        texts = [doc['content'] for doc in documents]
        embeddings = self.get_embeddings(texts)
        
        valid_docs, valid_vectors = [], []
        failed_count = 0
        
        for i, emb in enumerate(embeddings):
            if emb and len(emb) == self.embedding_dim:
                valid_docs.append(documents[i])
                valid_vectors.append(emb)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Warning: {failed_count} embeddings failed")
        
        if not valid_docs:
            logger.error("Critical error: all embeddings failed, no data inserted!")
            return
        
        try:
            data = [
                [doc['filename'] for doc in valid_docs],
                [doc['page'] for doc in valid_docs],
                [doc['chunk_id'] for doc in valid_docs],
                [doc['content'] for doc in valid_docs],
                valid_vectors
            ]
            self.collection.insert(data)
            self.collection.flush()
            logger.info(f"Successfully inserted {len(valid_vectors)} documents")
        except Exception as e:
            logger.error(f"Milvus insert error: {e}")
    
    def delete_document(self, filename: str) -> str:
        """Delete all chunks for a filename.
        
        Args:
            filename: Document filename to delete
            
        Returns:
            Status message
        """
        if not filename:
            return "Error: empty filename"
        try:
            self.collection.delete(expr=f'filename == "{filename}"')
            self.collection.flush()
            logger.info(f"Deleted document: {filename}")
            return f"Successfully deleted: {filename}"
        except Exception as e:
            err_msg = f"Delete failed: {e}"
            logger.error(err_msg)
            return err_msg
    
    def list_documents(self) -> List[str]:
        """List all unique filenames in collection."""
        try:
            res = self.collection.query(expr="id > 0", output_fields=["filename"], limit=16384)
            return sorted(list(set([r['filename'] for r in res])))
        except:
            return []
    
    def get_document_content(self, filename: str) -> str:
        """Get full content of a document."""
        try:
            res = self.collection.query(
                expr=f"filename == '{filename}'",
                output_fields=["content", "page"],
                limit=1000
            )
            res.sort(key=lambda x: x['page'])
            return "\n\n".join([r['content'] for r in res])
        except:
            return ""
    
    def test_self_recall(self, sample_size: int = 20) -> str:
        """Test self-recall rate."""
        try:
            total = self.collection.num_entities
            if total == 0:
                return "Error: collection is empty"
            
            limit = min(100, total)
            res = self.collection.query(expr="id > 0", output_fields=["id", "content"], limit=limit)
            if not res:
                return "Error: cannot fetch data"
            
            samples = random.sample(res, min(sample_size, len(res)))
            hits = 0
            for item in samples:
                doc_id = item['id']
                content = item['content']
                emb = self.embedding_client.get_embedding(content)
                if not emb:
                    continue
                
                search_res = self.collection.search(
                    data=[emb],
                    anns_field="embedding",
                    param={"metric_type": "L2", "params": {}},
                    limit=1,
                    output_fields=["id"]
                )
                
                if search_res and len(search_res[0]) > 0:
                    top1_id = search_res[0][0].id
                    if top1_id == doc_id:
                        hits += 1
            
            recall_rate = (hits / len(samples)) * 100
            return f"Recall test ({len(samples)} samples): accuracy {recall_rate:.1f}%"
            
        except Exception as e:
            return f"Test error: {e}"

"""
RAG - Retrieval Augmented Generation モジュール

長期記憶が必要なシナリオで使用するRAGモジュール。
ElasticSearch 8.+ または ChromaDBをバックエンドに選択可能。
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from ...services.embedding_service import EmbeddingService # Added import

class RAGBackend:
    """RAGのバックエンドインターフェース"""
    def __init__(self):
        pass
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """ドキュメントを追加"""
        raise NotImplementedError
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """クエリに基づいて検索"""
        raise NotImplementedError
    
    def delete_document(self, doc_id: str) -> bool:
        """ドキュメントを削除"""
        raise NotImplementedError


class ElasticSearchBackend(RAGBackend):
    """ElasticSearchをバックエンドとするRAG実装"""
    def __init__(self, es_url: str, index_name: str = "slm_rag"):
        """
        ElasticSearchバックエンドの初期化
        
        Parameters:
        -----------
        es_url: ElasticSearchのURL
        index_name: インデックス名
        """
        super().__init__()
        self.es_url = es_url
        self.index_name = index_name
        self.client = None
        self.embedding_service = EmbeddingService() # Instantiated EmbeddingService
        self._initialize()
    
    def _initialize(self):
        """ElasticSearchクライアントの初期化"""
        try:
            from elasticsearch import Elasticsearch
            self.client = Elasticsearch(self.es_url)
            
            # インデックスが存在しない場合は作成
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "text": {"type": "text"},
                                "embedding": {"type": "dense_vector", "dims": 384},
                                "metadata": {"type": "object"}
                            }
                        }
                    }
                )
            
            print(f"ElasticSearch backend initialized: {self.es_url}, index: {self.index_name}")
        except Exception as e:
            print(f"Error initializing ElasticSearch: {e}")
            raise
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        ドキュメントを追加
        
        Parameters:
        -----------
        text: ドキュメントテキスト
        metadata: メタデータ
        
        Returns:
        --------
        ドキュメントID
        """
        try:
            # Generate embedding using the service
            embedding = self.embedding_service.generate_embedding(text)
            
            # ドキュメントを追加
            response = self.client.index(
                index=self.index_name,
                document={
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata or {}
                }
            )
            
            return response["_id"]
        except Exception as e:
            print(f"Error adding document: {e}")
            return ""
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        クエリに基づいて検索
        
        Parameters:
        -----------
        query: 検索クエリ
        top_k: 返す結果の数
          Returns:
        --------
        検索結果のリスト
        """
        try:
            # Generate query embedding using the service
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # ベクトル検索
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    }
                }
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"]
                })
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        ドキュメントを削除
        
        Parameters:
        -----------
        doc_id: ドキュメントID
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            response = self.client.delete(
                index=self.index_name,
                id=doc_id            )
            return response["result"] == "deleted"
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False


class ChromaDBBackend(RAGBackend):
    """ChromaDBをバックエンドとするRAG実装"""
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        ChromaDBバックエンドの初期化
        
        Parameters:
        -----------
        persist_directory: データ永続化ディレクトリ
        """
        super().__init__()
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_service = EmbeddingService() # Instantiated EmbeddingService
        self._initialize()
    
    def _initialize(self):
        """ChromaDBクライアントの初期化"""
        try:
            import chromadb
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # クライアントの初期化
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # コレクションの取得または作成
            self.collection = self.client.get_or_create_collection(
                name="slm_rag",
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"ChromaDB backend initialized: {self.persist_directory}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        ドキュメントを追加
        
        Parameters:
        -----------
        text: ドキュメントテキスト
        metadata: メタデータ
        
        Returns:
        --------
        ドキュメントID
        """
        try:
            # ドキュメントIDを生成
            import uuid
            doc_id = str(uuid.uuid4())

            # Generate embedding using the service
            embedding = self.embedding_service.generate_embedding(text)
            
            # ドキュメントを追加
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )
            
            return doc_id
        except Exception as e:
            print(f"Error adding document: {e}")
            return ""
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        クエリに基づいて検索
        
        Parameters:
        -----------        query: 検索クエリ
        top_k: 返す結果の数
        
        Returns:
        --------
        検索結果のリスト
        """
        try:
            # Generate query embedding using the service
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # ベクトル検索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            formatted_results = []
            for i, (doc_id, text, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                formatted_results.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "score": 1.0 - distance  # コサイン距離を類似度スコアに変換
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        ドキュメントを削除
        
        Parameters:
        -----------
        doc_id: ドキュメントID
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False


class RAGAgent:
    """RAGを使用するエージェント"""
    def __init__(self, backend_type: str = "chroma", 
                 backend_config: Dict[str, Any] = None,
                 model_path: str = None):
        """
        RAGAgentの初期化
        
        Parameters:
        -----------
        backend_type: バックエンドタイプ ("elastic" または "chroma")
        backend_config: バックエンド設定
        model_path: モデルパス
        """
        self.backend_type = backend_type
        self.backend_config = backend_config or {}
        self.model_path = model_path
        self.backend = None
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """バックエンドとモデルの初期化"""
        # バックエンドの初期化
        if self.backend_type == "elastic":
            es_url = self.backend_config.get("es_url", "http://localhost:9200")
            index_name = self.backend_config.get("index_name", "slm_rag")
            self.backend = ElasticSearchBackend(es_url, index_name)
        else:
            persist_directory = self.backend_config.get("persist_directory", "./chroma_db")
            self.backend = ChromaDBBackend(persist_directory)
        
        # モデルの初期化（必要に応じて）
        if self.model_path:
            from ..agents.core import LLM
            self.model = LLM(self.model_path)
    
    async def generate_with_rag(self, query: str, top_k: int = 3) -> str:
        """
        RAGを使用してテキスト生成
        
        Parameters:
        -----------
        query: 入力クエリ
        top_k: 検索結果の数
        
        Returns:
        --------
        生成されたテキスト
        """
        # 関連ドキュメントを検索
        results = self.backend.search(query, top_k)
        
        # 検索結果をコンテキストとして結合
        context = "\n\n".join([f"Document {i+1}: {result['text']}" 
                              for i, result in enumerate(results)])
        
        # プロンプトを構築
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # モデルを使用して回答を生成
        if self.model:
            response = self.model.generate(prompt, max_tokens=100)
            return response
        else:
            return "Model not initialized"
    
    def add_to_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        テキストをメモリに追加
        
        Parameters:
        -----------
        text: テキスト
        metadata: メタデータ
        
        Returns:
        --------
        ドキュメントID
        """
        return self.backend.add_document(text, metadata)

"""
BlackBoard: Shared memory for agents, supporting different backends (SQLite, Redis)
and managing a topic summary. The summary previously involved TF-IDF like embeddings,
which have now been replaced with zero vectors.
"""

import asyncio
# import json # Removed unused import
# import time # Removed unused import
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .db_sqlite import SQLiteBackend
from .db_redis import RedisBackend

class BlackBoard:
    """
    Provides shared memory for agents (エージェント間の共有メモリを提供するブラックボード).
    Uses SQLiteBackend for local mode and RedisBackend for distributed mode.
    Manages messages and a topic summary (currently with zero vector embeddings).
    """
    def __init__(self, mode: str = "local", redis_url: Optional[str] = None, backend: Optional[Any] = None):
        """
        Initializes the BlackBoard. (BlackBoardの初期化)
        
        Parameters:
        -----------
        mode: str
            動作モード ("local" または "distributed") - Operating mode.
        redis_url: Optional[str]
            Redisサーバーの接続URL (分散モード時のみ使用) - Redis server URL (used in distributed mode only).
        backend: Optional[Any]
            外部から提供されるバックエンド（テスト用）- External backend for testing.
        """
        self.mode = mode
        self.redis_url = redis_url
        self.backend: Any = backend # Type set to Any to accommodate different backend types
        self.summary_vec: np.ndarray = np.zeros(384, dtype=np.float32)  # Default embedding dimension and type

        if backend is None:
            self._initialize()
    
    def _initialize(self):
        """Initializes the backend storage (SQLite or Redis). (バックエンドの初期化)"""
        if self.mode == "local":
            self.backend = SQLiteBackend(":memory:") # Use in-memory SQLite for local mode
        elif self.mode == "distributed":
            if not self.redis_url:
                raise ValueError("Redis URL must be provided for distributed mode.")
            self.backend = RedisBackend(self.redis_url)
        else:
            raise ValueError(f"Unknown BlackBoard mode: {self.mode}")
    
    async def push(self, message: Union[Dict[str, Any], int, str], text: Optional[str] = None) -> bool:
        """
        Adds a message to the blackboard. (メッセージをブラックボードに追加)
        
        Parameters:
        -----------
        message: Union[Dict[str, Any], int, str]
            メッセージデータ（辞書）または従来のagent_id - Message data (dictionary) or legacy agent_id.
        text: Optional[str]
            メッセージテキスト（messageが辞書でない場合に使用）- Message text (used if 'message' is not a dict).
        
        Returns:
        --------
        bool: 成功したかどうか - True if successful, False otherwise.
        """
        try:
            if isinstance(message, dict):
                agent_id = message.get("agent_id", 0) # Default agent_id if not provided
                msg_text = message.get("text", "")
                result = self.backend.push_message(agent_id, msg_text, message)
            else: # Legacy format
                agent_id = message # Here, message is agent_id
                msg_text = text if text is not None else ""
                result = self.backend.push_message(agent_id, msg_text) # Pass full message structure if needed by backend
            
            # Schedule summary update (non-blocking)
            asyncio.create_task(self._update_summary())
            return result
        except Exception as e:
            print(f"Error pushing to BlackBoard: {e}")
            return False
    
    async def pull(self, k: int = 16) -> List[str]:
        """
        Retrieves the latest k messages (text content only). (最新のk件のメッセージを取得)
        
        Parameters:
        -----------
        k: int
            取得するメッセージ数 - Number of messages to retrieve.
        
        Returns:
        --------
        List[str]: メッセージのリスト（文字列形式）- List of messages (text content).
        """
        try:
            raw_messages = self.backend.pull_messages(k)
            text_messages: List[str] = []
            for msg_data in raw_messages: # Renamed msg to msg_data
                if isinstance(msg_data, dict):
                    text_messages.append(str(msg_data.get('text', '')))
                else:
                    text_messages.append(str(msg_data)) # Should ideally be dicts from backend
            return text_messages
        except Exception as e:
            print(f"Error pulling from BlackBoard: {e}")
            return []
    
    async def pull_messages_raw(self, k: int = 16) -> List[Dict[str, Any]]:
        """
        Retrieves the latest k messages as raw dictionaries. (最新のk件のメッセージを辞書形式で取得)
        Used by components like the dashboard that need full message data.

        Parameters:
        -----------
        k: int
            取得するメッセージ数 - Number of messages to retrieve.
        
        Returns:
        --------
        List[Dict[str, Any]]: メッセージの辞書リスト - List of message dictionaries.
        """
        try:
            return self.backend.pull_messages(k) # Backend should return List[Dict[str, Any]]
        except Exception as e:
            print(f"Error pulling raw messages from BlackBoard: {e}")
            return []
    
    async def _update_summary(self, window: int = 64):
        """
        Updates the topic summary. (トピックサマリーを更新)
        The summary consists of a concatenated text snippet from recent messages
        and a zero vector representation (as actual embedding is disabled).
        
        Parameters:
        -----------
        window: int
            The number of recent messages (text content) to consider for the textual part of the summary.
        """
        try:
            messages_texts = await self.pull(window) # Gets list of strings
            if not messages_texts:
                return
            
            concatenated_text = " ".join(messages_texts)
            
            # Embedding generation removed, using zero vector.
            vector_dim = self.summary_vec.shape[0] if self.summary_vec is not None else 384
            current_dtype = self.summary_vec.dtype if self.summary_vec is not None else np.float32
            vector = np.zeros(vector_dim, dtype=current_dtype)
            
            # Save the textual part of the summary (first 200 chars) and the zero vector.
            self.backend.save_summary(concatenated_text[:200], vector)
            self.summary_vec = vector # Update in-memory summary vector
            
        except Exception as e:
            print(f"Error updating summary: {e}")
    
    async def set_param(self, key: str, value: Any) -> bool:
        """
        Sets a parameter in the backend's key-value store. (KVストアにパラメータを設定)
        
        Parameters:
        -----------
        key: str
            キー - The key for the parameter.
        value: Any
            値 - The value of the parameter.
        
        Returns:
        --------
        bool: 成功したかどうか - True if successful, False otherwise.
        """
        try:
            return self.backend.set_param(key, value)
        except Exception as e:
            print(f"Error setting parameter '{key}': {e}")
            return False
    
    async def get_param(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a parameter from the backend's key-value store. (KVストアからパラメータを取得)
        
        Parameters:
        -----------
        key: str
            キー - The key for the parameter.
        default: Any
            デフォルト値 - Default value to return if the key is not found.
        
        Returns:
        --------
        Any: 値 - The value of the parameter or the default.
        """
        print(f"[METRICS_DEBUG] BlackBoard.get_param called for key: {key}") # Adding requested debug print
        try:
            value = self.backend.get_param(key, default)
            print(f"[METRICS_DEBUG] BlackBoard.get_param for '{key}' returned: {value}") # Also log returned value
            return value
        except Exception as e:
            print(f"Error getting parameter '{key}': {e}")
            return default
    
    async def update_summary(self, summary_text: str, vector: np.ndarray) -> bool: # Renamed summary to summary_text
        """
        Directly updates the topic summary text and vector. (トピックサマリーを直接更新)
        
        Parameters:
        -----------
        summary_text: str
            要約テキスト - The new summary text.
        vector: np.ndarray
            埋め込みベクトル - The new embedding vector.
        
        Returns:
        --------
        bool: 成功したかどうか - True if successful, False otherwise.
        """
        try:
            result = self.backend.save_summary(summary_text, vector)
            if result:
                self.summary_vec = vector.copy() # Update in-memory vector if save was successful
            return result
        except Exception as e:
            print(f"Error updating summary directly: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """
        Clears all data from the blackboard (messages, summary, params). (すべてのデータをクリア)
        
        Returns:
        --------
        bool: 成功したかどうか - True if successful, False otherwise.
        """
        try:
            result = self.backend.clear_all()
            if result:
                # Reset in-memory summary vector as well
                self.summary_vec = np.zeros_like(self.summary_vec)
            return result
        except Exception as e:
            print(f"Error clearing BlackBoard: {e}")
            return False
    
    def get_topic_summary(self, max_words: int = 30) -> str:
        """
        Retrieves the current textual topic summary. (現在のトピックサマリーを取得)
        If no summary is stored, it generates a simple one from recent messages.

        Args:
            max_words (int): The maximum number of words for the returned summary.

        Returns:
            str: The topic summary string, potentially truncated.
        """
        summary_text_to_return = "" # Renamed to avoid conflict
        try:
            summary_info = self.backend.get_latest_summary() # This should return {'summary': str, 'vector': np.ndarray}
            
            if summary_info and "summary" in summary_info and summary_info["summary"]:
                summary_text_to_return = summary_info["summary"]
            else:
                # Fallback: generate a simple summary from the last 10 messages (text only)
                raw_messages_texts = asyncio.run(self.pull(k=10)) # pull returns List[str]
                if not raw_messages_texts:
                    return "" # No messages, no summary
                summary_text_to_return = " ".join(raw_messages_texts)

            # Truncate the summary to max_words
            words = summary_text_to_return.split()
            if len(words) > max_words:
                concise_summary = " ".join(words[:max_words]) + "..."
            else:
                concise_summary = " ".join(words)
            
            return concise_summary.strip()
            
        except Exception as e:
            print(f"Error getting topic summary: {e}")
            return ""


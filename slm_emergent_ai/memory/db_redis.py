"""
Redis Database Backend - BlackBoardのRedisバックエンド実装

BlackBoardの分散モード用Redisバックエンド実装。
メッセージログ、KVストア、トピックサマリーを管理する。
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
import numpy as np

class RedisBackend:
    """
    BlackBoardのRedisバックエンド
    
    メッセージログ、KVストア、トピックサマリーをRedisデータベースで管理する
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        RedisBackendの初期化
        
        Parameters:
        -----------
        redis_url: RedisサーバーのURL
        """
        self.redis_url = redis_url
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Redisクライアントの初期化"""
        try:
            import redis
            self.client = redis.Redis.from_url(self.redis_url)
            # 接続テスト
            self.client.ping()
            print(f"Redis backend initialized: {self.redis_url}")
        except Exception as e:
            print(f"Error initializing Redis backend: {e}")
            raise
    
    def close(self):
        """Redis接続を閉じる"""
        if self.client:
            self.client.close()
    
    def push_message(self, agent_id: int, text: str) -> bool:
        """
        メッセージをRedisに追加
        
        Parameters:
        -----------
        agent_id: エージェントID
        text: メッセージテキスト
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            timestamp = time.time()
            message = json.dumps({
                "agent_id": agent_id,
                "text": text,
                "timestamp": timestamp
            })
            self.client.lpush("messages", message)
            return True
        except Exception as e:
            print(f"Error pushing message to Redis: {e}")
            return False
    
    def pull_messages(self, k: int = 16) -> List[str]:
        """
        最新のk件のメッセージを取得
        
        Parameters:
        -----------
        k: 取得するメッセージ数
        
        Returns:
        --------
        メッセージのリスト
        """
        try:
            messages = self.client.lrange("messages", 0, k-1)
            return [json.loads(msg.decode())["text"] for msg in messages]
        except Exception as e:
            print(f"Error pulling messages from Redis: {e}")
            return []
    
    def set_param(self, key: str, value: Any) -> bool:
        """
        KVストアにパラメータを設定
        
        Parameters:
        -----------
        key: キー
        value: 値
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            value_str = json.dumps(value)
            self.client.set(f"kv:{key}", value_str)
            return True
        except Exception as e:
            print(f"Error setting parameter in Redis: {e}")
            return False
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        KVストアからパラメータを取得
        
        Parameters:
        -----------
        key: キー
        default: デフォルト値
        
        Returns:
        --------
        値
        """
        try:
            value = self.client.get(f"kv:{key}")
            if value:
                return json.loads(value.decode())
            return default
        except Exception as e:
            print(f"Error getting parameter from Redis: {e}")
            return default
    
    def save_summary(self, summary: str, vector: np.ndarray) -> bool:
        """
        トピックサマリーを保存
        
        Parameters:
        -----------
        summary: サマリーテキスト
        vector: 埋め込みベクトル
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            timestamp = time.time()
            summary_data = {
                "summary": summary,
                "vector": vector.tolist(),
                "timestamp": timestamp
            }
            self.client.set("topic_summary", json.dumps(summary_data))
            return True
        except Exception as e:
            print(f"Error saving summary to Redis: {e}")
            return False
    
    def get_latest_summary(self) -> Dict[str, Any]:
        """
        最新のトピックサマリーを取得
        
        Returns:
        --------
        サマリー情報
        """
        try:
            summary_data = self.client.get("topic_summary")
            if summary_data:
                data = json.loads(summary_data.decode())
                return {
                    "summary": data["summary"],
                    "vector": np.array(data["vector"]),
                    "timestamp": data["timestamp"]
                }
            return {
                "summary": "",
                "vector": np.zeros(384),
                "timestamp": 0.0
            }
        except Exception as e:
            print(f"Error getting summary from Redis: {e}")
            return {
                "summary": "",
                "vector": np.zeros(384),
                "timestamp": 0.0
            }
    
    def clear_all(self) -> bool:
        """
        すべてのデータを削除
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            self.client.delete("messages")
            keys = self.client.keys("kv:*")
            if keys:
                self.client.delete(*keys)
            self.client.delete("topic_summary")
            return True
        except Exception as e:
            print(f"Error clearing data from Redis: {e}")
            return False
    
    def publish_update(self, channel: str, data: Dict[str, Any]) -> bool:
        """
        更新をパブリッシュ
        
        Parameters:
        -----------
        channel: チャンネル名
        data: パブリッシュするデータ
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            self.client.publish(channel, json.dumps(data))
            return True
        except Exception as e:
            print(f"Error publishing update to Redis: {e}")
            return False
    
    def subscribe(self, channels: List[str]):
        """
        チャンネルをサブスクライブ
        
        Parameters:
        -----------
        channels: チャンネル名のリスト
        
        Returns:
        --------
        Redisのpubsubオブジェクト
        """
        try:
            pubsub = self.client.pubsub()
            pubsub.subscribe(*channels)
            return pubsub
        except Exception as e:
            print(f"Error subscribing to Redis channels: {e}")
            return None

"""
BlackBoardの拡張 - トピックサマリー取得メソッドの追加

プロンプトエンジニアリング方式のBoids制御に必要な
トピックサマリー取得メソッドを追加
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
import numpy as np

from .db_sqlite import SQLiteBackend
from .db_redis import RedisBackend

class BlackBoard:
    """
    エージェント間の共有メモリを提供するブラックボード
    
    ローカルモードではSQLiteBackendを使用し、分散モードではRedisBackendを使用する
    """
    def __init__(self, mode: str = "local", redis_url: Optional[str] = None, backend=None):
        """
        BlackBoardの初期化
        
        Parameters:
        -----------
        mode: 動作モード ("local" または "distributed")
        redis_url: Redisサーバーの接続URL (分散モード時のみ使用)
        backend: 外部から提供されるバックエンド（テスト用）
        """
        self.mode = mode
        self.redis_url = redis_url
        self.backend = backend
        self.summary_vec = np.zeros(384)  # 埋め込みベクトルの次元数
        if backend is None:
            self._initialize()
    
    def _initialize(self):
        """バックエンドの初期化"""
        if self.mode == "local":
            # ローカルモード: SQLiteBackendを使用
            self.backend = SQLiteBackend(":memory:")
        else:
            # 分散モード: RedisBackendを使用
            self.backend = RedisBackend(self.redis_url)
    
    async def push(self, agent_id: int, text: str) -> bool:
        """
        メッセージをブラックボードに追加
        
        Parameters:
        -----------
        agent_id: エージェントID
        text: メッセージテキスト
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            result = self.backend.push_message(agent_id, text)
            
            # 非同期で要約を更新
            asyncio.create_task(self._update_summary())
            
            return result
        except Exception as e:
            print(f"Error pushing to BlackBoard: {e}")
            return False
    
    async def pull(self, k: int = 16) -> List[str]:
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
            return self.backend.pull_messages(k)
        except Exception as e:
            print(f"Error pulling from BlackBoard: {e}")
            return []
    
    async def _update_summary(self, method: str = "minilm", window: int = 64):
        """
        トピックサマリーを更新
        
        Parameters:
        -----------
        method: 要約に使用するモデル
        window: 要約対象の最新メッセージ数
        """
        try:
            # 最新のwindow件のメッセージを取得
            messages = await self.pull(window)
            if not messages:
                return
            
            # メッセージを結合
            text = " ".join(messages)
            
            # 実際の実装では、sentence-transformersなどを使用して埋め込みベクトルを計算
            # ここではダミー実装
            vector = np.random.randn(384)
            # 正規化
            vector = vector / (np.linalg.norm(vector) + 1e-8)
            
            # 要約を保存
            self.backend.save_summary(text[:200], vector)
            
            # 要約ベクトルを更新
            self.summary_vec = vector
            
        except Exception as e:
            print(f"Error updating summary: {e}")
    
    async def set_param(self, key: str, value: Any) -> bool:
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
            return self.backend.set_param(key, value)
        except Exception as e:
            print(f"Error setting parameter: {e}")
            return False
    
    async def get_param(self, key: str, default: Any = None) -> Any:
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
            return self.backend.get_param(key, default)
        except Exception as e:
            print(f"Error getting parameter: {e}")
            return default
    
    async def update_summary(self, summary: str, vector: np.ndarray) -> bool:
        """
        トピックサマリーを直接更新
        
        Parameters:
        -----------
        summary: 要約テキスト
        vector: 埋め込みベクトル
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            result = self.backend.save_summary(summary, vector)
            
            # 要約ベクトルを更新
            self.summary_vec = vector.copy()
            
            return result
        except Exception as e:
            print(f"Error updating summary: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """
        すべてのデータをクリア
        
        Returns:
        --------
        成功したかどうか
        """
        try:
            result = self.backend.clear_all()
            
            # 要約ベクトルをリセット
            self.summary_vec = np.zeros(384)
            
            return result
        except Exception as e:
            print(f"Error clearing BlackBoard: {e}")
            return False
    
    def get_topic_summary(self) -> str:
        """
        現在のトピックサマリーを取得（プロンプトエンジニアリング用）
        
        Returns:
        --------
        トピックサマリー文字列
        """
        try:
            # 最新のサマリー情報を取得
            summary_info = self.backend.get_latest_summary()
            
            if summary_info and "summary" in summary_info:
                return summary_info["summary"]
            
            # サマリーがない場合は、最新のメッセージから簡易的に生成
            messages = self.backend.pull_messages(10)  # 最新の10件のメッセージを取得
            
            if not messages:
                return ""
            
            # 簡易的な要約（実際の実装ではより高度な要約技術を使用）
            words = " ".join(messages).split()
            if len(words) > 30:
                summary = " ".join(words[:30]) + "..."
            else:
                summary = " ".join(words)
            
            return summary
            
        except Exception as e:
            print(f"Error getting topic summary: {e}")
            return ""

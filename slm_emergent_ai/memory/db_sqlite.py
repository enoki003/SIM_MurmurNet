"""
SQLite Database Backend - BlackBoardのSQLiteバックエンド実装

BlackBoardのローカルモード用SQLiteバックエンド実装。
メッセージログ、KVストア、トピックサマリーを管理する。
"""

import sqlite3
import json
import time
from typing import Dict, List, Any, Optional, Union
import numpy as np

class SQLiteBackend:
    """
    BlackBoardのSQLiteバックエンド
    
    メッセージログ、KVストア、トピックサマリーをSQLiteデータベースで管理する
    """
    def __init__(self, db_path: str = ":memory:"):
        """
        SQLiteBackendの初期化
        
        Parameters:
        -----------
        db_path: データベースファイルのパス (":memory:"はメモリ内データベース)
        """
        self.db_path = db_path
        self.conn = None
        self._initialize()
    
    def _initialize(self):
        """データベースの初期化"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # メッセージログテーブル
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id INTEGER,
                text TEXT,
                timestamp REAL
            )
            """)
            
            # KVストアテーブル
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """)
            
            # トピックサマリーテーブル
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT,
                vector BLOB,
                timestamp REAL
            )
            """)
            
            self.conn.commit()
            print(f"SQLite backend initialized: {self.db_path}")
        except Exception as e:
            print(f"Error initializing SQLite backend: {e}")
            raise
    
    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
    
    def push_message(self, agent_id: int, text: str) -> bool:
        """
        メッセージをデータベースに追加
        
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
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO messages (agent_id, text, timestamp) VALUES (?, ?, ?)",
                (agent_id, text, timestamp)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error pushing message to SQLite: {e}")
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
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT text FROM messages ORDER BY timestamp DESC LIMIT ?",
                (k,)
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error pulling messages from SQLite: {e}")
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
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                (key, value_str)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error setting parameter in SQLite: {e}")
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
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return default
        except Exception as e:
            print(f"Error getting parameter from SQLite: {e}")
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
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO topic_summary (summary, vector, timestamp) VALUES (?, ?, ?)",
                (summary, vector.tobytes(), timestamp)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving summary to SQLite: {e}")
            return False
    
    def get_latest_summary(self) -> Dict[str, Any]:
        """
        最新のトピックサマリーを取得
        
        Returns:
        --------
        サマリー情報
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT summary, vector, timestamp FROM topic_summary ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                return {
                    "summary": row[0],
                    "vector": np.frombuffer(row[1]),
                    "timestamp": row[2]
                }
            return {
                "summary": "",
                "vector": np.zeros(384),
                "timestamp": 0.0
            }
        except Exception as e:
            print(f"Error getting summary from SQLite: {e}")
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
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM messages")
            cursor.execute("DELETE FROM kv_store")
            cursor.execute("DELETE FROM topic_summary")
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing data from SQLite: {e}")
            return False

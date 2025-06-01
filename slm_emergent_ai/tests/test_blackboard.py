import unittest
import asyncio
import numpy as np
from slm_emergent_ai.memory.blackboard import BlackBoard
from slm_emergent_ai.memory.db_sqlite import SQLiteBackend

class TestBlackBoard(unittest.TestCase):
    """BlackBoardのテスト"""
    
    def setUp(self):
        """テスト用データの準備"""
        # インメモリSQLiteバックエンドを使用
        self.backend = SQLiteBackend(":memory:")
        self.bb = BlackBoard(backend=self.backend)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        self.backend.close()
    
    def test_init(self):
        """初期化のテスト"""
        # BlackBoardが正しく初期化されていることを確認
        self.assertIsNotNone(self.bb)
        self.assertEqual(self.bb.backend, self.backend)
    
    def test_push_pull(self):
        """プッシュとプルのテスト"""
        # テスト用のメッセージ
        agent_id = 1
        text = "テストメッセージ"
        
        # メッセージをプッシュ
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.bb.push(agent_id, text))
        
        # メッセージをプル
        messages = loop.run_until_complete(self.bb.pull(k=1))
        
        # プルしたメッセージが正しいことを確認
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], text)
    
    def test_multiple_push_pull(self):
        """複数のプッシュとプルのテスト"""
        # テスト用のメッセージ
        messages = [
            (1, "メッセージ1"),
            (2, "メッセージ2"),
            (3, "メッセージ3"),
            (1, "メッセージ4"),
            (2, "メッセージ5")
        ]
        
        loop = asyncio.get_event_loop()
        
        # メッセージをプッシュ
        for agent_id, text in messages:
            loop.run_until_complete(self.bb.push(agent_id, text))
        
        # すべてのメッセージをプル
        pulled_messages = loop.run_until_complete(self.bb.pull(k=5))
        
        # プルしたメッセージの数が正しいことを確認
        self.assertEqual(len(pulled_messages), 5)
        
        # 最新のメッセージが先頭にあることを確認
        self.assertEqual(pulled_messages[0], "メッセージ5")
        self.assertEqual(pulled_messages[4], "メッセージ1")
    
    def test_set_get_param(self):
        """パラメータの設定と取得のテスト"""
        # テスト用のパラメータ
        key = "test_key"
        value = {"a": 1, "b": "test", "c": [1, 2, 3]}
        
        loop = asyncio.get_event_loop()
        
        # パラメータを設定
        loop.run_until_complete(self.bb.set_param(key, value))
        
        # パラメータを取得
        result = loop.run_until_complete(self.bb.get_param(key))
        
        # 取得したパラメータが正しいことを確認
        self.assertEqual(result, value)
    
    def test_update_summary(self):
        """サマリー更新のテスト"""
        # テスト用のサマリーとベクトル
        summary = "テストサマリー"
        vector = np.random.randn(384)
        
        loop = asyncio.get_event_loop()
        
        # サマリーを更新
        loop.run_until_complete(self.bb.update_summary(summary, vector))
        
        # サマリーベクトルが更新されていることを確認
        self.assertTrue(np.array_equal(self.bb.summary_vec, vector))
    
    def test_clear_all(self):
        """すべてのデータをクリアするテスト"""
        # テスト用のデータを追加
        agent_id = 1
        text = "テストメッセージ"
        key = "test_key"
        value = {"test": "value"}
        summary = "テストサマリー"
        vector = np.random.randn(384)
        
        loop = asyncio.get_event_loop()
        
        # データを追加
        loop.run_until_complete(self.bb.push(agent_id, text))
        loop.run_until_complete(self.bb.set_param(key, value))
        loop.run_until_complete(self.bb.update_summary(summary, vector))
        
        # データをクリア
        loop.run_until_complete(self.bb.clear_all())
        
        # データがクリアされていることを確認
        messages = loop.run_until_complete(self.bb.pull(k=1))
        self.assertEqual(len(messages), 0)
        
        result = loop.run_until_complete(self.bb.get_param(key))
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

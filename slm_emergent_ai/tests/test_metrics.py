import unittest
import numpy as np
from slm_emergent_ai.eval.metrics import (
    calculate_entropy,
    calculate_vdi,
    calculate_fcr,
    calculate_speed,
    MetricsTracker
)

class TestMetrics(unittest.TestCase):
    """メトリクス計算のテスト"""
    
    def setUp(self):
        """テスト用データの準備"""
        # テスト用のトークンリスト
        self.tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        
        # テスト用の確率分布
        self.probs = np.array([0.1, 0.2, 0.3, 0.4])
        
        # テスト用のファクトチェック結果
        self.fact_checks = [True, False, True, True, False]
        
        # テスト用の時間データ
        self.times = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.token_counts = [0, 2, 5, 9, 14]
        
        # MetricsTrackerインスタンス
        self.tracker = MetricsTracker()
    
    def test_calculate_entropy(self):
        """エントロピー計算のテスト"""
        # エントロピーを計算
        entropy = calculate_entropy(self.probs)
        
        # エントロピーが正の値であることを確認
        self.assertGreater(entropy, 0)
        
        # 一様分布のエントロピーが最大であることを確認
        uniform_probs = np.ones(4) / 4
        uniform_entropy = calculate_entropy(uniform_probs)
        self.assertGreaterEqual(uniform_entropy, entropy)
        
        # 確率が1つに集中している場合のエントロピーが0であることを確認
        concentrated_probs = np.zeros(4)
        concentrated_probs[0] = 1.0
        concentrated_entropy = calculate_entropy(concentrated_probs)
        self.assertAlmostEqual(concentrated_entropy, 0.0)
    
    def test_calculate_vdi(self):
        """VDI計算のテスト"""
        # VDIを計算
        vdi = calculate_vdi(self.tokens)
        
        # VDIが正の値であることを確認
        self.assertGreater(vdi, 0)
        
        # すべて同じトークンの場合のVDIが最小であることを確認
        same_tokens = [1] * 10
        same_vdi = calculate_vdi(same_tokens)
        self.assertLessEqual(same_vdi, vdi)
        
        # すべて異なるトークンの場合のVDIが最大であることを確認
        diff_tokens = list(range(10))
        diff_vdi = calculate_vdi(diff_tokens)
        self.assertGreaterEqual(diff_vdi, vdi)
    
    def test_calculate_fcr(self):
        """FCR計算のテスト"""
        # FCRを計算
        fcr = calculate_fcr(self.fact_checks)
        
        # FCRが0から1の間であることを確認
        self.assertGreaterEqual(fcr, 0.0)
        self.assertLessEqual(fcr, 1.0)
        
        # FCRが正しく計算されていることを確認
        expected_fcr = sum(self.fact_checks) / len(self.fact_checks)
        self.assertAlmostEqual(fcr, expected_fcr)
        
        # すべて正しい場合のFCRが1であることを確認
        all_true = [True] * 5
        all_true_fcr = calculate_fcr(all_true)
        self.assertAlmostEqual(all_true_fcr, 1.0)
        
        # すべて間違っている場合のFCRが0であることを確認
        all_false = [False] * 5
        all_false_fcr = calculate_fcr(all_false)
        self.assertAlmostEqual(all_false_fcr, 0.0)
    
    def test_calculate_speed(self):
        """速度計算のテスト"""
        # 速度を計算
        speed = calculate_speed(self.times, self.token_counts)
        
        # 速度が正の値であることを確認
        self.assertGreater(speed, 0)
        
        # 速度が正しく計算されていることを確認
        # 最後の時間と最初の時間の差分
        time_diff = self.times[-1] - self.times[0]
        # 最後のトークン数と最初のトークン数の差分
        token_diff = self.token_counts[-1] - self.token_counts[0]
        # 速度 = トークン数の差分 / 時間の差分
        expected_speed = token_diff / time_diff
        self.assertAlmostEqual(speed, expected_speed)
    
    def test_metrics_tracker_update(self):
        """MetricsTrackerの更新テスト"""
        # メトリクスを更新
        self.tracker.update_entropy(1.5)
        self.tracker.update_vdi(0.8)
        self.tracker.update_fcr(0.7)
        self.tracker.update_speed(10.0)
        self.tracker.update_lambda('λ_a', 0.3)
        self.tracker.update_lambda('λ_c', 0.3)
        self.tracker.update_lambda('λ_s', 0.1)
        
        # 更新されたメトリクスが正しいことを確認
        self.assertEqual(self.tracker.current_metrics['entropy'], 1.5)
        self.assertEqual(self.tracker.current_metrics['vdi'], 0.8)
        self.assertEqual(self.tracker.current_metrics['fcr'], 0.7)
        self.assertEqual(self.tracker.current_metrics['speed'], 10.0)
        self.assertEqual(self.tracker.current_metrics['λ_a'], 0.3)
        self.assertEqual(self.tracker.current_metrics['λ_c'], 0.3)
        self.assertEqual(self.tracker.current_metrics['λ_s'], 0.1)
        
        # 履歴が正しく更新されていることを確認
        self.assertEqual(len(self.tracker.metrics_history['entropy']), 1)
        self.assertEqual(self.tracker.metrics_history['entropy'][0], 1.5)
    
    def test_metrics_tracker_get_history(self):
        """MetricsTrackerの履歴取得テスト"""
        # メトリクスを複数回更新
        for i in range(5):
            self.tracker.update_entropy(1.0 + i * 0.1)
            self.tracker.update_vdi(0.5 + i * 0.1)
        
        # 履歴を取得
        entropy_history = self.tracker.get_metric_history('entropy')
        vdi_history = self.tracker.get_metric_history('vdi')
        
        # 履歴が正しいことを確認
        self.assertEqual(len(entropy_history), 5)
        self.assertEqual(len(vdi_history), 5)
        self.assertAlmostEqual(entropy_history[0], 1.0)
        self.assertAlmostEqual(entropy_history[-1], 1.4)
        self.assertAlmostEqual(vdi_history[0], 0.5)
        self.assertAlmostEqual(vdi_history[-1], 0.9)
    
    def test_metrics_tracker_reset(self):
        """MetricsTrackerのリセットテスト"""
        # メトリクスを更新
        self.tracker.update_entropy(1.5)
        self.tracker.update_vdi(0.8)
        
        # リセット
        self.tracker.reset()
        
        # リセット後のメトリクスが初期値であることを確認
        self.assertEqual(self.tracker.current_metrics['entropy'], 0.0)
        self.assertEqual(self.tracker.current_metrics['vdi'], 0.0)
        
        # 履歴がクリアされていることを確認
        self.assertEqual(len(self.tracker.metrics_history['entropy']), 0)
        self.assertEqual(len(self.tracker.metrics_history['vdi']), 0)


if __name__ == '__main__':
    unittest.main()

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
        # Reset tracker before each test method if it modifies state shared across tests
        # For now, each test method that uses self.tracker initializes or resets it as needed,
        # or uses it for a single conceptual test.
        # Let's ensure clean state for tests that accumulate history or check initial values.
    
    def tearDown(self):
        # Reset the shared tracker after each test if necessary,
        # or ensure tests instantiate their own trackers if state is an issue.
        # For now, we rely on specific tracker tests to manage their state.
        pass

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
        
        # すべて同じトークンの場合のVDIが最小であることを確認 (VDI* = 1 / (log(10)+1))
        same_tokens = [1] * 10 # TotalTokens = 10, UniqueTokens = 1
        same_vdi = calculate_vdi(same_tokens)
        expected_same_vdi = 1 / (np.log(10) + 1)
        self.assertAlmostEqual(same_vdi, expected_same_vdi)
        
        # すべて異なるトークンの場合のVDIが最大であることを確認 (VDI* = 10 / (log(10)+1))
        diff_tokens = list(range(10)) # TotalTokens = 10, UniqueTokens = 10
        diff_vdi = calculate_vdi(diff_tokens)
        expected_diff_vdi = 10 / (np.log(10) + 1)
        self.assertAlmostEqual(diff_vdi, expected_diff_vdi)

    def test_calculate_vdi_empty_list(self):
        """Test VDI calculation with an empty token list."""
        vdi_empty = calculate_vdi([])
        self.assertEqual(vdi_empty, 0.0, "VDI for an empty list should be 0.0.")

    def test_calculate_vdi_window_effect(self):
        """Test VDI calculation with window_size."""
        tokens = list(range(200)) # 200 unique tokens
        # VDI with default window_size=100
        vdi_window_100 = calculate_vdi(tokens)
        expected_vdi_window_100 = 100 / (np.log(100) + 1) # Considers last 100 tokens
        self.assertAlmostEqual(vdi_window_100, expected_vdi_window_100)

        # VDI with explicit window_size=50
        vdi_window_50 = calculate_vdi(tokens, window_size=50)
        expected_vdi_window_50 = 50 / (np.log(50) + 1) # Considers last 50 tokens
        self.assertAlmostEqual(vdi_window_50, expected_vdi_window_50)

        # VDI when token count is less than window_size
        short_tokens = list(range(30))
        vdi_short = calculate_vdi(short_tokens, window_size=50)
        expected_vdi_short = 30 / (np.log(30) + 1) # Considers all 30 tokens
        self.assertAlmostEqual(vdi_short, expected_vdi_short)

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
        self.assertEqual(len(self.tracker.times), 1) # Initial time
        self.assertEqual(len(self.tracker.token_counts), 1) # Initial token count

    def test_metrics_tracker_initial_state(self):
        """Test the initial state of MetricsTracker."""
        tracker = MetricsTracker() # New instance for this test
        self.assertEqual(tracker.current_metrics['entropy'], 0.0)
        self.assertEqual(tracker.current_metrics['vdi'], 0.0)
        self.assertEqual(tracker.current_metrics['fcr'], 0.0) # Default FCR should be 0.0 before any update
        self.assertEqual(tracker.current_metrics['speed'], 0.0)
        self.assertEqual(len(tracker.times), 1)
        self.assertEqual(tracker.token_counts[0], 0)

    def test_metrics_tracker_update_fcr_empty(self):
        """Test MetricsTracker update_fcr with empty list (should be 1.0)."""
        tracker = MetricsTracker()
        # calculate_fcr([]) returns 1.0. Let's ensure tracker reflects this.
        fcr_val = calculate_fcr([])
        self.assertEqual(fcr_val, 1.0)
        tracker.update_fcr(fcr_val)
        self.assertEqual(tracker.current_metrics['fcr'], 1.0)

    def test_metrics_tracker_update_entropy_zero(self):
        """Test MetricsTracker update_entropy with 0.0."""
        tracker = MetricsTracker()
        tracker.update_entropy(0.0)
        self.assertEqual(tracker.current_metrics['entropy'], 0.0)
        self.assertEqual(tracker.metrics_history['entropy'][-1], 0.0)

    def test_metrics_tracker_update_token_count_and_speed(self):
        """Test update_token_count and its effect on speed calculation."""
        tracker = MetricsTracker() # Fresh tracker

        # Simulate token updates over time
        # Mock time.time() or manually set times list for predictable speed calculation

        # Initial state (time[0], tokens[0]=0) is set by __init__
        # Let's say initial time was 100.0
        tracker.times = [100.0]
        tracker.token_counts = [0]

        # First update
        # time.time() would be called in update_token_count. We manually append.
        tracker.update_token_count(100) # 100 tokens at time T1 (e.g. 101.0)
        tracker.times[-1] = 101.0 # Manually set time for this update for testing

        # Speed: (100-0) / (101.0-100.0) = 100 / 1.0 = 100.0 tok/s
        # calculate_speed uses times[-1] - times[0] and token_counts[-1] - token_counts[0]
        # For the first meaningful calculation, it needs at least two points.
        # After first update_token_count, we have two points in times and token_counts.
        expected_speed1 = (100 - 0) / (101.0 - 100.0)
        self.assertAlmostEqual(tracker.current_metrics['speed'], expected_speed1)

        # Second update
        tracker.update_token_count(250) # 150 more tokens (total 250) at time T2 (e.g. 102.0)
        tracker.times[-1] = 102.0 # Manually set time

        # Speed: (250-0) / (102.0-100.0) = 250 / 2.0 = 125.0 tok/s
        # This calculation is based on the total elapsed time and total tokens from the very start.
        expected_speed2 = (250 - 0) / (102.0 - 100.0)
        self.assertAlmostEqual(tracker.current_metrics['speed'], expected_speed2)

        # Third update, very short interval, few tokens
        tracker.update_token_count(260) # 10 more tokens (total 260) at time T3 (e.g. 102.1)
        tracker.times[-1] = 102.1 # Manually set time

        expected_speed3 = (260 - 0) / (102.1 - 100.0) # (260 / 2.1)
        self.assertAlmostEqual(tracker.current_metrics['speed'], expected_speed3)

        self.assertEqual(len(tracker.times), 4) # Initial + 3 updates
        self.assertEqual(len(tracker.token_counts), 4)
        self.assertEqual(tracker.token_counts[-1], 260)


if __name__ == '__main__':
    unittest.main()

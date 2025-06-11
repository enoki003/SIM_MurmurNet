import unittest
import numpy as np
from slm_emergent_ai.boids.rules import (
    alignment_rule,
    cohesion_rule,
    separation_rule,
    apply_boids_rules,
    calculate_entropy,
    calculate_vdi
)

class TestBoidsRules(unittest.TestCase):
    """Boidsルールのテスト"""
    
    def setUp(self):
        """テスト用データの準備"""
        # テスト用のlogits
        self.logits = np.random.randn(100)
        
        # テスト用の近傍ベクトル
        self.neighbor_vecs = np.random.randn(5, 384)
        # 正規化
        norms = np.linalg.norm(self.neighbor_vecs, axis=1, keepdims=True)
        self.neighbor_vecs = self.neighbor_vecs / (norms + 1e-8)
        
        # テスト用のサマリーベクトル
        self.summary_vec = np.random.randn(384)
        # 正規化
        self.summary_vec = self.summary_vec / (np.linalg.norm(self.summary_vec) + 1e-8)
        
        # テスト用のλパラメータ
        self.lambda_params = {
            'λ_a': 0.3,
            'λ_c': 0.3,
            'λ_s': 0.1
        }
    
    def test_alignment_rule(self):
        """整列ルールのテスト"""
        # 整列ルールを適用
        result = alignment_rule(self.logits, self.neighbor_vecs, self.lambda_params['λ_a'])
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(np.array_equal(result, self.logits))
    
    def test_cohesion_rule(self):
        """結合ルールのテスト"""
        # 結合ルールを適用
        result = cohesion_rule(self.logits, self.summary_vec, self.lambda_params['λ_c'])
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(np.array_equal(result, self.logits))
    
    def test_separation_rule(self):
        """分離ルールのテスト"""
        # 分離ルールを適用
        result = separation_rule(self.logits, self.lambda_params['λ_s'], seed=42)
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(np.array_equal(result, self.logits))

        # 同じシードで2回実行した結果が同じであることを確認
        result2 = separation_rule(self.logits, self.lambda_params['λ_s'], seed=42)
        self.assertTrue(np.array_equal(result, result2))
    
    def test_apply_boids_rules(self):
        """すべてのBoidsルールを適用するテスト"""
        # すべてのルールを適用
        result = apply_boids_rules(
            self.logits,
            self.neighbor_vecs,
            self.summary_vec,
            self.lambda_params,
            seed=42
        )
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(np.array_equal(result, self.logits))
    
    def test_calculate_entropy(self):
        """エントロピー計算のテスト"""
        # テスト用の確率分布
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        
        # エントロピーを計算
        entropy = calculate_entropy(probs)
        
        # エントロピーが正の値であることを確認
        self.assertGreater(entropy, 0)
        
        # 一様分布のエントロピーが最大であることを確認
        uniform_probs = np.ones(4) / 4
        uniform_entropy = calculate_entropy(uniform_probs)
        self.assertGreaterEqual(uniform_entropy, entropy)
    
    def test_calculate_vdi(self):
        """VDI計算のテスト"""
        # テスト用のトークンリスト
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        
        # VDIを計算
        vdi = calculate_vdi(tokens)
        
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


if __name__ == '__main__':
    unittest.main()

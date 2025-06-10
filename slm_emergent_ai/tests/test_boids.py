import unittest
import numpy as np
import torch
from slm_emergent_ai.boids.rules import (
    alignment_rule,
    cohesion_rule,
    separation_rule,
    apply_boids_rules
)
from slm_emergent_ai.utils.metrics import calculate_entropy, calculate_vdi

class TestBoidsRules(unittest.TestCase):
    """Boidsルールのテスト"""
    
    def setUp(self):
        """テスト用データの準備"""
        # テスト用のlogits（PyTorchテンソル）
        self.logits = torch.randn(100, requires_grad=True)
        
        # テスト用の近傍ベクトル
        self.neighbor_vecs = torch.randn(5, 384)
        # 正規化
        self.neighbor_vecs = torch.nn.functional.normalize(self.neighbor_vecs, dim=1)
        
        # テスト用のサマリーベクトル
        self.summary_vec = torch.randn(384)
        # 正規化
        self.summary_vec = torch.nn.functional.normalize(self.summary_vec, dim=0)
        
        # テスト用のλパラメータ
        self.lambda_params = {
            'lambda_alignment': 0.3,
            'lambda_cohesion': 0.3,
            'lambda_separation': 0.1
        }
    
    def test_alignment_rule(self):
        """整列ルールのテスト"""
        # 整列ルールを適用
        result = alignment_rule(self.logits, self.neighbor_vecs, self.lambda_params['lambda_alignment'])
        
        # 結果がPyTorchテンソルであることを確認
        self.assertIsInstance(result, torch.Tensor)
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(torch.equal(result, self.logits))
    
    def test_cohesion_rule(self):
        """結合ルールのテスト"""
        # 結合ルールを適用
        result = cohesion_rule(self.logits, self.summary_vec, self.lambda_params['lambda_cohesion'])
        
        # 結果がPyTorchテンソルであることを確認
        self.assertIsInstance(result, torch.Tensor)
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(torch.equal(result, self.logits))
    
    def test_separation_rule(self):
        """分離ルールのテスト"""
        # 分離ルールを適用
        result = separation_rule(self.logits, self.lambda_params['lambda_separation'])
        
        # 結果がPyTorchテンソルであることを確認
        self.assertIsInstance(result, torch.Tensor)
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認（ノイズが追加されるため）
        self.assertFalse(torch.equal(result, self.logits))
    
    def test_apply_boids_rules(self):
        """すべてのBoidsルールを適用するテスト"""
        # すべてのルールを適用
        result = apply_boids_rules(
            self.logits,
            self.neighbor_vecs,
            self.summary_vec,
            self.lambda_params
        )
        
        # 結果がPyTorchテンソルであることを確認
        self.assertIsInstance(result, torch.Tensor)
        
        # 結果の形状が元のlogitsと同じであることを確認
        self.assertEqual(result.shape, self.logits.shape)
        
        # 結果が元のlogitsと異なることを確認
        self.assertFalse(torch.equal(result, self.logits))
    
    def test_calculate_entropy(self):
        """エントロピー計算のテスト"""
        # テスト用の確率分布
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        
        # エントロピーを計算
        entropy = calculate_entropy(probs)
        
        # エントロピーが正の値であることを確認
        self.assertGreater(entropy.item(), 0)
        
        # 一様分布のエントロピーが最大であることを確認
        uniform_probs = torch.ones(4) / 4
        uniform_entropy = calculate_entropy(uniform_probs)
        self.assertGreaterEqual(uniform_entropy.item(), entropy.item())
    
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

    def test_lambda_parameters_effect(self):
        """λパラメータの効果をテスト"""
        # λパラメータを0にした場合
        zero_lambda = {
            'lambda_alignment': 0.0,
            'lambda_cohesion': 0.0,
            'lambda_separation': 0.0
        }
        
        result_zero = apply_boids_rules(
            self.logits,
            self.neighbor_vecs,
            self.summary_vec,
            zero_lambda
        )
        
        # λが0の場合、結果は元のlogitsとほぼ同じであることを確認
        self.assertTrue(torch.allclose(result_zero, self.logits, atol=1e-6))


if __name__ == '__main__':
    unittest.main()

"""
メトリクス評価モジュール - 評価指標の実装

システムの性能を評価するための指標を実装するモジュール。
語彙多様性指数(VDI)、Fake Correction Rate(FCR)、生成速度(tok/s)などを計算。
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Union
import re


def calculate_entropy(probs: np.ndarray) -> float:
    """
    確率分布のエントロピーを計算
    
    Parameters:
    -----------
    probs: 確率分布
    
    Returns:
    --------
    エントロピー値
    """
    # 0の確率を小さな値に置き換えて対数計算を安定化
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs))


def calculate_vdi(tokens: List[int], window_size: int = 100) -> float:
    """
    語彙多様性指数 (VDI) 改良版を計算
    
    VDI* = (UniqueTokens) / (log(TotalTokens)+1)
    
    Parameters:
    -----------
    tokens: トークンのリスト
    window_size: 計算ウィンドウサイズ
    
    Returns:
    --------
    VDI*値
    """
    # 最新のwindow_size個のトークンを使用
    recent_tokens = tokens[-window_size:] if len(tokens) > window_size else tokens
    
    # ユニークトークン数を計算
    unique_tokens = len(set(recent_tokens))
    
    # 総トークン数を計算
    total_tokens = len(recent_tokens)
    
    # VDI*を計算
    vdi = unique_tokens / (np.log(total_tokens) + 1) if total_tokens > 0 else 0
    
    return vdi


def calculate_fcr(fact_checks: List[bool]) -> float:
    """
    Fake Correction Rate (FCR) を計算
    
    Parameters:
    -----------
    fact_checks: 事実チェック結果のリスト（True: 正しい、False: 間違い）
    
    Returns:
    --------
    FCR値
    """
    if not fact_checks:
        return 1.0  # 事実チェックがない場合は満点
    
    return sum(fact_checks) / len(fact_checks)


def calculate_speed(times: List[float], token_counts: List[int]) -> float:
    """
    生成速度 (tok/s) を計算
    
    Parameters:
    -----------
    times: 時間のリスト
    token_counts: 各時間でのトークン数
    
    Returns:
    --------
    tok/s
    """
    if len(times) < 2 or len(token_counts) < 2:
        return 0.0
    
    elapsed_time = times[-1] - times[0]
    token_diff = token_counts[-1] - token_counts[0]
    
    if elapsed_time <= 0 or token_diff <= 0:
        return 0.0
    
    return token_diff / elapsed_time


class MetricsTracker:
    """メトリクス追跡クラス"""
    def __init__(self):
        """初期化"""
        self.reset()
    
    def reset(self):
        """メトリクスをリセット"""
        self.current_metrics = {
            'entropy': 0.0,
            'vdi': 0.0,
            'fcr': 0.0,
            'speed': 0.0,
            'λ_a': 0.0,
            'λ_c': 0.0,
            'λ_s': 0.0
        }
        self.metrics_history = {
            'entropy': [],
            'vdi': [],
            'fcr': [],
            'speed': [],
            'λ_a': [],
            'λ_c': [],
            'λ_s': []
        }
        self.times = [time.time()]
        self.token_counts = [0]
    
    def update_entropy(self, entropy: float):
        """エントロピーを更新"""
        self.current_metrics['entropy'] = entropy
        self.metrics_history['entropy'].append(entropy)
    
    def update_vdi(self, vdi: float):
        """VDIを更新"""
        self.current_metrics['vdi'] = vdi
        self.metrics_history['vdi'].append(vdi)
    
    def update_fcr(self, fcr: float):
        """FCRを更新"""
        self.current_metrics['fcr'] = fcr
        self.metrics_history['fcr'].append(fcr)
    
    def update_speed(self, speed: float):
        """速度を更新"""
        self.current_metrics['speed'] = speed
        self.metrics_history['speed'].append(speed)
    
    def update_lambda(self, lambda_name: str, value: float):
        """λパラメータを更新"""
        if lambda_name in self.current_metrics:
            self.current_metrics[lambda_name] = value
            self.metrics_history[lambda_name].append(value)
    
    def update_token_count(self, count: int):
        """トークン数を更新"""
        self.token_counts.append(count)
        self.times.append(time.time())
        
        # 速度を計算して更新
        speed = calculate_speed(self.times, self.token_counts)
        self.update_speed(speed)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        メトリクスの履歴を取得
        
        Parameters:
        -----------
        metric_name: メトリクス名
        
        Returns:
        --------
        メトリクスの履歴
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        現在のメトリクスを取得
        
        Returns:
        --------
        メトリクス辞書
        """
        return self.current_metrics.copy()

"""
メトリクス評価モジュール - 評価指標の実装

システムの性能を評価するための指標を実装するモジュール。
語彙多様性指数(VDI)、Fake Correction Rate(FCR)、生成速度(tok/s)などを計算。
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Union
import re
from sklearn.metrics.pairwise import cosine_similarity


def calculate_entropy(data: List[Dict[str, Any]], agent_key: str = "agent_id") -> float:
    """
    エージェント活動からエントロピーを計算（符号修正済み）
    
    Parameters:
    -----------
    data: メッセージデータのリスト
    agent_key: エージェントIDのキー
    
    Returns:
    --------
    エントロピー値（正の値）
    """
    if not data:
        return 0.0
    
    # エージェント別のメッセージ数をカウント
    agent_counts = {}
    for item in data:
        if isinstance(item, dict) and agent_key in item:
            agent_id = item[agent_key]
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
    
    if not agent_counts:
        return 0.0
    
    # 確率を計算
    total_messages = sum(agent_counts.values())
    probabilities = [count / total_messages for count in agent_counts.values()]
    
    # エントロピー計算（符号修正：負号を使用して正の値にする）
    entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities if p > 0)
    
    return max(0.0, entropy)  # 念のため非負値を保証

def calculate_similarity_matrix_without_diagonal(embeddings: np.ndarray) -> np.ndarray:
    """
    対角セルを除去した類似度行列を計算
    
    Parameters:
    -----------
    embeddings: 埋め込みベクトルの配列
    
    Returns:
    --------
    対角セルを除去した類似度行列
    """
    if len(embeddings) < 2:
        return np.array([])
    
    # 類似度行列を計算
    similarity_matrix = cosine_similarity(embeddings)
    
    # 対角セルをNaNに設定して除外
    np.fill_diagonal(similarity_matrix, np.nan)
    
    return similarity_matrix

def calculate_diversity_score(similarity_matrix: np.ndarray) -> float:
    """
    対角セル除去済み類似度行列から多様性スコアを計算
    
    Parameters:
    -----------
    similarity_matrix: 対角セル除去済み類似度行列
    
    Returns:
    --------
    多様性スコア（1 - 平均類似度）
    """
    if similarity_matrix.size == 0:
        return 0.0
    
    # NaNを除いた有効な類似度値のみを使用
    valid_similarities = similarity_matrix[~np.isnan(similarity_matrix)]
    
    if len(valid_similarities) == 0:
        return 0.0
    
    avg_similarity = np.mean(valid_similarities)
    diversity_score = 1.0 - avg_similarity
    
    return max(0.0, min(1.0, diversity_score))


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
    if not tokens:
        return 0.0
    
    # 最新のwindow_size個のトークンを使用
    recent_tokens = tokens[-window_size:] if len(tokens) > window_size else tokens
    
    # ユニークトークン数を計算
    unique_tokens = len(set(recent_tokens))
    
    # 総トークン数を計算
    total_tokens = len(recent_tokens)
    
    if total_tokens == 0:
        return 0.0
    
    # VDI*を計算
    vdi = unique_tokens / (np.log(total_tokens) + 1)
    
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


def calculate_response_diversity_fcr(messages: List[str]) -> float:
    """
    応答の多様性に基づくFCRを計算（代替実装）
    
    Parameters:
    -----------
    messages: メッセージのリスト
    
    Returns:
    --------
    多様性に基づくFCR値（0.0-1.0）
    """
    if not messages or len(messages) < 2:
        return 1.0
    
    # メッセージの重複を確認
    unique_messages = set()
    total_messages = 0
    
    for msg in messages:
        if isinstance(msg, str) and len(msg.strip()) > 10:
            # 最初の50文字で類似性を判定
            msg_signature = msg.strip()[:50].lower()
            unique_messages.add(msg_signature)
            total_messages += 1
    
    if total_messages == 0:
        return 1.0
    
    # 多様性比率を計算
    diversity_ratio = len(unique_messages) / total_messages
    return min(diversity_ratio, 1.0)


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
        print(f"[METRICS_DEBUG] MetricsTracker.update_entropy received: {entropy}")
        self.current_metrics['entropy'] = entropy
        self.metrics_history['entropy'].append(entropy)
    
    def update_vdi(self, vdi: float):
        """VDIを更新"""
        print(f"[METRICS_DEBUG] MetricsTracker.update_vdi received: {vdi}")
        self.current_metrics['vdi'] = vdi
        self.metrics_history['vdi'].append(vdi)
    
    def update_fcr(self, fcr: float):
        """FCRを更新"""
        print(f"[METRICS_DEBUG] MetricsTracker.update_fcr received: {fcr}")
        self.current_metrics['fcr'] = fcr
        self.metrics_history['fcr'].append(fcr)
    
    def update_speed(self, speed: float):
        """速度を更新"""
        print(f"[METRICS_DEBUG] MetricsTracker.update_speed received: {speed}")
        self.current_metrics['speed'] = speed
        self.metrics_history['speed'].append(speed)
    
    def update_lambda(self, lambda_name: str, value: float):
        """λパラメータを更新"""
        if lambda_name in self.current_metrics:
            self.current_metrics[lambda_name] = value
            self.metrics_history[lambda_name].append(value)
    
    def update_token_count(self, count: int):
        """トークン数を更新"""
        print(f"[METRICS_DEBUG] MetricsTracker.update_token_count received: {count}")
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
    
    def calculate_advanced_entropy(self, messages: List[Dict[str, Any]]) -> float:
        """
        高度なエントロピー計算（複数の要素を考慮）
        """
        if not messages:
            return 0.0
        
        # エージェント活動のエントロピー
        agent_entropy = calculate_entropy(messages, "agent_id")
        
        # 時間的分布のエントロピー（時間帯別活動）
        if len(messages) > 5:
            timestamps = [msg.get("timestamp", 0) for msg in messages if isinstance(msg, dict)]
            if timestamps:
                # 時間を区間に分割してエントロピーを計算
                time_bins = np.histogram(timestamps, bins=min(10, len(timestamps)))[0]
                time_probs = time_bins / np.sum(time_bins) if np.sum(time_bins) > 0 else []
                time_entropy = -sum(p * np.log2(p + 1e-10) for p in time_probs if p > 0)
            else:
                time_entropy = 0.0
        else:
            time_entropy = 0.0
        
        # 重み付き合成エントロピー
        combined_entropy = 0.7 * agent_entropy + 0.3 * time_entropy
        
        return combined_entropy
    
    def update_with_advanced_metrics(self, messages: List[Dict[str, Any]], 
                                   embeddings: Optional[np.ndarray] = None):
        """
        高度なメトリクス計算での更新
        """
        # 修正済みエントロピー計算
        entropy = self.calculate_advanced_entropy(messages)
        self.update_entropy(entropy)
        
        # 対角セル除去での多様性計算
        if embeddings is not None and len(embeddings) > 1:
            similarity_matrix = calculate_similarity_matrix_without_diagonal(embeddings)
            diversity = calculate_diversity_score(similarity_matrix)
            self.update_vdi(diversity)
        
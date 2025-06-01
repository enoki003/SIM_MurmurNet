"""
Boids Rules - Boidsアルゴリズムの実装

言語モデルの出力に適用するBoidsアルゴリズムのルール実装。
整列(Alignment)、結合(Cohesion)、分離(Separation)の3つの基本ルールを実装。
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def alignment_rule(logits: np.ndarray, neighbor_vecs: np.ndarray, λ_a: float) -> np.ndarray:
    """
    整列ルール - 近傍の平均方向に合わせる
    
    Parameters:
    -----------
    logits: 元のモデル出力logits
    neighbor_vecs: 近傍のベクトル表現
    λ_a: 整列ルールの重み係数
    
    Returns:
    --------
    整列ルールを適用した後のlogits
    """
    if neighbor_vecs is None or len(neighbor_vecs) == 0:
        return logits
    
    # 近傍の平均方向を計算
    mean_dir = np.mean(neighbor_vecs, axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm > 0:
        mean_dir = mean_dir / norm
    
    # logitsに反映（埋め込み空間からlogits空間への変換）
    # テスト用に簡略化した実装 - 実際のモデルでは埋め込み次元とlogits次元が一致する必要はない
    if mean_dir.shape[0] != logits.shape[0]:
        # 次元が一致しない場合は、ランダム投影行列を使用して次元を合わせる
        projection = np.random.randn(mean_dir.shape[0], logits.shape[0])
        alignment_effect = np.dot(mean_dir, projection)
    else:
        alignment_effect = mean_dir
    
    # 正規化
    alignment_effect = (alignment_effect - np.mean(alignment_effect)) / (np.std(alignment_effect) + 1e-8)
    
    return logits + λ_a * alignment_effect

def cohesion_rule(logits: np.ndarray, summary_vec: np.ndarray, λ_c: float) -> np.ndarray:
    """
    結合ルール - トピックの中心に向かう
    
    Parameters:
    -----------
    logits: 元のモデル出力logits
    summary_vec: トピックサマリーのベクトル表現
    λ_c: 結合ルールの重み係数
    
    Returns:
    --------
    結合ルールを適用した後のlogits
    """
    if summary_vec is None or np.all(summary_vec == 0):
        return logits
    
    # トピック中心との類似度を計算
    # テスト用に簡略化した実装 - 実際のモデルでは埋め込み次元とlogits次元が一致する必要はない
    if summary_vec.shape[0] != logits.shape[0]:
        # 次元が一致しない場合は、ランダム投影行列を使用して次元を合わせる
        projection = np.random.randn(summary_vec.shape[0], logits.shape[0])
        cohesion_effect = np.dot(summary_vec, projection)
    else:
        cohesion_effect = summary_vec
    
    # 正規化
    cohesion_effect = (cohesion_effect - np.mean(cohesion_effect)) / (np.std(cohesion_effect) + 1e-8)
    
    return logits + λ_c * cohesion_effect

def separation_rule(logits: np.ndarray, λ_s: float, seed: Optional[int] = None) -> np.ndarray:
    """
    分離ルール - 冗長な表現を避ける（エントロピーを増加させる）
    
    Parameters:
    -----------
    logits: 元のモデル出力logits
    λ_s: 分離ルールの重み係数
    seed: 乱数シード（再現性のため）
    
    Returns:
    --------
    分離ルールを適用した後のlogits
    """
    if seed is not None:
        np.random.seed(seed)
    
    # ランダムノイズを生成
    noise = np.random.randn(*logits.shape)
    
    # 正規化
    noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-8)
    
    return logits + λ_s * noise

def apply_boids_rules(logits: np.ndarray, 
                     neighbor_vecs: Optional[np.ndarray], 
                     summary_vec: Optional[np.ndarray], 
                     λ: Dict[str, float],
                     seed: Optional[int] = None) -> np.ndarray:
    """
    すべてのBoidsルールを適用
    
    Parameters:
    -----------
    logits: 元のモデル出力logits
    neighbor_vecs: 近傍のベクトル表現
    summary_vec: トピックサマリーのベクトル表現
    λ: 各ルールの重み係数
    seed: 乱数シード（再現性のため）
    
    Returns:
    --------
    すべてのルールを適用した後のlogits
    """
    # 整列ルール
    logits = alignment_rule(logits, neighbor_vecs, λ.get('λ_a', 0.3))
    
    # 結合ルール
    logits = cohesion_rule(logits, summary_vec, λ.get('λ_c', 0.3))
    
    # 分離ルール
    logits = separation_rule(logits, λ.get('λ_s', 0.1), seed)
    
    return logits

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
    語彙多様性指数 (VDI) を計算
    
    Parameters:
    -----------
    tokens: トークンのリスト
    window_size: 計算ウィンドウサイズ
    
    Returns:
    --------
    VDI値
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

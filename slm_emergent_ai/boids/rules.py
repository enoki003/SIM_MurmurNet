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
    if mean_dir.shape[0] != logits.shape[0]:
        # 次元が一致しない場合は、決定論的な投影を使用
        # シンプルな線形変換（平均プーリングまたは反復）
        if mean_dir.shape[0] > logits.shape[0]:
            # ダウンサンプリング: 平均プーリング
            pool_size = mean_dir.shape[0] // logits.shape[0]
            alignment_effect = np.array([
                np.mean(mean_dir[i*pool_size:(i+1)*pool_size]) 
                for i in range(logits.shape[0])
            ])
        else:
            # アップサンプリング: 繰り返し
            alignment_effect = np.tile(mean_dir, (logits.shape[0] // mean_dir.shape[0] + 1))[:logits.shape[0]]
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
    if summary_vec.shape[0] != logits.shape[0]:
        # 次元が一致しない場合は、決定論的な投影を使用
        if summary_vec.shape[0] > logits.shape[0]:
            # ダウンサンプリング: 平均プーリング
            pool_size = summary_vec.shape[0] // logits.shape[0]
            cohesion_effect = np.array([
                np.mean(summary_vec[i*pool_size:(i+1)*pool_size]) 
                for i in range(logits.shape[0])
            ])
        else:
            # アップサンプリング: 繰り返し
            cohesion_effect = np.tile(summary_vec, (logits.shape[0] // summary_vec.shape[0] + 1))[:logits.shape[0]]
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
    分離ルールを適用した後のlogits    """
    # 決定論的な分離効果を生成（logitsの変動に基づく）
    if len(logits) <= 1:
        return logits
    
    # logitsの標準偏差に基づいた分離効果
    logits_std = np.std(logits)
    if logits_std == 0:
        # 全て同じ値の場合は小さな分離を加える
        separation_effect = np.linspace(-0.01, 0.01, len(logits))
    else:
        # 正規化されたlogitsの偏差を分離効果として使用
        separation_effect = (logits - np.mean(logits)) / logits_std
        # 分離効果を制限
        separation_effect = np.clip(separation_effect, -1.0, 1.0)
    
    return logits + λ_s * separation_effect

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

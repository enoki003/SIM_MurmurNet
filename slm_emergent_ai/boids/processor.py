"""
LogitsProcessor - Boidsルールをモデル推論に適用するためのプロセッサ

transformersライブラリのLogitsProcessorを拡張し、
Boidsアルゴリズムをモデル生成時に適用するためのプロセッサ実装。
"""

import torch
from typing import Dict, List, Optional, Union, Any
from ..boids.blackboard import BlackBoard  # Import BlackBoard from the appropriate module
import numpy as np
from transformers import LogitsProcessor

from ..boids.rules import apply_boids_rules


class BoidsLogitsProcessor(LogitsProcessor):
    """
    Boidsアルゴリズムを適用するLogitsProcessor
    
    transformersライブラリのLogitsProcessorを拡張し、
    生成時にBoidsルールを適用する
    """
    def __init__(self, 
                 neighbor_vecs: Optional[np.ndarray] = None,
                 summary_vec: Optional[np.ndarray] = None,
                 λ: Dict[str, float] = None,
                 seed: Optional[int] = None):
        """
        BoidsLogitsProcessorの初期化
        
        Parameters:
        -----------
        neighbor_vecs: 近傍のベクトル表現
        summary_vec: トピックサマリーのベクトル表現
        λ: 各ルールの重み係数
        seed: 乱数シード（再現性のため）
        """
        self.neighbor_vecs = neighbor_vecs
        self.summary_vec = summary_vec
        self.λ = λ if λ is not None else {'λ_a': 0.3, 'λ_c': 0.3, 'λ_s': 0.1}
        self.seed = seed
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        LogitsProcessorの呼び出しメソッド
        
        Parameters:
        -----------
        input_ids: 入力トークンID
        scores: モデルが出力したスコア（logits）
        
        Returns:
        --------
        Boidsルールを適用した後のスコア
        """
        # PyTorch tensorをNumPy配列に変換
        logits_np = scores.detach().cpu().numpy()
        
        # Boidsルールを適用
        modified_logits = apply_boids_rules(
            logits_np,
            self.neighbor_vecs,
            self.summary_vec,
            self.λ,
            self.seed
        )
        
        # NumPy配列をPyTorch tensorに戻す
        modified_scores = torch.tensor(
            modified_logits, 
            dtype=scores.dtype, 
            device=scores.device
        )
        
        return modified_scores


class BoidsProcessor:
    """
    Boidsアルゴリズムを適用するプロセッサ
    
    transformersライブラリを使用しない場合のプロセッサ
    """
    def __init__(self, 
                 bb: 'BlackBoard',
                 λ: Dict[str, float] = None,
                 seed: Optional[int] = None):
        """
        BoidsProcessorの初期化
        
        Parameters:
        -----------
        bb: BlackBoardインスタンス
        λ: 各ルールの重み係数
        seed: 乱数シード（再現性のため）
        """
        self.bb = bb
        self.λ = λ if λ is not None else {'λ_a': 0.3, 'λ_c': 0.3, 'λ_s': 0.1}
        self.seed = seed
    
    def process(self, logits: np.ndarray, k: int = 16) -> np.ndarray:
        """
        logitsにBoidsルールを適用
        
        Parameters:
        -----------
        logits: モデルが出力したlogits
        k: 取得する近傍メッセージ数
        
        Returns:
        --------
        Boidsルールを適用した後のlogits
        """        # BlackBoardから近傍情報を取得
        messages = self.bb.pull(k)
        
        # 実際の埋め込みベクトルサービスが利用できない場合のフォールバック
        # 実際の実装では、sentence-transformersなどを使用してベクトル化
        neighbor_vecs = []
        for msg in messages:
            try:
                # 簡単なTF-IDFベースの代替実装
                words = msg.lower().split()
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                # 固定次元のベクトルを生成
                vec = np.zeros(384)
                for i, word in enumerate(sorted(word_freq.keys())[:384]):
                    vec[i] = word_freq[word]
                
                # 正規化
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                else:
                    vec = np.zeros(384)
                    
                neighbor_vecs.append(vec)
            except Exception as e:
                print(f"Error generating message embedding: {e}")
                # エラーの場合はゼロベクトルを使用
                neighbor_vecs.append(np.zeros(384))
        
        neighbor_vecs = np.array(neighbor_vecs) if neighbor_vecs else np.array([]).reshape(0, 384)
        
        # Boidsルールを適用
        modified_logits = apply_boids_rules(
            logits,
            neighbor_vecs,
            self.bb.summary_vec,
            self.λ,
            self.seed
        )
        
        return modified_logits

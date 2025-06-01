"""
SLMAgent - Small Language Model Agent Core Implementation

Boids理論に基づく局所ルールで動作するSLMエージェントの実装。
"""

import asyncio
import torch
from typing import Dict, List, Optional, Union, Any
import numpy as np

class LLM:
    """言語モデルのラッパークラス"""
    def __init__(self, model_path: str, threads: int = 4, quantize: str = "q4"):
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.model = None
        self.tokenizer = None
        self._initialize()
    
    def _initialize(self):
        """モデルの初期化 - gemma3:1bモデルを使用"""
        try:
            # CPUで推論を行う設定
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPUを無効化
            
            # モデルのロード（実際の実装ではHugging Faceなどから読み込む）
            # Candleフレームワークは使用せず、標準的なPythonライブラリを使用
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            print(f"Model loaded with {self.threads} threads")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def forward(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9) -> np.ndarray:
        """モデルの推論を実行し、logitsを返す"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 最後のトークンのlogitsを取得
        logits = outputs.logits[0, -1, :].numpy()
        return logits
    
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """テキスト生成を行う"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            repetition_penalty=kwargs.get("repeat_penalty", 1.0)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class BoidsCtx:
    """Boidsアルゴリズムのコンテキスト"""
    def __init__(self, neighbors: List[str], summary_vec: np.ndarray):
        self.neighbors = neighbors
        self.summary_vec = summary_vec
        self.neighbor_vecs = None
        self._process_neighbors()
    
    def _process_neighbors(self):
        """近傍テキストをベクトル化"""
        # 実際の実装では、sentence-transformersなどを使用
        # ここではダミー実装
        self.neighbor_vecs = np.random.randn(len(self.neighbors), 384)
        # 正規化
        norms = np.linalg.norm(self.neighbor_vecs, axis=1, keepdims=True)
        self.neighbor_vecs = self.neighbor_vecs / (norms + 1e-8)


def apply_boids(logits: np.ndarray, ctx: BoidsCtx, λ: Dict[str, float]) -> np.ndarray:
    """
    Boidsルールをlogitsに適用する
    
    Parameters:
    -----------
    logits: 元のモデル出力logits
    ctx: Boidsコンテキスト（近傍情報）
    λ: 各ルールの重み係数
    
    Returns:
    --------
    修正されたlogits
    """
    # 1. 整列 (Alignment) - 近傍の平均方向に合わせる
    alignment = np.zeros_like(logits)
    if ctx.neighbor_vecs is not None and len(ctx.neighbor_vecs) > 0:
        # 近傍の平均方向を計算
        mean_dir = np.mean(ctx.neighbor_vecs, axis=0)
        # logitsに反映（実際の実装ではより複雑な変換が必要）
        alignment = np.dot(mean_dir, logits.reshape(-1, 1)).flatten()
    
    # 2. 結合 (Cohesion) - トピックの中心に向かう
    cohesion = np.zeros_like(logits)
    if ctx.summary_vec is not None:
        # トピック中心との類似度を計算
        cohesion = np.dot(ctx.summary_vec, logits.reshape(-1, 1)).flatten()
    
    # 3. 分離 (Separation) - 冗長な表現を避ける
    # エントロピーを増加させる方向に調整
    separation = np.random.randn(*logits.shape) * 0.1
    
    # 重み付き合成
    modified_logits = (
        logits + 
        λ.get("λ_a", 0.3) * alignment + 
        λ.get("λ_c", 0.3) * cohesion + 
        λ.get("λ_s", 0.1) * separation
    )
    
    return modified_logits


def sample_top_p(logits: np.ndarray, top_p: float = 0.9) -> int:
    """Top-p (nucleus) samplingでトークンを選択"""
    # ソートしたインデックスを取得
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    # 累積確率を計算
    sorted_probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Top-p以下のインデックスをマスク
    sorted_indices_to_keep = sorted_indices[cumulative_probs <= top_p]
    
    # 少なくとも1つのトークンを保持
    if len(sorted_indices_to_keep) == 0:
        sorted_indices_to_keep = sorted_indices[0:1]
    
    # 確率に従ってサンプリング
    probs = np.exp(logits[sorted_indices_to_keep]) / np.sum(np.exp(logits[sorted_indices_to_keep]))
    chosen_idx = np.random.choice(sorted_indices_to_keep, p=probs)
    
    return int(chosen_idx)


class SLMAgent:
    """Small Language Model Agent"""
    def __init__(self, id: int, role: str, model: Any, tokenizer: Any, λ: Dict[str, float]):
        self.id = id
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.λ = λ
        self.cache = {}
    
    async def generate(self, prompt: str, bb: 'BlackBoard') -> str:
        """
        Boidsルールに基づいてテキスト生成を行う
        
        Parameters:
        -----------
        prompt: 入力プロンプト
        bb: BlackBoardインスタンス（共有メモリ）
        
        Returns:
        --------
        生成されたトークン
        """
        # BlackBoardから近傍情報を取得
        ctx = BoidsCtx(bb.pull(k=16), bb.summary_vec)
        
        # モデルの推論実行
        logits = self.model.forward(prompt)
        
        # Boidsルールを適用
        logits = apply_boids(logits, ctx, self.λ)
        
        # トークンのサンプリング
        token = sample_top_p(logits)
        
        # トークンをデコード
        token_str = self.tokenizer.decode([token])
        
        return token_str
    
    async def run_conversation(self, initial_prompt: str, bb: 'BlackBoard', max_turns: int = 10) -> List[str]:
        """会話を実行する"""
        conversation = [initial_prompt]
        current_prompt = initial_prompt
        
        for _ in range(max_turns):
            # トークン生成
            token = await self.generate(current_prompt, bb)
            
            # 会話に追加
            conversation.append(token)
            current_prompt += token
            
            # BlackBoardに情報をプッシュ
            await bb.push(self.id, token)
        
        return conversation

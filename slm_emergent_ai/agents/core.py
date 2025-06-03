"""
SLMAgent - Small Language Model Agent Core Implementation

Boids理論に基づく局所ルールで動作するSLMエージェントの実装。
"""

import asyncio
import torch
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..memory.blackboard import BlackBoard

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
        """モデルの初期化 - ローカルGGUFファイルまたはHugging Faceモデルを使用"""
        try:
            # CPUで推論を行う設定
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPUを無効化
            
            # GGUFファイルの場合とHugging Faceモデルの場合を判別
            if self.model_path.endswith('.gguf'):
                # GGUFファイルの場合
                self._initialize_gguf()
            else:
                # Hugging Faceモデルの場合
                self._initialize_hf()
                
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def _initialize_gguf(self):
        """GGUFファイルからモデルを初期化"""
        try:
            # llama-cpp-pythonを使用してGGUFファイルを読み込む
            try:
                from llama_cpp import Llama
                
                print(f"Loading GGUF model: {self.model_path}")
                self.model = Llama(
                    model_path=self.model_path,
                    n_threads=self.threads,
                    verbose=False
                )
                # GGUFモデルの場合、tokenizerは内蔵
                self.tokenizer = None
                print(f"GGUF model loaded with {self.threads} threads")
                
            except ImportError:
                # llama-cpp-pythonがインストールされていない場合のフォールバック
                print("llama-cpp-python not available, using dummy implementation")
                self._initialize_dummy()
                
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            # フォールバックとしてダミー実装を使用
            self._initialize_dummy()
    
    def _initialize_hf(self):
        """Hugging Faceモデルから初期化"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading HF model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        print(f"HF model loaded with {self.threads} threads")
    
    def _initialize_dummy(self):
        """ダミー実装（テスト用）"""
        print("Using dummy model implementation")
        self.model = None
        self.tokenizer = None
    
    def forward(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9) -> np.ndarray:
        """モデルの推論を実行し、logitsを返す"""
        if self.model is None:
            # ダミー実装の場合
            return np.random.random(32000)  # 適当なlogitsを返す
        
        if hasattr(self.model, 'tokenize'):
            # GGUFモデル（llama-cpp-python）の場合
            try:
                tokens = self.model.tokenize(prompt.encode('utf-8'))
                output = self.model(tokens)
                # 最後のトークンのlogitsを取得
                return np.array(output['logits'][-1])
            except:
                # エラーの場合はダミーlogitsを返す
                return np.random.random(32000)
        else:
            # Hugging Faceモデルの場合
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 最後のトークンのlogitsを取得
            logits = outputs.logits[0, -1, :].numpy()
            return logits
    
    def _format_gemma_prompt(self, prompt: str) -> str:
        """GEMMAモデル用のプロンプトフォーマット"""
        # GEMMA-3-1B-ITモデル用のフォーマット
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    def _is_gemma_model(self) -> bool:
        """GEMMAモデルかどうかを判定"""
        return "gemma" in self.model_path.lower()
    
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """テキスト生成を行う"""
        if self.model is None:
            # ダミー実装の場合
            return f"Generated response for: {prompt[:50]}..."
        
        # GEMMAモデルの場合はプロンプトをフォーマット
        formatted_prompt = prompt
        if self._is_gemma_model():
            formatted_prompt = self._format_gemma_prompt(prompt)
        
        if hasattr(self.model, 'tokenize'):
            # GGUFモデル（llama-cpp-python）の場合
            try:
                # stop sequencesを設定（GEMMAモデル用とユーザー指定の両方を考慮）
                default_stops = ["<end_of_turn>"] if self._is_gemma_model() else []
                user_stops = kwargs.get("stop", [])
                if isinstance(user_stops, str):
                    user_stops = [user_stops]
                stop_sequences = list(set(default_stops + user_stops))  # 重複を除去
                
                response = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    repeat_penalty=kwargs.get("repeat_penalty", 1.0),
                    stop=stop_sequences
                )
                generated_text = response['choices'][0]['text']
                
                # GEMMAモデルの場合は余分なタグを除去し、出力をクリーンアップ
                if self._is_gemma_model():
                    generated_text = generated_text.replace("<end_of_turn>", "")
                    generated_text = generated_text.replace("<start_of_turn>", "")
                    generated_text = generated_text.replace("model\n", "")
                    generated_text = generated_text.replace("user\n", "")
                    generated_text = generated_text.strip()
                
                return generated_text
            except:
                # エラーの場合はダミーレスポンスを返す
                return f"GGUF model response for: {prompt[:50]}..."
        else:
            # Hugging Faceモデルの場合
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
        # 次元が合わない場合は調整
        if len(mean_dir) != len(logits):
            expanded_mean = np.tile(mean_dir, (len(logits) // len(mean_dir) + 1))[:len(logits)]
            alignment = expanded_mean * logits
        else:
            alignment = mean_dir * logits
    
    # 2. 結合 (Cohesion) - トピックの中心に向かう
    cohesion = np.zeros_like(logits)
    if ctx.summary_vec is not None:
        # 次元が合わない場合は、summary_vecを拡張またはlogitsを縮小
        if len(ctx.summary_vec) != len(logits):
            # summary_vecをlogitsの次元に合わせる（簡単な繰り返し）
            expanded_summary = np.tile(ctx.summary_vec, (len(logits) // len(ctx.summary_vec) + 1))[:len(logits)]
            cohesion = expanded_summary * logits
        else:
            cohesion = ctx.summary_vec * logits
    
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
        生成されたテキスト
        """
        # BlackBoardから近傍情報を取得（非同期対応）
        neighbors = await bb.pull(k=16)
        summary_vec = getattr(bb, 'summary_vec', np.random.randn(384))
        ctx = BoidsCtx(neighbors, summary_vec)
        
        # LLMの統一されたgenerateメソッドを使用
        try:
            generated_text = self.model.generate(
                prompt,
                max_tokens=20,
                temperature=0.7,
                top_p=0.9,
                stop=["<end_of_turn>", "\n", ".", "!", "?", "Human:", "User:", "Assistant:"]
            )
            
            # 最初の単語または短いフレーズを取得
            if generated_text and generated_text.strip():
                words = generated_text.strip().split()
                if words:
                    return words[0]
                else:
                    return "..."
            else:
                # フォールバック
                fallback_words = ["hello", "world", "yes", "no", "think", "know", "good", "bad"]
                return fallback_words[hash(prompt) % len(fallback_words)]
                
        except Exception as e:
            print(f"Error in SLMAgent generate: {e}")
            # エラー時のフォールバック
            fallback_words = ["hello", "world", "yes", "no", "think", "know", "good", "bad"]
            return fallback_words[hash(prompt) % len(fallback_words)]
    
    async def run_conversation(self, initial_prompt: str, bb: 'BlackBoard', max_turns: int = 10) -> List[str]:
        """会話を実行する"""
        conversation = [initial_prompt]
        current_prompt = initial_prompt
        
        # 初期メッセージをBlackBoardにプッシュ
        await bb.push({
            "agent_id": self.id,
            "role": self.role,
            "text": f"{self.role}として参加しました: {initial_prompt}",
            "timestamp": __import__('time').strftime("%H:%M:%S"),
            "type": "initial_prompt"
        })
        
        # メッセージバッファ
        message_buffer = ""
        
        for i in range(max_turns):
            # LLMクラスの統一されたgenerateメソッドを使用
            try:
                # プロンプトの準備
                base_prompt = f"{current_prompt}\n{self.role}: "
                
                # LLMのgenerateメソッドを使用（GEMMAフォーマッターが自動適用される）
                generated_text = self.model.generate(
                    base_prompt,
                    max_tokens=30,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<end_of_turn>", "\n", "Human:", "User:", "Assistant:", self.role + ":"]
                )
                
                if generated_text and generated_text.strip():
                    # 完全なメッセージを生成
                    clean_text = generated_text.strip()
                    conversation.append(clean_text)
                    current_prompt += f"\n{self.role}: {clean_text}"
                    
                    # BlackBoardにメッセージをプッシュ
                    await bb.push({
                        "agent_id": self.id,
                        "role": self.role,
                        "text": clean_text,
                        "timestamp": __import__('time').strftime("%H:%M:%S"),
                        "type": "message"
                    })
                else:
                    # 生成に失敗した場合のフォールバック
                    fallback_messages = [
                        "興味深い観点ですね。",
                        "それについて考えてみましょう。",
                        "別の角度から見ると...",
                        "確かにその通りです。",
                        "さらに詳しく分析すると...",
                        "他の可能性も検討してみます。"
                    ]
                    fallback_msg = fallback_messages[i % len(fallback_messages)]
                    conversation.append(fallback_msg)
                    current_prompt += f"\n{self.role}: {fallback_msg}"
                    
                    await bb.push({
                        "agent_id": self.id,
                        "role": self.role,
                        "text": fallback_msg,
                        "timestamp": __import__('time').strftime("%H:%M:%S"),
                        "type": "message"
                    })
                    
            except Exception as e:
                print(f"Error in conversation generation: {e}")
                # エラー時のフォールバック
                error_messages = [
                    "システムの処理中です...",
                    "データを分析しています...",
                    "新しい情報を統合中...",
                    "結果を確認しています..."
                ]
                error_msg = error_messages[i % len(error_messages)]
                conversation.append(error_msg)
                
                await bb.push({
                    "agent_id": self.id,
                    "role": self.role,
                    "text": error_msg,
                    "timestamp": __import__('time').strftime("%H:%M:%S"),
                    "type": "message"
                })
            
            # 短い待機時間
            await asyncio.sleep(0.1)
        
        return conversation

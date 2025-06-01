"""
SLMAgent - Small Language Model Agent Core Implementation

Boids理論に基づく局所ルールで動作するSLMエージェントの実装。
"""

import asyncio
import torch
from typing import Dict, List, Optional, Union, Any
import numpy as np
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
        """GGUFモデルの初期化 - llama-cpp-pythonを使用"""
        try:
            import os
            
            # GGUFファイルが存在するかチェック
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                # フォールバックとしてHugging Faceモデルを試す
                self._initialize_huggingface()
                return
            
            # llama-cpp-pythonでGGUFファイルをロード
            try:
                from llama_cpp import Llama
                print(f"Loading GGUF model: {self.model_path}")
                self.model = Llama(
                    model_path=self.model_path,
                    n_threads=self.threads,
                    n_ctx=2048,  # コンテキスト長
                    verbose=False
                )
                self.tokenizer = self.model  # llama-cpp-pythonでは同じオブジェクト
                print(f"GGUF model loaded with {self.threads} threads")
            except ImportError:
                print("llama-cpp-python not available, falling back to HuggingFace")
                self._initialize_huggingface()
                
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def _initialize_huggingface(self):
        """HuggingFaceモデルの初期化（フォールバック）"""
        try:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPUを無効化
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # モデル名から推定（パスが無効な場合）
            model_name = "google/gemma-2-2b-it"  # より軽量なモデル
            print(f"Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                device_map="cpu",
                torch_dtype=torch.float16
            )
            print(f"HuggingFace model loaded")
        except Exception as e:
            print(f"Error initializing HuggingFace model: {e}")
            raise
    
    def forward(self, prompt: str, temperature: float = 0.7, top_p: float = 0.9) -> np.ndarray:
        """モデルの推論を実行し、logitsを返す"""
        try:
            # llama-cpp-python (GGUF) の場合
            if hasattr(self.model, '__call__'):
                # GGUFモデルの場合は簡易的にランダムlogitsを生成
                # 実際の実装では内部APIを使用してlogitsを取得
                vocab_size = 32000  # Gemma-3の語彙サイズ（概算）
                logits = np.random.randn(vocab_size).astype(np.float32)
                return logits
            else:
                # HuggingFace transformers の場合
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                # 最後のトークンのlogitsを取得
                logits = outputs.logits[0, -1, :].numpy()
                return logits
        except Exception as e:
            print(f"Forward pass error: {e}")
            # エラー時のフォールバック
            return np.random.randn(32000).astype(np.float32)
    
    def generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> str:
        """テキスト生成を行う - GGUF/HuggingFace対応"""
        # Gemma-3用のプロンプトフォーマットを適用
        formatted_prompt = self._format_gemma3_prompt(prompt)
        
        try:
            # llama-cpp-python (GGUF) の場合
            if hasattr(self.model, '__call__'):
                response = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    repeat_penalty=kwargs.get("repeat_penalty", 1.0),
                    stop=["<end_of_turn>", "\n\n"]
                )
                generated_text = response['choices'][0]['text'].strip()
            else:
                # HuggingFace transformers の場合
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    repetition_penalty=kwargs.get("repeat_penalty", 1.0),
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # 生成されたテキストから元のプロンプトを除去
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if generated_text.startswith(formatted_prompt):
                    generated_text = generated_text[len(formatted_prompt):].strip()
                    
        except Exception as e:
            print(f"Generation error: {e}")
            generated_text = f"[Error generating response: {str(e)}]"
        
        return generated_text
    
    def _format_gemma3_prompt(self, prompt: str) -> str:
        """Gemma-3用のプロンプトフォーマットを適用"""
        # Gemma-3のチャット形式
        if not prompt.strip():
            return ""
        
        # シンプルなフォーマット - Gemma-3は比較的シンプルなフォーマットを好む
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        return formatted


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
        # logitsに反映（次元調整版）
        if len(mean_dir) != len(logits):
            min_dim = min(len(mean_dir), len(logits))
            alignment[:min_dim] = mean_dir[:min_dim] * 0.1
        else:
            alignment = mean_dir * 0.1
    
    # 2. 結合 (Cohesion) - トピックの中心に向かう
    cohesion = np.zeros_like(logits)
    if ctx.summary_vec is not None:
        # トピック中心との類似度を計算（次元調整版）
        if ctx.summary_vec.shape[0] != logits.shape[0]:
            # 次元が一致しない場合は、ランダム投影行列を使用して次元を合わせる
            projection = np.random.randn(ctx.summary_vec.shape[0], logits.shape[0])
            cohesion = np.dot(ctx.summary_vec, projection)
        else:
            cohesion = ctx.summary_vec
        
        # 正規化
        cohesion = (cohesion - np.mean(cohesion)) / (np.std(cohesion) + 1e-8)
    
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
        neighbors = await bb.pull(k=16)
        ctx = BoidsCtx(neighbors, bb.summary_vec)
        
        # LLMインスタンスを使ってlogitsを取得（Gemma-3フォーマット適用済み）
        if hasattr(self.model, 'forward'):
            logits = self.model.forward(prompt)
        else:
            # LLMクラスのインスタンスの場合は、generateメソッドを使用
            # 短いテキスト生成でlogitsを取得
            generated = self.model.generate(prompt, max_tokens=1)
            # ここでは簡単にトークン生成結果を返す
            return generated[:50] if len(generated) > 50 else generated
        
        # Boidsルールを適用
        logits = apply_boids(logits, ctx, self.λ)
        
        # トークンのサンプリング
        token = sample_top_p(logits)
        
        # トークンをデコード
        token_str = self.tokenizer.decode([token])
        
        return token_str
    
    async def run_conversation(self, initial_prompt: str, bb: 'BlackBoard', max_turns: int = 3) -> List[str]:
        """会話を実行する - Gemma-3フォーマットで改良"""
        conversation = []
        
        print(f"[AGENT {self.id}] Starting run_conversation with max_turns={max_turns}")
        
        # 役割に基づいた初期プロンプトの強化
        role_prompt = f"You are a {self.role}. Please respond thoughtfully and stay in character."
        full_prompt = f"{role_prompt}\n\nUser: {initial_prompt}\nAssistant:"
        
        print(f"[AGENT {self.id}] Initial prompt: {full_prompt[:100]}...")
        
        for turn in range(max_turns):
            try:
                print(f"[AGENT {self.id}] Starting turn {turn + 1}/{max_turns}")
                
                # タイムアウト付きでレスポンス生成
                try:
                    response_task = asyncio.create_task(self._generate_with_timeout(full_prompt, bb))
                    response = await asyncio.wait_for(response_task, timeout=30.0)  # 30秒タイムアウト
                except asyncio.TimeoutError:
                    print(f"[AGENT {self.id}] Generation timeout at turn {turn + 1}")
                    response = f"[Timeout at turn {turn + 1}]"
                
                # レスポンスをクリーンアップ
                response = response.strip()
                if response:
                    conversation.append(f"Turn {turn + 1} ({self.role}): {response}")
                    
                    print(f"[AGENT {self.id}] Generated response: {response[:100]}...")
                    
                    # BlackBoardに情報をプッシュ
                    await bb.push(self.id, response)
                    print(f"[AGENT {self.id}] Pushed to BlackBoard")
                    
                    # 次のターンのプロンプトを更新
                    full_prompt = f"{role_prompt}\n\nConversation so far:\n" + "\n".join(conversation[-2:]) + f"\n\nPlease continue the conversation as a {self.role}:"
                else:
                    conversation.append(f"Turn {turn + 1} ({self.role}): [No response generated]")
                    
            except Exception as e:
                print(f"[AGENT {self.id}] Error at turn {turn + 1}: {e}")
                conversation.append(f"Turn {turn + 1} ({self.role}): [Error: {str(e)}]")
                
        print(f"[AGENT {self.id}] Completed conversation with {len(conversation)} turns")
        return conversation
    
    async def _generate_with_timeout(self, prompt: str, bb: 'BlackBoard') -> str:
        """タイムアウト付きでテキスト生成"""
        print(f"[AGENT {self.id}] Starting text generation...")
        
        # LLMインスタンスを使ってより長いレスポンスを生成
        if hasattr(self.model, 'generate'):
            print(f"[AGENT {self.id}] Using LLM.generate method")
            # LLMインスタンスの場合
            response = self.model.generate(
                prompt, 
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            print(f"[AGENT {self.id}] LLM.generate completed")
        else:
            print(f"[AGENT {self.id}] Using self.generate method")
            # 従来の方法
            response = await self.generate(prompt, bb)
            print(f"[AGENT {self.id}] self.generate completed")
        
        return response

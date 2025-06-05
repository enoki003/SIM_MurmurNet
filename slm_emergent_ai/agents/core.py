"""
SLMAgent - Small Language Model Agent Core Implementation

Boids理論に基づく局所ルールで動作するSLMエージェントの実装。
"""

import asyncio
import torch
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import numpy as np
import os
import random
import time

if TYPE_CHECKING:
    from ..memory.blackboard import BlackBoard


class LLM:
    """言語モデルのラッパークラス"""
    
    def __init__(self, model_path: str, threads: int = 4, quantize: str = "q4", n_ctx: int = 512):
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.n_ctx = n_ctx
        self.model = None
        self.tokenizer = None
        self._dummy_mode = False
        self._initialize()

    # ------------------------------------------------------------------
    # モデル初期化
    # ------------------------------------------------------------------
    
    def _initialize(self):
        """モデルの初期化 - ローカルGGUFファイルまたはHugging Faceモデルを使用"""        # GPU を明示的に無効化
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            if self.model_path.endswith(".gguf"):
                self._initialize_gguf()
            else:
                self._initialize_hf()
        except Exception as e:
            print(f"[WARNING] モデル初期化に失敗しました。ダミーモードで動作します: {e}")
            # ダミーモードに切り替え
            self.model = None
            self.tokenizer = None
            self._dummy_mode = True

    def _initialize_gguf(self):
        """GGUFファイルからモデルを初期化"""
        try:
            from llama_cpp import Llama

            # モデルファイルの存在確認
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            print(f"Loading GGUF model: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                n_threads=self.threads,
                n_ctx=self.n_ctx,
                verbose=False,
            )
            self.tokenizer = None  # GGUF では組み込み
            print(
                f"GGUF model loaded successfully with {self.threads} threads, context length: {self.n_ctx}"
            )
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install it with: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {e}")

    def _initialize_hf(self):
        """Hugging Face モデルから初期化"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading HF model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, low_cpu_mem_usage=True, device_map="cpu"
            )
            print("HF model loaded successfully on CPU")
        except ImportError:
            raise ImportError(
                "transformers is required for Hugging Face models. "
                "Install it with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model: {e}")

    # ------------------------------------------------------------------
    # 推論ユーティリティ
    # ------------------------------------------------------------------
    def forward(self, prompt: str) -> np.ndarray:
        """モデルの推論を実行し logits を返す（最後のトークンのみ）"""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Cannot perform forward pass.")

        # llama‑cpp‑python (GGUF) か transformers (HF) かで分岐
        if hasattr(self.model, "tokenize"):
            try:
                tokens = self.model.tokenize(prompt.encode("utf-8"))
                output = self.model(tokens)
                return np.array(output["logits"][-1])
            except Exception as e:
                raise RuntimeError(f"GGUF model forward pass failed: {e}")
        else:
            try:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                return outputs.logits[0, -1, :].numpy()
            except Exception as e:
                raise RuntimeError(f"HF model forward pass failed: {e}")

    # ------------------------------------------------------------------
    # テキスト生成
    # ------------------------------------------------------------------
    
    def _format_gemma_prompt(self, prompt: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _is_gemma_model(self) -> bool:
        return "gemma" in self.model_path.lower()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """テキスト生成を実行"""
        if self._dummy_mode or self.model is None:
            # ダミーモードの場合、ロール情報を含まない応答を生成
            dummy_responses = [
                "そうですね、興味深い観点です。",
                "もう少し詳しく説明していただけますか？",
                "それについて考えてみましょう。",
                "別の視点から見るとどうでしょうか？",
                "具体的な例があるといいですね。"
            ]
            # プロンプトの長さに基づいて疑似ランダムに選択
            response_idx = len(prompt) % len(dummy_responses)
            return dummy_responses[response_idx]

        formatted_prompt = (
            self._format_gemma_prompt(prompt) if self._is_gemma_model() else prompt
        )

        print(f"[DEBUG] Generating text with prompt: {formatted_prompt[:100]}...")

        # llama‑cpp‑python (GGUF)
        if hasattr(self.model, "tokenize"):
            try:
                stop_sequences = stop or []
                if self._is_gemma_model():
                    stop_sequences = list(set(stop_sequences + ["<end_of_turn>"]))

                response = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences,
                )
                text = response["choices"][0]["text"]
                
                print(f"[DEBUG] Raw model response: {text}")

                if self._is_gemma_model():
                    for tag in ["<end_of_turn>", "<start_of_turn>", "model\n", "user\n"]:
                        text = text.replace(tag, "")
                
                cleaned_text = text.strip()
                print(f"[DEBUG] Cleaned response: {cleaned_text}")
                
                return cleaned_text
                
            except Exception as e:
                raise RuntimeError(f"GGUF text generation failed: {e}")

        # transformers (HF)
        else:
            try:
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repeat_penalty,
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 元のプロンプトを除去
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                print(f"[DEBUG] HF generated text: {generated_text}")
                return generated_text
                
            except Exception as e:
                raise RuntimeError(f"HF text generation failed: {e}")


# ======================================================================
# Boids 関連ユーティリティ
# ======================================================================

class BoidsCtx:
    """Boids アルゴリズムのコンテキスト"""

    def __init__(self, neighbors: List[str], summary_vec: np.ndarray):
        self.neighbors = neighbors
        self.summary_vec = summary_vec
        self.neighbor_vecs: Optional[np.ndarray] = None
        self._process_neighbors()

    def _process_neighbors(self):
        """近傍データからベクトル表現を計算"""
        if not self.neighbors:
            self.neighbor_vecs = np.zeros((0, 384))
            return
            
        # 実際のテキストベースでベクトルを生成
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = []
        for neighbor in self.neighbors:
            if isinstance(neighbor, dict) and 'text' in neighbor:
                texts.append(neighbor['text'])
            elif isinstance(neighbor, str):
                texts.append(neighbor)
            else:
                texts.append(str(neighbor))
        
        if not texts or all(not text.strip() for text in texts):
            self.neighbor_vecs = np.zeros((len(self.neighbors), 384))
            return
            
        try:
            vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            self.neighbor_vecs = tfidf_matrix.toarray()
            
            # ベクトル正規化
            norms = np.linalg.norm(self.neighbor_vecs, axis=1, keepdims=True)
            self.neighbor_vecs = self.neighbor_vecs / (norms + 1e-8)
        except Exception as e:
            print(f"[WARNING] TF-IDF vectorization failed: {e}")
            # フォールバック: 単語頻度ベースの単純ベクトル化
            self.neighbor_vecs = np.zeros((len(texts), 384))
            for i, text in enumerate(texts):
                words = text.lower().split()
                for j, word in enumerate(words[:384]):
                    self.neighbor_vecs[i][j % 384] = len(word) / 10.0            # 正規化
            norms = np.linalg.norm(self.neighbor_vecs, axis=1, keepdims=True)
            self.neighbor_vecs = self.neighbor_vecs / (norms + 1e-8)


def apply_boids(logits: np.ndarray, ctx: BoidsCtx, λ: Dict[str, float]) -> np.ndarray:
    """Boids ルールを logits に適用"""

    alignment = np.zeros_like(logits)
    if ctx.neighbor_vecs is not None and len(ctx.neighbor_vecs) > 0:
        mean_dir = np.mean(ctx.neighbor_vecs, axis=0)
        if len(mean_dir) != len(logits):
            mean_dir = np.tile(mean_dir, (len(logits) // len(mean_dir) + 1))[: len(logits)]
        alignment = mean_dir * logits

    cohesion = np.zeros_like(logits)
    if ctx.summary_vec is not None:
        if len(ctx.summary_vec) != len(logits):
            summary = np.tile(ctx.summary_vec, (len(logits) // len(ctx.summary_vec) + 1))[
                : len(logits)
            ]
        else:
            summary = ctx.summary_vec
        cohesion = summary * logits

    # Separation: 分離効果を計算（ランダムではなく、logitsの分散に基づく）
    separation = np.zeros_like(logits)
    if len(logits) > 1:
        # ログit値の標準偏差に基づいて分離を計算
        logits_std = np.std(logits)
        if logits_std > 0:
            # 標準偏差に基づいた分離効果（正規化された違い）
            normalized_logits = (logits - np.mean(logits)) / logits_std
            separation = normalized_logits * 0.1

    modified_logits = (
        logits
        + λ.get("λ_a", 0.3) * alignment
        + λ.get("λ_c", 0.3) * cohesion
        + λ.get("λ_s", 0.1) * separation
    )
    return modified_logits


def sample_top_p(logits: np.ndarray, top_p: float = 0.9) -> int:
    """Top‑p (nucleus) sampling"""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    # 数値オーバーフローを防ぐためにlogitsを正規化
    sorted_logits = sorted_logits - sorted_logits.max()
    sorted_probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
    cumulative_probs = np.cumsum(sorted_probs)

    keep_mask = cumulative_probs <= top_p
    if not np.any(keep_mask):
        keep_mask[0] = True  # 少なくとも一つ保持

    indices_to_keep = sorted_indices[keep_mask]
    keep_logits = logits[indices_to_keep] - logits[indices_to_keep].max()
    probs = np.exp(keep_logits) / np.sum(np.exp(keep_logits))
    return int(np.random.choice(indices_to_keep, p=probs))


# ======================================================================
# SLM エージェント本体
# ======================================================================

class SLMAgent:
    """Small Language Model Agent"""

    def __init__(self, id: int, role: str, model: LLM, tokenizer: Any, λ: Dict[str, float]):
        self.id = id
        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.λ = λ
        self.cache: Dict[str, Any] = {}

        random.seed(time.time() + id)
        self.alignment_weight = random.uniform(0.2, 0.5)
        self.separation_weight = random.uniform(0.1, 0.4)
        self.cohesion_weight = 1.0 - (self.alignment_weight + self.separation_weight)
        self.view_radius = random.uniform(3.0, 7.0)
        self.response_style = self._determine_response_style()

        print(
            f"[DEBUG] Agent {id} ({role}) initialized with weights: "
            f"a={self.alignment_weight:.2f}, s={self.separation_weight:.2f}, "
            f"c={self.cohesion_weight:.2f}, radius={self.view_radius:.2f}"
        )

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------
    def _determine_response_style(self) -> str:
        role_styles = {
            "質問者": "question_focused",
            "回答者": "solution_focused",
            "批評者": "analysis_focused",
            "批判者": "analysis_focused",
        }
        return role_styles.get(self.role, "neutral")

    def _build_role_specific_prompt(
        self, base_prompt: str, conversation_history: List[str]
    ) -> str:
        role_instructions = {
            "質問者": "あなたは好奇心旺盛な質問者です。新しい角度から質問を投げかけ、議論を深めてください。",
            "回答者": "あなたは知識豊富な回答者です。具体的で建設的な解決策や回答を提供してください。",
            "批評者": "あなたは冷静な批評者です。客観的な視点で分析し、改善点や問題点を指摘してください。",
            "批判者": "あなたは冷静な批評者です。客観的な視点で分析し、改善点や問題点を指摘してください。",
        }

        instruction = role_instructions.get(self.role, "あなたは議論に参加する一員です。建設的な貢献をしてください。")

        prompt = f"{instruction}\n\n"
        if conversation_history:
            prompt += "これまでの会話:\n" + "\n".join(conversation_history[-5:]) + "\n\n"
        prompt += f"あなた({self.role})としての次の発言:"

        print(f"[DEBUG] Agent {self.id} ({self.role}) prompt:\n{prompt}\n--- End Prompt ---")
        return prompt    # ------------------------------------------------------------------
    # 応答生成
    # ------------------------------------------------------------------
    async def generate(self, prompt: str, bb: "BlackBoard") -> str:
        """
        Boidsルールに基づいてテキスト生成を行う
        """
        neighbors = await bb.pull(k=16)
        # BlackBoardから実際の要約ベクトルを取得、なければゼロベクトル
        summary_vec = getattr(bb, "summary_vec", np.zeros(384))
        ctx = BoidsCtx(neighbors, summary_vec)

        print(f"[DEBUG] Agent {self.id} ({self.role}) received {len(neighbors)} neighbors from BlackBoard")

        # Boidsプロンプトプロセッサーを使用してプロンプトを構築
        from ..boids.prompt_processor import BoidsPromptProcessor
        processor = BoidsPromptProcessor(bb, self.λ)
        enhanced_prompt = await processor.process_prompt(prompt, self.id, self.role, k=16)

        print(f"[DEBUG] Agent {self.id} ({self.role}) enhanced prompt:\n{enhanced_prompt[:200]}...")

        try:
            generated_text = self.model.generate(
                enhanced_prompt,
                max_tokens=50,
                temperature=0.8,
                top_p=0.9,
                stop=[
                    "<end_of_turn>",
                    "\n\n",
                    "Human:",
                    "User:",
                    "Assistant:",
                    "質問者:",
                    "回答者:",
                    "批評者:",
                    "批判者:",
                ],
            )
            
            # テキストのクリーンアップ
            clean_text = generated_text.strip()
            for role in ["質問者", "回答者", "批評者", "批判者"]:
                if clean_text.startswith(role + ":"):
                    clean_text = clean_text[len(role) + 1:].strip()
            
            # 空の応答の場合は、役割に応じたフォールバック
            if not clean_text:
                clean_text = self._get_role_based_fallback()
            
            print(f"[DEBUG] Agent {self.id} ({self.role}) final response: {clean_text}")
            return clean_text
            
        except Exception as e:
            print(f"[ERROR] Agent {self.id} generate error: {e}")
            # モデルエラーの場合は、より具体的なフォールバック
            return self._get_role_based_fallback()
    
    def _get_role_based_fallback(self) -> str:
        """役割に基づいたフォールバック応答を取得"""
        fallback_responses = {
            "質問者": [
                "なぜそう思うのですか？",
                "他にはどんな可能性がありますか？", 
                "具体例を教えてください",
                "それはどのような理由からですか？"
            ],
            "回答者": [
                "それについて説明しましょう",
                "解決策を提案します", 
                "具体的には次のようになります",
                "私の理解では以下の通りです"
            ],
            "批評者": [
                "その点について検討が必要です",
                "別の視点から見ると",
                "改善の余地があります",
                "より詳細な分析が必要です"
            ],
            "批判者": [
                "その点について検討が必要です",
                "別の視点から見ると", 
                "改善の余地があります",
                "より詳細な分析が必要です"
            ]
        }
        
        responses = fallback_responses.get(self.role, ["興味深い観点ですね"])
        # エージェントIDに基づいて一意の応答を選択
        return responses[self.id % len(responses)]

    # ------------------------------------------------------------------
    # 会話ループ
    # ------------------------------------------------------------------
    async def run_conversation(
        self, initial_prompt: str, bb: "BlackBoard", max_turns: int = 10
    ) -> List[str]:
        conversation: List[str] = [initial_prompt]
        current_prompt = initial_prompt

        # 初期メッセージを BlackBoard へプッシュ
        await bb.push(
            {
                "agent_id": self.id,
                "role": self.role,
                "text": f"{self.role}として参加しました: {initial_prompt}",
                "timestamp": time.strftime("%H:%M:%S"),
                "type": "initial_prompt",
            }
        )

        for _ in range(max_turns):
            try:
                response = await self.generate(current_prompt, bb)
            except Exception as e:
                print(f"Error in conversation generation: {e}")
                response = "システムの処理中です..."

            conversation.append(response)
            current_prompt += f"\n{self.role}: {response}"

            message_data = {
                "agent_id": self.id,
                "role": self.role,
                "text": response,
                "timestamp": time.strftime("%H:%M:%S"),
                "type": "message",
                "response_style": self.response_style,
            }
            print(f"[DEBUG] Agent {self.id} pushing to BlackBoard: {message_data}")
            await bb.push(message_data)

        return conversation

"""
SLMAgent - Small Language Model Agent Core Implementation

Boids理論に基づく局所ルールで動作するSLMエージェントの実装。
ダミーモード削除、echo-back問題修正済み。
"""

import asyncio
import os
import random
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..memory.blackboard import BlackBoard

from .boids_enhancer import BoidsPromptEnhancer # Added import

# グローバルセマフォ - モデルへの同時アクセスを制御
_model_semaphore = asyncio.Semaphore(1)  # 同時アクセス数を1に制限


class LLM:
    """言語モデルのラッパークラス（ダミーモード削除済み）"""

    def __init__(
        self, model_path: str, threads: int = 4, quantize: str = "q4", n_ctx: int = 512
    ):
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.n_ctx = n_ctx
        self.model = None
        self.tokenizer = None
        self.use_dummy = False  # 互換用
        self._initialize()

    # ------------------------------------------------------------------
    # モデル初期化
    # ------------------------------------------------------------------

    def _initialize(self):
        """モデルの初期化 - ローカルGGUFファイルまたはHugging Faceモデルを使用"""
        # GPU を明示的に無効化
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            if self.model_path.endswith(".gguf"):
                self._initialize_gguf()
            else:
                self._initialize_hf()
        except Exception as e:
            error_detail = {
                "error_type": "ModelInitializationFailed",
                "model_path": self.model_path,
                "exception": str(e),
                "exception_type": type(e).__name__,
                "message": f"Model initialization failed: {e}",
            }
            raise RuntimeError(f"モデル読み込み失敗: {error_detail}")

    def _initialize_gguf(self):
        """GGUFファイルからモデルを初期化"""
        try:
            from llama_cpp import Llama

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
    # プロンプト整形
    # ------------------------------------------------------------------

    def _format_gemma_prompt(self, prompt: str) -> str:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _is_gemma_model(self) -> bool:
        return "gemma" in self.model_path.lower()

    # ------------------------------------------------------------------
    # テキスト生成
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:

        if self.model is None:
            error_detail = {
                "error_type": "ModelNotInitialized",
                "model_path": self.model_path,
                "message": "Model is not initialized. Cannot generate text.",
            }
            return f"[ERROR] {error_detail}"

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
        try:
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

            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].strip()

            print(f"[DEBUG] HF generated text: {generated_text}")
            return generated_text

        except Exception as e:
            raise RuntimeError(f"HF text generation failed: {e}")


# ----------------------------------------------------------------------
# SLM エージェント本体
# ----------------------------------------------------------------------


class SLMAgent:
    """Small Language Model Agent（echo-back問題修正済み）"""

    def __init__(
        self,
        id: int,
        role: str,
        model: LLM,
        tokenizer: Any,
        λ: Dict[str, float],
        name: Optional[str] = None,
        # Added blackboard and boids_config for BoidsPromptEnhancer
        blackboard: "BlackBoard",
        boids_config: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.role = role
        self.name = name if name else f"{role}_{id}"
        self.model = model
        self.tokenizer = tokenizer # This might be model.tokenizer if model is always LLM instance
        self.λ = λ
        self.cache: Dict[str, Any] = {}

        # Instantiate BoidsPromptEnhancer
        # LLM model and tokenizer can be passed from self.model if needed by enhancer in future
        self.boids_enhancer = BoidsPromptEnhancer(
            blackboard=blackboard,
            config=boids_config,
            llm_model=self.model, # Pass the agent's LLM model
            tokenizer=self.model.tokenizer if hasattr(self.model, 'tokenizer') else self.tokenizer # Pass the tokenizer
        )

        # Boids parameters (alignment_weight, etc.) are now managed by BoidsPromptEnhancer if needed,
        # or can be kept here if they influence other agent logic beyond prompt enhancement.
        # For now, let's assume they are not directly used by SLMAgent after refactoring.
        # self.alignment_weight = random.uniform(0.2, 0.5) # Example, remove if not used
        # self.separation_weight = random.uniform(0.1, 0.4) # Example, remove if not used
        # self.cohesion_weight = 1.0 - (self.alignment_weight + self.separation_weight) # Example, remove if not used
        # self.view_radius = random.uniform(3.0, 7.0) # Example, remove if not used
        self.response_style = self._determine_response_style()

        print(
            f"[DEBUG] Agent {id} ({self.name}/{role}) initialized."
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
        self, base_prompt: str, conversation_history: List[Dict[str, Any]]
    ) -> str:
        role_instructions = {
            "質問者": "あなたは好奇心旺盛な質問者です。新しい角度から質問を投げかけ、議論を深めてください。",
            "回答者": "あなたは知識豊富な回答者です。具体的で建設的な解決策や回答を提供してください。",
            "批評者": "あなたは冷静な批評者です。客観的な視点で分析し、改善点や問題点を指摘してください。",
            "批判者": "あなたは冷静な批評者です。客観的な視点で分析し、改善点や問題点を指摘してください。",
        }

        instruction = role_instructions.get(
            self.role, "あなたは議論に参加する一員です。建設的な貢献をしてください。"
        )

        prompt = f"{instruction}\n\n"
        if conversation_history:
            formatted_history = []
            for msg in conversation_history[-5:]:  # 最新5件
                if isinstance(msg, dict):
                    name = msg.get("agent_name", msg.get("name", "Unknown"))
                    role = msg.get("role", "")
                    text = msg.get("text", str(msg))
                    formatted_history.append(f"{name}({role}): {text}")
                else:
                    formatted_history.append(str(msg))

            prompt += "これまでの会話:\n" + "\n".join(formatted_history) + "\n\n"
        prompt += f"あなた({self.role})としての次の発言:"

        print(f"[DEBUG] Agent {self.id} ({self.role}) prompt:\n{prompt}\n--- End Prompt ---")
        return prompt    # ------------------------------------------------------------------
    # Boids 強化プロンプトを生成し、テキスト生成実行
    # ------------------------------------------------------------------

    async def run_conversation(self, initial_prompt: str, bb: "BlackBoard", max_turns: int = 10) -> None:
        """
        会話を実行するメインメソッド（echo-back問題修正済み）
        
        Parameters:
        -----------
        initial_prompt: 初期プロンプト
        bb: BlackBoardインスタンス  
        max_turns: 最大ターン数
        """
        print(f"[DEBUG] Agent {self.id} ({self.name}/{self.role}) starting conversation with max_turns: {max_turns}")
        
        try:
            for turn in range(max_turns):
                # BlackBoardから他のエージェントの発言を取得
                conversation_history = await bb.pull_messages_raw(k=10)
                
                # 役割特化プロンプトを生成
                base_agent_prompt = self._build_role_specific_prompt(initial_prompt, conversation_history)
                
                # Gather agent's own recent messages for the enhancer
                own_recent_messages = []
                if conversation_history: # conversation_history contains all recent messages
                    for msg in conversation_history: # Iterate from oldest to newest
                        if isinstance(msg, dict) and msg.get("agent_id") == self.id:
                            own_recent_messages.append({
                                "id": msg.get("id", str(msg.get("timestamp"))), # Ensure ID for enhancer
                                "text": msg.get("text", "")
                            })
                own_recent_messages = own_recent_messages[-3:] # Take last 3, or make configurable

                # Boidsアルゴリズムを適用してプロンプトを強化
                # Note: enhanced_prompt from _build_role_specific_prompt is now base_agent_prompt
                boids_enhanced_prompt = await self.boids_enhancer.enhance_prompt(
                    base_prompt=base_agent_prompt,
                    agent_role=self.role,
                    agent_id=self.id,
                    agent_recent_messages=own_recent_messages
                )
                
                # LLMで応答生成
                response = await self._generate_response(boids_enhanced_prompt)
                
                if response and response.strip():
                    # BlackBoardに応答を投稿（辞書形式）
                    message = {
                        "agent_id": self.id,
                        "agent_name": self.name,
                        "role": self.role,
                        "text": response.strip(),
                        "turn": turn,
                        "timestamp": time.time()
                    }
                    
                    await bb.push(message)
                    print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) turn {turn}: {response.strip()[:100]}...")
                else:
                    print(f"[WARNING] Agent {self.id} ({self.name}/{self.role}) generated empty response at turn {turn}")
                
                # ターン間隔の調整
                await asyncio.sleep(random.uniform(2.0, 5.0))
                
        except Exception as e:
            print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) conversation failed: {e}")
    async def _generate_response(self, prompt: str) -> str:
        """
        応答生成メソッド（非同期対応、セマフォで排他制御）
        
        Parameters:
        -----------
        prompt: 生成用プロンプト
        
        Returns:
        --------
        生成されたテキスト
        """
        global _model_semaphore
        
        # セマフォを取得してモデルアクセスを排他制御
        async with _model_semaphore:
            try:
                print(f"[DEBUG] Agent {self.id} ({self.name}/{self.role}) acquired model lock")
                
                # 非同期でLLMを実行
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    self.model.generate,
                    prompt,
                    50,  # max_tokens
                    0.7, # temperature
                    0.9, # top_p
                    1.1  # repeat_penalty
                )
                
                print(f"[DEBUG] Agent {self.id} ({self.name}/{self.role}) generated: {response[:100]}...")
                return response
                
            except Exception as e:
                print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) generation failed: {e}")
                return f"[ERROR] 応答生成エラー: {e}"
            finally:
                print(f"[DEBUG] Agent {self.id} ({self.name}/{self.role}) released model lock")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        同期的な生成メソッド（後方互換性のため）
        
        Parameters:
        -----------
        prompt: 生成用プロンプト
        **kwargs: 追加パラメータ
        
        Returns:
        --------
        生成されたテキスト
        """
        try:
            return self.model.generate(prompt, **kwargs)
        except Exception as e:
            print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) sync generation failed: {e}")
            return f"[ERROR] 応答生成エラー: {e}"

# Boids アルゴリズム関連メソッド have been moved to BoidsPromptEnhancer
# ------------------------------------------------------------------
# ユーティリティ関数
# ------------------------------------------------------------------

def sample_top_p(logits: np.ndarray, p: float = 0.9) -> int:
    """
    Top-p サンプリング
    
    Parameters:
    -----------
    logits: ロジット配列
    p: Top-p 閾値
    
    Returns:
    --------
    選択されたトークンのインデックス
    """
    try:
        # ソフトマックス適用
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        
        # 確率順でソート
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # 累積確率計算
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Top-p 閾値を超える部分をマスク
        mask = cumsum_probs <= p
        mask[0] = True  # 最低1つは選択
        
        valid_indices = sorted_indices[mask]
        valid_probs = sorted_probs[mask]
        
        # 正規化
        valid_probs = valid_probs / np.sum(valid_probs)
        
        # サンプリング
        selected_idx = np.random.choice(valid_indices, p=valid_probs)
        return selected_idx
        
    except Exception as e:
        print(f"[WARNING] Top-p sampling failed: {e}, using greedy selection")
        return np.argmax(logits)

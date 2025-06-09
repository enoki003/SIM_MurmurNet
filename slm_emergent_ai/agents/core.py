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
    from transformers import PreTrainedModel # For type hinting LogitsProcessor model

from transformers import LogitsProcessor # Actual import
from ..llm_extensions.boids_logits_processor import BoidsLogitsProcessor

# グローバルセマフォ - モデルへの同時アクセスを制御
_model_semaphore = asyncio.Semaphore(1)  # 同時アクセス数を1に制限


class LLM:
    """言語モデルのラッパークラス（ダミーモード削除済み）"""

    def __init__(
        self,
        model_path: str,
        threads: int = 4,
        quantize: str = "q4",
        n_ctx: int = 512,
        # BoidsLogitsProcessor related parameters for LLM class
        boids_enabled: bool = True,
        w_align: float = 0.1,
        w_sep: float = 0.1,
        w_cohesion: float = 0.1,
        n_align_tokens: int = 10,
        m_sep_tokens: int = 10,
        theta_sep: float = 0.8,
        cohesion_prompt_text: Optional[str] = None
    ):
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.n_ctx = n_ctx
        self.model: Optional[Any] = None # Can be HF PreTrainedModel or LlamaCpp model
        self.tokenizer: Optional[Any] = None # Can be HF Tokenizer or None
        self.use_dummy = False

        self.boids_enabled = boids_enabled
        self.w_align = w_align
        self.w_sep = w_sep
        self.w_cohesion = w_cohesion
        self.n_align_tokens = n_align_tokens
        self.m_sep_tokens = m_sep_tokens
        self.theta_sep = theta_sep
        self.cohesion_prompt_text = cohesion_prompt_text
        self.boids_processor: Optional[BoidsLogitsProcessor] = None

        self._initialize()

    def _initialize(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            if self.model_path.endswith(".gguf"):
                self._initialize_gguf()
            else:
                self._initialize_hf()
        except Exception as e:
            error_detail = {
                "error_type": "ModelInitializationFailed", "model_path": self.model_path,
                "exception": str(e), "exception_type": type(e).__name__,
                "message": f"Model initialization failed: {e}",
            }
            raise RuntimeError(f"モデル読み込み失敗: {error_detail}")

    def _initialize_gguf(self):
        try:
            from llama_cpp import Llama
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            print(f"Loading GGUF model: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path, n_threads=self.threads,
                n_ctx=self.n_ctx, verbose=False,
            )
            self.tokenizer = None
            print(f"GGUF model loaded successfully with {self.threads} threads, context length: {self.n_ctx}")
            if self.boids_enabled:
                self.boids_processor = None
                print("[WARNING] BoidsLogitsProcessor is currently not supported for GGUF models. Boids processing will be skipped.")
        except ImportError:
            raise ImportError("llama-cpp-python is required for GGUF models. Install it with: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {e}")

    def _initialize_hf(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"Loading HF model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, low_cpu_mem_usage=True, device_map="auto"
            )
            model_device = self.model.device if hasattr(self.model, 'device') else 'cpu'
            print(f"HF model loaded successfully on device: {model_device}")

            if self.boids_enabled:
                if self.model and self.tokenizer and not isinstance(self.model, str):
                    self.boids_processor = BoidsLogitsProcessor(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        w_align=self.w_align, w_sep=self.w_sep, w_cohesion=self.w_cohesion,
                        n_align_tokens=self.n_align_tokens, m_sep_tokens=self.m_sep_tokens,
                        theta_sep=self.theta_sep, cohesion_prompt_text=self.cohesion_prompt_text,
                        device=str(model_device)
                    )
                    print(f"[INFO] BoidsLogitsProcessor initialized for HF model on device {model_device}.")
                else:
                    print("[WARNING] HF Model (actual instance) or Tokenizer not available, BoidsLogitsProcessor cannot be initialized.")
                    self.boids_processor = None
        except ImportError:
            raise ImportError("transformers is required for Hugging Face models. Install it with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model: {e}")

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
        internal_logits_processors = []
        if self.boids_enabled and self.boids_processor:
            internal_logits_processors.append(self.boids_processor)

        if self.model is None:
            return f"[ERROR] ModelNotInitialized: Model is not initialized."

        formatted_prompt = self._format_gemma_prompt(prompt) if self._is_gemma_model() else prompt

        if hasattr(self.model, "create_completion"):
            try:
                stop_sequences = stop or []
                if self._is_gemma_model(): stop_sequences = list(set(stop_sequences + ["<end_of_turn>"]))
                response = self.model.create_completion(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    repeat_penalty=repeat_penalty, stop=stop_sequences,
                    logits_processor=internal_logits_processors if internal_logits_processors else None
                )
                text = response["choices"][0]["text"]
                if self._is_gemma_model():
                    for tag in ["<end_of_turn>", "<start_of_turn>", "model\n", "user\n"]: text = text.replace(tag, "")
                return text.strip()
            except Exception as e:
                raise RuntimeError(f"GGUF text generation failed: {e}")
        elif hasattr(self.model, 'generate') and self.tokenizer:
            try:
                model_device = self.model.device if hasattr(self.model, 'device') else 'cpu'
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(model_device)

                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=True,
                    temperature=temperature, top_p=top_p, repetition_penalty=repeat_penalty,
                    logits_processor=internal_logits_processors if internal_logits_processors else None,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
                )
                input_length = inputs.input_ids.shape[1]
                new_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return generated_text.strip()
            except Exception as e:
                raise RuntimeError(f"HF text generation failed: {e}")
        else:
            return "[ERROR] Model type not recognized for generation or tokenizer missing for HF."


class SLMAgent:
    def __init__(
        self,
        id: int,
        role: str,
        model: LLM,
        tokenizer: Any,
        name: Optional[str] = None,
    ):
        self.id = id
        self.role = role
        self.name = name if name else f"Agent_{id}"
        self.model_wrapper = model
        self.tokenizer = tokenizer
        self.cache: Dict[str, Any] = {}
        print(f"[DEBUG] Agent {id} ({self.name}/{self.role}) initialized.")

    def _build_role_specific_prompt(self, base_prompt: str, conversation_history: List[Dict[str, Any]]) -> str:
        instruction = "You are a helpful AI agent participating in a discussion. Please provide a thoughtful and constructive response based on the current task and conversation history."
        prompt_parts = [instruction]
        if conversation_history:
            formatted_history = []
            for msg in conversation_history[-5:]:
                if isinstance(msg, dict):
                    name_hist = msg.get("agent_name", msg.get("name", "Unknown"))
                    role_hist = msg.get("role", "")
                    text_hist = msg.get("text", str(msg))
                    formatted_history.append(f"{name_hist}({role_hist}): {text_hist}")
                else:
                    formatted_history.append(str(msg))
            if formatted_history:
                prompt_parts.append("これまでの会話 (Recent Conversation):\n" + "\n".join(formatted_history))
        
        prompt_parts.append(f"現在のタスク (Current Task): {base_prompt}")
        prompt_parts.append(f"あなた ({self.role}) としての応答 (Your response as {self.role}):")
        
        final_prompt = "\n\n".join(prompt_parts)
        return final_prompt

    async def run_conversation(self, initial_prompt: str, bb: "BlackBoard", max_turns: int = 10) -> None:
        current_task_prompt = initial_prompt
        print(f"[DEBUG] Agent {self.id} ({self.name}/{self.role}) starting conversation with max_turns: {max_turns}, initial task: '{current_task_prompt}'")
        try:
            for turn in range(max_turns):
                conversation_history = await bb.pull_messages_raw(k=10)
                prompt_for_llm = self._build_role_specific_prompt(current_task_prompt, conversation_history)
                response = await self._generate_response(prompt_for_llm)
                
                if response and response.strip():
                    message = {
                        "agent_id": self.id, "agent_name": self.name, "role": self.role,
                        "text": response.strip(), "turn": turn, "timestamp": time.time()
                    }
                    await bb.push(message)
                    print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) turn {turn}: {response.strip()[:100]}...")
                else:
                    print(f"[WARNING] Agent {self.id} ({self.name}/{self.role}) generated empty response at turn {turn}")
                await asyncio.sleep(random.uniform(2.0, 5.0))
        except Exception as e:
            print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) conversation failed: {e}")

    async def _generate_response(self, prompt: str) -> str:
        global _model_semaphore
        async with _model_semaphore:
            try:
                loop = asyncio.get_event_loop()
                generate_kwargs = {
                    "prompt": prompt, "max_tokens": 150, "temperature": 0.7,
                    "top_p": 0.9, "repeat_penalty": 1.1
                }
                response = await loop.run_in_executor(None, self.model_wrapper.generate, **generate_kwargs)
                return response
            except Exception as e:
                print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) generation failed: {e}")
                return f"[ERROR] Agent response generation error: {e}"

    def generate(self, prompt: str, **kwargs) -> str: # Sync version for compatibility
        try:
            return self.model_wrapper.generate(prompt, **kwargs)
        except Exception as e:
            print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) sync generation failed: {e}")
            return f"[ERROR] Agent sync response generation error: {e}"

def sample_top_p(logits: np.ndarray, p: float = 0.9) -> int:
    try:
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        mask = cumsum_probs <= p
        mask[0] = True
        valid_indices = sorted_indices[mask]
        valid_probs = sorted_probs[mask]
        valid_probs = valid_probs / np.sum(valid_probs)
        selected_idx = np.random.choice(valid_indices, p=valid_probs)
        return selected_idx
    except Exception as e:
        print(f"[WARNING] Top-p sampling failed: {e}, using greedy selection")
        return np.argmax(logits)

```

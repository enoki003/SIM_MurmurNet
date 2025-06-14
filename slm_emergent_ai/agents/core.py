"""
SLMAgent - Small Language Model Agent Core Implementation

This module defines the core classes for language model interaction (LLM)
and the agent behavior (SLMAgent).
The LLM class acts as a wrapper for different model types (GGUF, Hugging Face),
and integrates Boids-inspired logits processing.
The SLMAgent class defines the agent's lifecycle, prompt construction,
and interaction with the shared blackboard memory.
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
try:
    from llama_cpp import LogitsProcessorList # For GGUF Logits Processing
except ImportError:
    LogitsProcessorList = None  # Fallback if llama_cpp not available

try:
    from ..discussion_boids.manager import DiscussionBoidsManager
    from ..summarizer.conversation_summarizer import ConversationSummarizer
except ImportError:
    # Fallback for potential execution context issues if run directly
    from slm_emergent_ai.discussion_boids.manager import DiscussionBoidsManager
    from slm_emergent_ai.summarizer.conversation_summarizer import ConversationSummarizer

# Global semaphore to control simultaneous access to the model, preventing resource contention.
_model_semaphore = asyncio.Semaphore(1)


class LLM:
    """
    Language Model Wrapper (LLM).
    Handles loading of different model types (GGUF via llama-cpp-python, or Hugging Face Transformers)
    and provides a unified interface for text generation.
    """

    def __init__(
        self,
        model_path: str,
        threads: int = 4,
        quantize: str = "q4",
        n_ctx: int = 512,
        boids_enabled: bool = True,
        w_align: float = 0.1,
        w_sep: float = 0.1,
        w_cohesion: float = 0.1,
        n_align_tokens: int = 10,
        m_sep_tokens: int = 10,
        theta_sep: float = 0.8,
        cohesion_prompt: str = "",
        cohesion_prompt_text: str = ""
    ):
        """
        Initializes the LLM wrapper.

        Args:
            model_path (str): Path to the model file or Hugging Face model identifier.
            threads (int): Number of threads for GGUF model processing.
            quantize (str): Quantization level for the model (e.g., "q4").
            n_ctx (int): Context length for the model.
            boids_enabled (bool): Whether to enable boids behavior.
            w_align (float): Weight for alignment behavior.
            w_sep (float): Weight for separation behavior.            w_cohesion (float): Weight for cohesion behavior.
            n_align_tokens (int): Number of tokens for alignment.
            m_sep_tokens (int): Number of tokens for separation.
            theta_sep (float): Threshold for separation.
            cohesion_prompt (str): Prompt for cohesion behavior.
            cohesion_prompt_text (str): Text prompt for cohesion behavior.
        """
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.n_ctx = n_ctx
        self.boids_enabled = boids_enabled
        self.w_align = w_align
        self.w_sep = w_sep
        self.w_cohesion = w_cohesion
        self.n_align_tokens = n_align_tokens
        self.m_sep_tokens = m_sep_tokens
        self.theta_sep = theta_sep
        self.cohesion_prompt = cohesion_prompt
        self.cohesion_prompt_text = cohesion_prompt_text
        self.model = None
        self.tokenizer = None

        self._initialize()

    def _initialize(self):
        """Initializes the model (GGUF or Hugging Face)."""
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
            raise RuntimeError(f"LLM Initialization Failed: {error_detail}")

    def _initialize_gguf(self):
        """Initializes a GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"GGUF model file not found: {self.model_path}")
            print(f"Loading GGUF model: {self.model_path}")
            self.model = Llama( # This is the llama_cpp.Llama instance
                model_path=self.model_path, n_threads=self.threads,
                n_ctx=self.n_ctx, verbose=False,
                logits_all=True # Important: For logits_processor to receive logits
            )
            self.tokenizer = None # Stays None for GGUF in terms of HF tokenizer
            print(f"GGUF model loaded successfully (threads: {self.threads}, context: {self.n_ctx})")

        except ImportError:
            raise ImportError("llama-cpp-python is required for GGUF models. Install with: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model '{self.model_path}': {e}")

    def _initialize_hf(self):
        """Initializes a Hugging Face Transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print(f"Loading HF model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Check if CUDA is available, otherwise use CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            if device == "cpu":
                # Load model on CPU with optimizations for memory efficiency
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            else:
                # Load model on GPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            
            model_device = str(self.model.device if hasattr(self.model, 'device') else device)
            print(f"HF model loaded successfully on device: {model_device}")

        except ImportError:
            raise ImportError("transformers is required for Hugging Face models. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model '{self.model_path}': {e}")

    def _format_gemma_prompt(self, prompt: str) -> str:
        """Formats a prompt for Gemma models according to its specific chat template."""
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _is_gemma_model(self) -> bool:
        """Checks if the model path indicates a Gemma model."""
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
        """
        Generates text using the loaded language model.

        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            repeat_penalty (float): Penalty for repeating tokens.
            stop (Optional[List[str]]): List of stop sequences.

        Returns:
            str: The generated text.

        Raises:
            RuntimeError: If generation fails for any model type.
        """
        if self.model is None:
            return "[ERROR] ModelNotInitialized: Model is not initialized."

        formatted_prompt = self._format_gemma_prompt(prompt) if self._is_gemma_model() else prompt

        if hasattr(self.model, "create_completion"): # GGUF model
            try:
                # LogitsProcessorList is imported at the top

                final_logits_processor_arg = None

                stop_sequences = stop or []
                if self._is_gemma_model(): # This check might need refinement for GGUF Gemma
                    stop_sequences = list(set(stop_sequences + ["<end_of_turn>"]))

                response = self.model.create_completion(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences,
                    logits_processor=final_logits_processor_arg
                )
                text = response["choices"][0]["text"]
                if self._is_gemma_model(): # This check might need refinement for GGUF Gemma
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
                    logits_processor=None,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
                )
                input_length = inputs.input_ids.shape[1]
                new_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return generated_text.strip()
            except Exception as e:
                raise RuntimeError(f"HF text generation failed: {e}")
        else:
            return "[ERROR] Model type not recognized for generation or tokenizer missing for HF model."


class SLMAgent:
    """
    Small Language Model Agent (SLMAgent).
    Represents an individual agent in the simulation. It has a role (now generic),
    uses an LLM instance for text generation (which handles Boids logic internally),
    and interacts with other agents via a shared Blackboard.
    Its behavior during conversation is guided by a generic instruction and conversation history.
    """
    def __init__(
        self,
        id: int,
        role: str,
        model: LLM,
        tokenizer: Any, # Tokenizer from the LLM wrapper, used by SLMAgent for prompt construction.
        discussion_boids_manager: DiscussionBoidsManager, # Keep this, even if Optional for now
        summarizer: Optional[ConversationSummarizer],
        name: Optional[str] = None,
    ):
        """
        Initializes an SLMAgent.

        Args:
            id (int): Unique identifier for the agent.
            role (str): The role of the agent (typically "Agent").
            model (LLM): The LLM instance used for text generation.
            tokenizer (Any): The tokenizer associated with the LLM.
            discussion_boids_manager (DiscussionBoidsManager): Manager for Boids discussion rules.
            summarizer (Optional[ConversationSummarizer]): Summarizer for conversation history.
            name (Optional[str]): Optional name for the agent. Defaults to "Agent_{id}".
        """
        self.id = id
        self.role = role
        self.name = name if name else f"Agent_{id}"
        self.model_wrapper = model
        self.tokenizer = tokenizer
        self.discussion_boids_manager = discussion_boids_manager
        self.summarizer = summarizer
        self.cache: Dict[str, Any] = {}
        self.shutdown_flag = False  # シャットダウンフラグを追加
        print(f"[DEBUG] Agent {id} ({self.name}/{self.role}) initialized.")

    def _build_role_specific_prompt(self, base_task_prompt: str, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Constructs a prompt for the LLM based on a generic agent instruction,
        conversation history, and current task.

        Args:
            base_task_prompt (str): The fundamental task or question for the current turn.
            conversation_history (List[Dict[str, Any]]): Recent messages from the blackboard.

        Returns:
            str: The fully constructed prompt for the LLM.
        """
        # 会話口調の指示に変更
        instruction = "あなたは議論に参加するAIエージェントです。自然で親しみやすい口調で、思慮深い回答をしてください。専門用語を使う場合は分かりやすく説明し、相手との対話を大切にしながら発言してください。"

        prompt_parts = [instruction]

        if conversation_history: # conversation_history is now processed_history_for_prompt
            formatted_history = []
            for msg in conversation_history: # Iterate through what's passed (summary + recent verbatim)
                if isinstance(msg, dict):
                    sender_name = msg.get("agent_name", "エージェント")
                    text_content = msg.get("text", "")

                    if msg.get("type") == "summary":
                        # Summary text is already formatted by the summarizer, including its own internal newlines.
                        formatted_history.append(f"これまでの議論の要約：\n{text_content}")
                    else:
                        # Truncate long individual verbatim messages
                        if len(text_content) > 150: # Increased truncation limit for verbatim messages
                            text_content = text_content[:150] + "..."
                        formatted_history.append(f"{sender_name}：「{text_content}」")
                else: # Should ideally not happen if processed_history_for_prompt is well-formed
                    formatted_history.append(str(msg)[:150])

            if formatted_history:
                # Use a more generic header as it might contain summary + recent, or just recent.
                prompt_parts.append("会話の流れ：\n" + "\n".join(formatted_history))

        boids_directive_text = ""
        # Boids manager still uses the potentially summarized history.
        # This is acceptable as Boids manager analyzes text content, and summary is text.
        # The 'System' agent for summary will be correctly filtered by Boids manager if needed.
        if self.discussion_boids_manager and conversation_history:
            boids_directive_text = self.discussion_boids_manager.get_directive_for_agent(
                agent_id_to_prompt=self.name,
                recent_messages_history=conversation_history # Pass the processed history
            )

        # Truncate the task prompt if it's too long
        task_prompt = base_task_prompt[:200] + "..." if len(base_task_prompt) > 200 else base_task_prompt

        if boids_directive_text:
            prompt_parts.append(f"議論の方向性の提案：{boids_directive_text}")

        prompt_parts.append(f"現在のお題：{task_prompt}")
        prompt_parts.append("あなたの自然な会話での発言をお願いします：")

        final_prompt = "\n\n".join(prompt_parts)
        return final_prompt

    def _post_process_response(self, response: str) -> str:
        """
        生成された応答を会話口調に後処理する
        
        Args:
            response (str): 生成された応答
            
        Returns:
            str: 会話口調に調整された応答
        """
        if not response or not response.strip():
            return response
            
        # 基本的な後処理
        processed = response.strip()
        
        # 文末の調整（より会話的に）
        if processed and not processed.endswith(('。', '！', '？', 'ね', 'よ', 'です', 'ます')):
            if len(processed) > 20:  # 長い文の場合
                if not processed.endswith('.'):
                    processed += "ですね。"
            else:  # 短い文の場合
                processed += "。"
        
        # 冒頭の調整
        conversation_starters = ['そうですね、', 'なるほど、', 'たしかに、', 'そういえば、', 'ところで、']
        if len(processed) > 30 and not any(processed.startswith(starter) for starter in conversation_starters):
            if random.random() < 0.3:  # 30%の確率で会話的な接続詞を追加
                starter = random.choice(['そうですね、', 'なるほど、', 'たしかに、'])
                processed = starter + processed.lower()[0] + processed[1:] if processed else processed
        
        return processed

    async def run_conversation(self, initial_task_prompt: str, bb: "BlackBoard", max_turns: int = 10) -> None:
        """
        Runs the agent's conversation loop for a specified number of turns.
        The agent constructs a prompt using its generic role instruction and history,
        then calls the LLM (which has Boids logic) to generate a response.

        Args:
            initial_task_prompt (str): The initial prompt or task for the agent.
            bb (BlackBoard): The shared blackboard instance for message passing.
            max_turns (int): Maximum number of turns the agent will participate in.
        """
        current_task_prompt = initial_task_prompt
        print(f"[DEBUG] Agent {self.id} ({self.name}/{self.role}) starting conversation (max_turns: {max_turns}, initial_task: '{current_task_prompt:.50}...')")
        
        try:
            for turn in range(max_turns):
                print(f"[DEBUG] Agent {self.id} ({self.name}) starting turn {turn}...")
                # シャットダウンフラグをチェック（ローカルとグローバル）
                # Check for global shutdown from run_sim module
                import sys
                run_sim_module = sys.modules.get('slm_emergent_ai.run_sim') or sys.modules.get('__main__')
                global_shutdown = getattr(run_sim_module, 'shutdown_requested', False) if run_sim_module else False
                
                if self.shutdown_flag or global_shutdown:
                    print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) received shutdown signal, stopping conversation.")
                    break

                # --- History Processing with Summarization ---
                history_fetch_limit: int = 20  # How many messages to pull for potential summarization
                num_recent_verbatim: int = 3 # How many of the very latest messages to keep raw for detailed context
                                             # Also used by Boids manager if no summary.

                raw_messages_for_history = await bb.pull_messages_raw(k=history_fetch_limit)

                processed_history_for_prompt: List[Dict[str, Any]] = []
                summary_text = ""

                if self.summarizer and len(raw_messages_for_history) > self.summarizer.summarize_threshold:
                    # Ensure num_recent_verbatim doesn't exceed available messages if list is short but > threshold
                    actual_num_recent_verbatim = min(num_recent_verbatim, len(raw_messages_for_history))

                    messages_to_summarize = raw_messages_for_history[:-actual_num_recent_verbatim] if actual_num_recent_verbatim > 0 else raw_messages_for_history
                    recent_verbatim_messages = raw_messages_for_history[-actual_num_recent_verbatim:] if actual_num_recent_verbatim > 0 else []

                    if messages_to_summarize: # Only summarize if there are messages designated for summarization
                        summary_text = self.summarizer.summarize(messages_to_summarize)
                    
                    if summary_text:
                        processed_history_for_prompt.append({"agent_name": "System", "text": summary_text, "type": "summary"})
                    processed_history_for_prompt.extend(recent_verbatim_messages)

                else: # No summarizer, or not enough messages for the summarizer's threshold
                      # Fallback to a limited number of recent raw messages.
                    actual_num_recent_verbatim = min(num_recent_verbatim, len(raw_messages_for_history))
                    processed_history_for_prompt = raw_messages_for_history[-actual_num_recent_verbatim:]

                prompt_for_llm = self._build_role_specific_prompt(current_task_prompt, processed_history_for_prompt)
                print(f"[DEBUG] Agent {self.id} ({self.name}) turn {turn}: Generating response with prompt length {len(prompt_for_llm)}")
                response = await self._generate_response(prompt_for_llm)
                print(f"[DEBUG] Agent {self.id} ({self.name}) turn {turn}: Response generated: {response[:50] if response else 'None'}...")
                
                if response and response.strip():
                    message_to_post = {
                        "agent_id": self.id, "agent_name": self.name, "role": self.role,
                        "text": response.strip(), "turn": turn, "timestamp": time.time()
                    }
                    await bb.push(message_to_post)
                    print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) turn {turn}: {response.strip()[:100]}...")
                else:
                    print(f"[WARNING] Agent {self.id} ({self.name}/{self.role}) generated empty response at turn {turn}")

                # シャットダウンフラグを再チェック（スリープ前）
                if self.shutdown_flag:
                    print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) received shutdown signal, stopping conversation.")
                    break
                    
                await asyncio.sleep(random.uniform(2.0, 5.0))
        except asyncio.CancelledError:
            print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) conversation cancelled gracefully.")
        except Exception as e:
            print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) conversation failed: {e}")
        finally:
            print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) conversation ended.")

    async def _generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM, using a global semaphore for model access."""
        global _model_semaphore
        print(f"[DEBUG] Agent {self.id} ({self.name}) attempting to generate response...")
        async with _model_semaphore:
            try:
                print(f"[DEBUG] Agent {self.id} ({self.name}) acquired model semaphore, calling LLM...")
                loop = asyncio.get_event_loop()
                # Fix: Use partial function to pass keyword arguments properly
                from functools import partial
                generate_func = partial(
                    self.model_wrapper.generate,
                    prompt,  # Pass prompt as positional argument
                    max_tokens=150,
                    temperature=0.8,  # 少し高めの温度で自然さを向上
                    top_p=0.9,
                    repeat_penalty=1.1
                )
                response = await loop.run_in_executor(None, generate_func)
                
                # 会話口調への後処理を追加
                processed_response = self._post_process_response(response)
                
                print(f"[DEBUG] Agent {self.id} ({self.name}) LLM response received: {processed_response[:50] if processed_response else 'None'}...")
                return processed_response
            except Exception as e:
                print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) generation failed: {e}")
                return f"すみません、応答の生成でエラーが発生しました：{e}"

    def shutdown(self):
        """エージェントの安全なシャットダウンを開始"""
        self.shutdown_flag = True
        print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) shutdown initiated.")

    def generate(self, prompt: str, **kwargs) -> str: # Sync version for compatibility or testing
        """Synchronous version of text generation for compatibility or testing."""
        try:
            return self.model_wrapper.generate(prompt, **kwargs)
        except Exception as e:
            print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) sync generation failed: {e}")
            return f"[ERROR] Agent sync response generation error: {e}"

def sample_top_p(logits: np.ndarray, p: float = 0.9) -> int:
    """
    Top-p (nucleus) sampling.

    Args:
        logits (np.ndarray): Raw logits from the model.
        p (float): Cumulative probability threshold for nucleus sampling.

    Returns:
        int: Index of the selected token.
    """
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
        print(f"[WARNING] Top-p sampling failed: {e}, using greedy selection (argmax).")
        return np.argmax(logits)

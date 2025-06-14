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
from llama_cpp import LogitsProcessorList # For GGUF Logits Processing

try:
    from ..discussion_boids.manager import DiscussionBoidsManager # This might be unused now
    from ..summarizer.conversation_summarizer import ConversationSummarizer # Keep for type hinting if AgentHistoryManager needs it
    from .history_manager import AgentHistoryManager
except ImportError:
    # Fallback for potential execution context issues if run directly
    from slm_emergent_ai.discussion_boids.manager import DiscussionBoidsManager
    from slm_emergent_ai.summarizer.conversation_summarizer import ConversationSummarizer
    AgentHistoryManager = None # Or raise error if critical

# Global semaphore to control simultaneous access to the model, preventing resource contention.
_model_semaphore = asyncio.Semaphore(1)


class LLM:
    """
    Language Model Wrapper (LLM).
    Handles loading of different model types (GGUF via llama-cpp-python, or Hugging Face Transformers)
    and provides a unified interface for text generation.
    It also integrates BoidsLogitsProcessor for modifying token generation probabilities
    based on Boids-inspired rules, if enabled and configured.
    """

    def __init__(
        self,
        model_path: str,
        threads: int = 4,
        quantize: str = "q4",
        n_ctx: int = 512
    ):
        """
        Initializes the LLM wrapper.

        Args:
            model_path (str): Path to the model file or Hugging Face model identifier.
            threads (int): Number of threads for GGUF model processing.
            quantize (str): Quantization level for the model (e.g., "q4").
            n_ctx (int): Context length for the model.
        """
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.n_ctx = n_ctx
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

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
            # GGUFBoidsLogitsProcessorWrapper is imported at the top
            # LogitsProcessorList is imported at the top

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
            print(f"Loading HF model: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, low_cpu_mem_usage=True, device_map="auto"
            )
            model_device = str(self.model.device if hasattr(self.model, 'device') else 'cpu')
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
        history_manager: AgentHistoryManager,
        name: Optional[str] = None,
    ):
        """
        Initializes an SLMAgent.

        Args:
            id (int): Unique identifier for the agent.
            role (str): The role of the agent (typically "Agent").
            model (LLM): The LLM instance used for text generation.
            history_manager (AgentHistoryManager): Manages conversation history processing.
            name (Optional[str]): Optional name for the agent. Defaults to "Agent_{id}".
        """
        if AgentHistoryManager is None and history_manager is None: # Defensive check if fallback import was hit
            raise ImportError("AgentHistoryManager could not be imported and is required by SLMAgent.")

        self.id = id
        self.role = role
        self.name = name if name else f"Agent_{id}"
        self.model_wrapper = model
        self.history_manager = history_manager
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
        instruction = f"You are an AI agent with the role of {self.role}. Provide a thoughtful response."

        prompt_parts = [instruction]

        # conversation_history is actually processed_history_for_prompt from run_conversation
        formatted_history_strings = self.history_manager.format_history_for_prompt(conversation_history)

        if formatted_history_strings:
            # Use a more generic header as it might contain summary + recent, or just recent.
            prompt_parts.append("Conversation Context:\n" + "\n".join(formatted_history_strings))

        boids_directive_text = ""
        # Boids manager still uses the potentially summarized history.
        # This is acceptable as Boids manager analyzes text content, and summary is text.
        # The 'System' agent for summary will be correctly filtered by Boids manager if needed.

        # Truncate the task prompt if it's too long
        task_prompt = base_task_prompt[:200] + "..." if len(base_task_prompt) > 200 else base_task_prompt

        prompt_parts.append(f"Task: {task_prompt}")
        prompt_parts.append("Your response:")

        final_prompt = "\n\n".join(prompt_parts)
        return final_prompt

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
                # シャットダウンフラグをチェック
                if self.shutdown_flag:
                    print(f"[INFO] Agent {self.id} ({self.name}/{self.role}) received shutdown signal, stopping conversation.")
                    break

                # Use AgentHistoryManager to get processed history
                # These limits could be made configurable (e.g., from agent config or run_sim)
                processed_history_for_prompt = await self.history_manager.get_processed_history(
                    history_fetch_limit=20,
                    num_recent_verbatim=3
                )

                prompt_for_llm = self._build_role_specific_prompt(current_task_prompt, processed_history_for_prompt)
                response = await self._generate_response(prompt_for_llm)
                
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
        async with _model_semaphore:
            try:
                loop = asyncio.get_event_loop()
                # Fix: Use partial function to pass keyword arguments properly
                from functools import partial
                generate_func = partial(
                    self.model_wrapper.generate,
                    prompt,  # Pass prompt as positional argument
                    max_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1
                )
                response = await loop.run_in_executor(None, generate_func)
                return response
            except Exception as e:
                print(f"[ERROR] Agent {self.id} ({self.name}/{self.role}) generation failed: {e}")
                return f"[ERROR] Agent response generation error: {e}"

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

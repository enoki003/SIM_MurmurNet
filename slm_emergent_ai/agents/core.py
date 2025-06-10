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
from ..llm_extensions.boids_logits_processor import BoidsLogitsProcessor
from ..llm_extensions.gguf_boids_logits_processor_wrapper import GGUFBoidsLogitsProcessorWrapper
from llama_cpp import LogitsProcessorList # For GGUF Logits Processing

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
        n_ctx: int = 512,
        # BoidsLogitsProcessor related parameters
        boids_enabled: bool = True,
        w_align: float = 0.1,
        w_sep: float = 0.1,
        w_cohesion: float = 0.1,
        n_align_tokens: int = 10,
        m_sep_tokens: int = 10,
        theta_sep: float = 0.8,
        cohesion_prompt_text: Optional[str] = None
    ):
        """
        Initializes the LLM wrapper.

        Args:
            model_path (str): Path to the model file or Hugging Face model identifier.
            threads (int): Number of threads for GGUF model processing.
            quantize (str): Quantization level for the model (e.g., "q4").
            n_ctx (int): Context length for the model.
            boids_enabled (bool): Whether to enable BoidsLogitsProcessor.
            w_align (float): Weight for the alignment rule in Boids.
            w_sep (float): Weight for the separation rule in Boids.
            w_cohesion (float): Weight for the cohesion rule in Boids.
            n_align_tokens (int): Number of recent tokens to consider for alignment vector.
            m_sep_tokens (int): Number of recent tokens to consider for separation check.
            theta_sep (float): Similarity threshold for the separation rule.
            cohesion_prompt_text (Optional[str]): Text used to generate the initial cohesion vector.
        """
        self.model_path = model_path
        self.threads = threads
        self.quantize = quantize
        self.n_ctx = n_ctx
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

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
        """Initializes the model (GGUF or Hugging Face) and BoidsLogitsProcessor if enabled."""
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

            if self.boids_enabled:
                try:
                    print("[INFO] Attempting to initialize GGUFBoidsLogitsProcessorWrapper for GGUF model...")
                    
                    # GGUF時はcohesion_prompt_textを強制的にオフにする
                    cohesion_text_for_gguf = None
                    if self.cohesion_prompt_text is not None:
                        print("[WARNING] GGUF mode detected: cohesion_prompt_text is not supported and will be disabled.")
                        print(f"[WARNING] Original cohesion_prompt_text was: '{self.cohesion_prompt_text[:100]}...'")
                    
                    # w_cohesionが0より大きい場合も警告
                    if self.w_cohesion > 0:
                        print(f"[WARNING] GGUF mode: w_cohesion ({self.w_cohesion}) > 0 may not work properly without embedding matrix access.")
                        print("[WARNING] Consider setting w_cohesion to 0 in config for GGUF models.")
                    
                    self.boids_processor = GGUFBoidsLogitsProcessorWrapper(
                        gguf_model=self.model,
                        hf_tokenizer_for_cohesion=None, # Passing None for now, self.tokenizer is None for GGUF
                        w_align=self.w_align,
                        w_sep=self.w_sep,
                        w_cohesion=self.w_cohesion,
                        n_align_tokens=self.n_align_tokens,
                        m_sep_tokens=self.m_sep_tokens,
                        theta_sep=self.theta_sep,
                        cohesion_prompt_text=cohesion_text_for_gguf,  # Always None for GGUF
                        device='cpu'
                    )
                    if self.boids_processor and hasattr(self.boids_processor, 'boids_processor_internal') and self.boids_processor.boids_processor_internal is not None:
                        print("[INFO] GGUFBoidsLogitsProcessorWrapper initialized successfully for GGUF model.")
                    else:
                        # This condition implies GGUFBoidsLogitsProcessorWrapper was created, but its internal BoidsLogitsProcessor might have failed.
                        print("[WARNING] GGUFBoidsLogitsProcessorWrapper created, but its internal BoidsLogitsProcessor may not have initialized correctly. Boids rules might be skipped if internal processor is None.")
                        if not self.boids_processor: # If wrapper itself failed to init
                             self.boids_processor = None # Ensure it's None
                except Exception as e:
                    print(f"[WARNING] Failed to initialize GGUFBoidsLogitsProcessorWrapper for GGUF model: {e}. Boids processing will be skipped.")
                    self.boids_processor = None
            else:
                self.boids_processor = None # Ensure it's None if not enabled

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

            if self.boids_enabled:
                if self.model and self.tokenizer and hasattr(self.model, 'get_input_embeddings'): # Check for get_input_embeddings
                    self.boids_processor = BoidsLogitsProcessor(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        w_align=self.w_align, w_sep=self.w_sep, w_cohesion=self.w_cohesion,
                        n_align_tokens=self.n_align_tokens, m_sep_tokens=self.m_sep_tokens,
                        theta_sep=self.theta_sep, cohesion_prompt_text=self.cohesion_prompt_text,
                        device=model_device
                    )
                    print(f"[INFO] BoidsLogitsProcessor initialized for HF model on device {model_device}.")
                else:
                    print("[WARNING] HF Model (actual instance), Tokenizer, or get_input_embeddings method not available. BoidsLogitsProcessor cannot be initialized.")
                    self.boids_processor = None
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
        internal_logits_processors = []
        if self.boids_enabled and self.boids_processor:
            internal_logits_processors.append(self.boids_processor)

        if self.model is None:
            return "[ERROR] ModelNotInitialized: Model is not initialized."

        formatted_prompt = self._format_gemma_prompt(prompt) if self._is_gemma_model() else prompt

        if hasattr(self.model, "create_completion"): # GGUF model
            try:
                # LogitsProcessorList is imported at the top

                active_logits_processors = []
                # internal_logits_processors was previously defined, but GGUF needs specific handling
                if self.boids_enabled and self.boids_processor:
                    print("[DEBUG] LLM.generate: Adding GGUF Boids processor to list for GGUF model.")
                    active_logits_processors.append(self.boids_processor) # self.boids_processor is the GGUF wrapper

                final_logits_processor_arg = LogitsProcessorList(active_logits_processors) if active_logits_processors else None

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
        name: Optional[str] = None,
    ):
        """
        Initializes an SLMAgent.

        Args:
            id (int): Unique identifier for the agent.
            role (str): The role of the agent (typically "Agent").
            model (LLM): The LLM instance (which includes Boids config) used for text generation.
            tokenizer (Any): The tokenizer associated with the LLM (can be None for GGUF).
            name (Optional[str]): Optional name for the agent. Defaults to "Agent_{id}".
        """
        self.id = id
        self.role = role
        self.name = name if name else f"Agent_{id}"
        self.model_wrapper = model
        self.tokenizer = tokenizer
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
        instruction = "You are an AI agent in a discussion. Provide a thoughtful response."

        prompt_parts = [instruction]

        if conversation_history:
            formatted_history = []
            # Limit to only the last 2 messages to keep prompt short
            for msg in conversation_history[-2:]:
                if isinstance(msg, dict):
                    sender_name = msg.get("agent_name", "Agent")
                    text_content = msg.get("text", "")
                    # Truncate long messages
                    if len(text_content) > 100:
                        text_content = text_content[:100] + "..."
                    formatted_history.append(f"{sender_name}: {text_content}")
                else:
                    formatted_history.append(str(msg)[:100])
            if formatted_history:
                prompt_parts.append("Recent:\n" + "\n".join(formatted_history))

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
                    
                # Limit conversation history to reduce context usage
                conversation_history = await bb.pull_messages_raw(k=3)
                prompt_for_llm = self._build_role_specific_prompt(current_task_prompt, conversation_history)
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

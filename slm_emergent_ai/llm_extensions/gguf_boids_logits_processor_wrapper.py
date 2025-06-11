import numpy as np
import torch
from typing import Optional, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase # For type hinting if BoidsLogitsProcessor needs it

from .boids_logits_processor import BoidsLogitsProcessor

class GGUFBoidsLogitsProcessorWrapper:
    def __init__(
        self,
        gguf_model: Any,  # Should be llama_cpp.Llama instance
        hf_tokenizer_for_cohesion: Optional[PreTrainedTokenizerBase], # Allow passing a HF tokenizer if available
        w_align: float,
        w_sep: float,
        w_cohesion: float,
        n_align_tokens: int,
        m_sep_tokens: int,
        theta_sep: float,
        cohesion_prompt_text: Optional[str] = None,
        device: str = 'cpu' # GGUF is primarily CPU
    ):
        self.gguf_model = gguf_model # Keep for potential future use (e.g., direct embedding calls)
        self.device = device
        self.input_ids_dtype = np.intc
        self.scores_dtype = np.single

        try:
            print(f"[GGUF WRAPPER] Initializing BoidsLogitsProcessor for GGUF mode.")
            self.boids_processor_internal = BoidsLogitsProcessor(
                model=None,  # Pass None, as gguf_model is not a PreTrainedModel
                tokenizer=None, # Pass original tokenizer as None
                external_tokenizer=hf_tokenizer_for_cohesion, # Pass this to the new param
                external_embedding_matrix=None, # Explicitly None for now for GGUF
                is_gguf_mode=True, # Indicate GGUF context
                w_align=w_align,
                w_sep=w_sep,
                w_cohesion=w_cohesion,
                n_align_tokens=n_align_tokens,
                m_sep_tokens=m_sep_tokens,
                theta_sep=theta_sep,
                cohesion_prompt_text=cohesion_prompt_text,
                device=device
            )
            if self.boids_processor_internal:
                 print("[GGUF WRAPPER] BoidsLogitsProcessor initialized within wrapper for GGUF mode.")
            else:
                 print("[GGUF WRAPPER] BoidsLogitsProcessor internal instance is None after init attempt.")

        except Exception as e:
            print(f"[GGUF WRAPPER] Error initializing BoidsLogitsProcessor: {e}")
            self.boids_processor_internal = None

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if self.boids_processor_internal is None:
            return scores

        # Ensure input_ids and scores are NumPy arrays with correct dtypes
        # llama-cpp-python provides scores as float32. input_ids are typically int32.
        current_input_ids = np.array(input_ids, dtype=self.input_ids_dtype)
        current_scores = np.array(scores, dtype=self.scores_dtype)

        try:
            # Convert NumPy arrays to PyTorch tensors
            # Input IDs for BoidsLogitsProcessor are expected as (batch_size, sequence_length)
            # For llama-cpp-python, input_ids in the logits processor callback might be 1D array of current context tokens.
            # BoidsLogitsProcessor expects a batch dimension.

            # The input_ids from llama-cpp might be a flat list of tokens in the current context.
            # The BoidsLogitsProcessor expects a batch, e.g., (1, num_tokens).
            # We need to clarify the shape of input_ids provided by llama-cpp-python's callback.
            # Assuming input_ids is flat [context_len] and scores is [vocab_size] (for the next token).
            # BoidsLogitsProcessor's __call__ expects scores [batch_size, vocab_size].

            # Let's assume input_ids is [current_context_length] and scores is [vocab_size]
            # We need to make them 2D for BoidsLogitsProcessor: input_ids_pt [1, current_context_length], scores_pt [1, vocab_size]
            input_ids_pt = torch.from_numpy(current_input_ids).long().unsqueeze(0).to(self.device)
            scores_pt = torch.from_numpy(current_scores).float().unsqueeze(0).to(self.device)

            # Apply the Boids rules via the wrapped processor
            modified_scores_pt = self.boids_processor_internal(input_ids_pt, scores_pt)

            # Convert the modified PyTorch tensor back to a NumPy array, remove batch dim
            modified_scores_np = modified_scores_pt.squeeze(0).cpu().numpy()

            return modified_scores_np.astype(self.scores_dtype)

        except Exception as e:
            print(f"[GGUFBoidsLogitsProcessorWrapper] Error during Boids processing: {e}")
            # Return original scores in case of an error
            return current_scores

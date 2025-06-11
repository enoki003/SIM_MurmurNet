import torch
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizerBase
from typing import Optional, List
import torch.nn.functional as F
import numpy as np

class BoidsLogitsProcessor(LogitsProcessor):
    """
    A LogitsProcessor that modifies token generation logits based on Boids-inspired rules:
    Alignment, Separation, and Cohesion.
    It aims to guide text generation towards certain characteristics by biasing logits
    based on the semantic similarity of candidate tokens to contextual vectors derived
    from recent token history (for alignment and separation) or a predefined concept (for cohesion).
    This processor is designed to work with Hugging Face PreTrainedModel instances.
    It can handle inputs as PyTorch tensors or NumPy arrays (converting NumPy to PyTorch internally).
    """
    def __init__(
        self,
        model: Optional[PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizerBase],
        w_align: float,
        w_sep: float,
        w_cohesion: float,
        n_align_tokens: int,
        m_sep_tokens: int,
        theta_sep: float,
        cohesion_prompt_text: Optional[str] = None,
        device: str = 'cpu',
        external_embedding_matrix: Optional[torch.Tensor] = None,
        external_tokenizer: Optional[Any] = None,
        is_gguf_mode: bool = False
    ):
        """
        Initializes the BoidsLogitsProcessor.

        Args:
            model (Optional[PreTrainedModel]): The Hugging Face model instance. Can be None if external_embedding_matrix is provided or in GGUF mode.
            tokenizer (Optional[PreTrainedTokenizerBase]): The tokenizer. Can be None if external_tokenizer is provided.
            w_align (float): Weight for the alignment rule.
            w_sep (float): Weight for the separation rule.
            w_cohesion (float): Weight for the cohesion rule.
            n_align_tokens (int): Number of recent tokens in the input sequence to use for calculating the alignment vector.
            m_sep_tokens (int): Number of recent tokens in the input sequence to check against for separation.
            theta_sep (float): Similarity threshold for the separation rule.
            cohesion_prompt_text (Optional[str]): Text for `v_cohesion`.
            device (str): Device for tensor operations.
            external_embedding_matrix (Optional[torch.Tensor]): Pre-computed embedding matrix.
            external_tokenizer (Optional[Any]): An external tokenizer instance.
            is_gguf_mode (bool): Flag for GGUF model specific handling.
        """
        self.model = model
        self.original_tokenizer = tokenizer # Keep original if needed
        self.external_embedding_matrix = external_embedding_matrix
        self.external_tokenizer_ref = external_tokenizer # Store the external one
        self.tokenizer = external_tokenizer if external_tokenizer is not None else tokenizer # Effective tokenizer
        self.is_gguf_mode = is_gguf_mode
        self.device = device # Ensure device is set before using it for embedding_matrix

        self.embedding_matrix: Optional[torch.Tensor] = None
        if self.external_embedding_matrix is not None:
            self.embedding_matrix = self.external_embedding_matrix.to(self.device)
            print("[BoidsLogitsProcessor] Using provided external_embedding_matrix.")
        elif isinstance(self.model, PreTrainedModel) and hasattr(self.model, 'get_input_embeddings') and self.model.get_input_embeddings() is not None:
            try:
                self.embedding_matrix = self.model.get_input_embeddings().weight.detach().clone().to(self.device) # Added .clone()
                print("[BoidsLogitsProcessor] Successfully fetched and stored embedding_matrix from model.")
            except Exception as e:
                print(f"[BoidsLogitsProcessor] Error fetching embedding_matrix from model: {e}. It will remain None.")
        else:
            if not self.is_gguf_mode: # Only warn if not GGUF mode and no other source
                 print("[BoidsLogitsProcessor] Warning: Embedding matrix could not be obtained from model and no external one was provided. Rules requiring it will be skipped.")
            # For GGUF mode, this is expected if no external_embedding_matrix is passed.
            # The BoidsLogitsProcessor's own warning about missing external_embedding_matrix in GGUF mode should cover this.

        # Validation for model/embedding_matrix (adjusted)
        if not self.model and not self.embedding_matrix and not self.is_gguf_mode:
            raise ValueError("A Hugging Face PreTrainedModel ('model') that provides embeddings or an 'external_embedding_matrix' must be provided if not in GGUF mode and no internal matrix could be derived.")
        elif self.is_gguf_mode and not self.embedding_matrix : # Check effective embedding_matrix
             print("[BoidsLogitsProcessor] Warning: In GGUF mode, and no 'external_embedding_matrix' was effectively used/provided (self.embedding_matrix is None). Rules requiring vocab-wide similarity will be skipped.")
        elif self.model and not isinstance(self.model, PreTrainedModel) and not self.is_gguf_mode: # Original check for HF model type
             raise ValueError("BoidsLogitsProcessor requires a Hugging Face PreTrainedModel instance for 'model' when not in GGUF mode and not using external_embedding_matrix.")


        if cohesion_prompt_text and not self.tokenizer:
            raise ValueError("Cohesion prompt text provided, but no tokenizer available (original or external).")

        if cohesion_prompt_text and self.tokenizer and not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            print(f"[BoidsLogitsProcessor] Warning: Cohesion prompt text provided, but the tokenizer ({type(self.tokenizer)}) is not a PreTrainedTokenizerBase. Cohesion vector calculation might fail or be suboptimal.")

        self.w_align = w_align
        self.w_sep = w_sep
        self.w_cohesion = w_cohesion
        self.n_align_tokens = n_align_tokens
        self.m_sep_tokens = m_sep_tokens
        self.theta_sep = theta_sep
        self.cohesion_prompt_text = cohesion_prompt_text
        # self.device = device # Moved up
        self.v_cohesion: Optional[torch.Tensor] = None

        if self.cohesion_prompt_text and self.tokenizer:
            # Check if tokenizer is valid HF before precalculating, or try-catch inside _precalculate
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                self._precalculate_cohesion_vector()
            else:
                print(f"[BoidsLogitsProcessor] Cohesion prompt present but tokenizer type ({type(self.tokenizer)}) is not PreTrainedTokenizerBase. Skipping v_cohesion precalculation.")
                # Or, attempt with a try-catch in _precalculate_cohesion_vector if it can handle non-HF tokenizers
        elif self.cohesion_prompt_text and not self.tokenizer:
            print("[BoidsLogitsProcessor] Cohesion prompt text provided, but no tokenizer. Cohesion disabled.")


    def _get_token_embeddings(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Helper to get token embeddings for a list of token IDs."""
        # This method will need further adaptation for GGUF mode if self.model is Llama object
        # and external_embedding_matrix is not used for this specific call.
        # For now, it assumes self.model is a HF PreTrainedModel if external_embedding_matrix is not the source.
        if token_ids.numel() == 0:
            if self.external_embedding_matrix is not None:
                embedding_dim = self.external_embedding_matrix.shape[1]
            elif isinstance(self.model, PreTrainedModel) and hasattr(self.model, 'get_input_embeddings') and self.model.get_input_embeddings() is not None:
                embedding_dim = self.model.get_input_embeddings().weight.shape[1]
            else: # Fallback or error if no embedding source
                print("[BOIDS_DEBUG] _get_token_embeddings: Cannot determine embedding dimension (no model or external matrix). Returning zero tensor.")
                return torch.zeros(0, device=self.device) # Should indicate an issue
            return torch.zeros(embedding_dim, device=self.device)

        if isinstance(self.model, PreTrainedModel) and hasattr(self.model, 'get_input_embeddings') and self.model.get_input_embeddings() is not None:
            return self.model.get_input_embeddings()(token_ids).detach()
        # Placeholder for GGUF/Llama.cpp direct embedding access if external_embedding_matrix not used for this
        elif self.is_gguf_mode and hasattr(self.model, 'embed'):
             # This part is highly experimental and depends on how llama-cpp-python handles embeddings
             # And if Boids needs individual token embeddings vs a full matrix
            print("[BOIDS_DEBUG] _get_token_embeddings: Attempting GGUF model.embed() - Not fully implemented/tested for Boids needs here.")
            # embeddings = [self.model.embed(token_id.item()) for token_id in token_ids]
            # return torch.tensor(np.array(embeddings), device=self.device).float() # Requires llama_cpp.Llama.embed to return suitable format
            # For now, let's signify this path is not ready for general use in Boids rules if external matrix isn't used.
            raise NotImplementedError("_get_token_embeddings for GGUF model direct access is not fully implemented for Boids logic. Provide external_embedding_matrix or ensure model is HF for this path.")
        else:
            raise ValueError("_get_token_embeddings: No valid method to get token embeddings (model type or missing external matrix).")


    def _precalculate_cohesion_vector(self):
        """Pre-calculates `v_cohesion` from `cohesion_prompt_text`."""
        if not self.cohesion_prompt_text or not self.tokenizer:
            print("[BoidsLogitsProcessor] Cohesion prompt text or tokenizer not available. Cohesion disabled.")
            self.v_cohesion = None
            return

        # Ensure tokenizer is PreTrainedTokenizerBase for this HuggingFace-specific tokenization
        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            print(f"[BoidsLogitsProcessor] Warning: _precalculate_cohesion_vector requires a PreTrainedTokenizerBase, but got {type(self.tokenizer)}. Cohesion vector calculation skipped.")
            self.v_cohesion = None
            return

        try:
            inputs = self.tokenizer(
                self.cohesion_prompt_text, return_tensors="pt", padding=True, truncation=True,
                max_length=getattr(self.tokenizer, 'model_max_length', 512) # Use getattr for safety
            )
            input_ids_tensor = inputs.input_ids.to(self.device)
            if input_ids_tensor.numel() == 0:
                print("[BoidsLogitsProcessor] Tokenization of cohesion_prompt_text resulted in empty input_ids. Cohesion disabled.")
                self.v_cohesion = None; return

            # Use _get_token_embeddings which is now more flexible or will raise error if no source
            token_embeddings = self._get_token_embeddings(input_ids_tensor.squeeze(0)) # Squeeze batch dim if tokenizer adds it

            if token_embeddings.numel() > 0 and token_embeddings.ndim > 0 and token_embeddings.shape[0] > 0 :
                self.v_cohesion = torch.mean(token_embeddings, dim=0)
                self.v_cohesion = F.normalize(self.v_cohesion, p=2, dim=0)
                print(f"[BoidsLogitsProcessor] Pre-calculated v_cohesion (Device: {self.v_cohesion.device}, Shape: {self.v_cohesion.shape})")
            else:
                print("[BoidsLogitsProcessor] No valid token embeddings for cohesion_prompt_text. Cohesion disabled.")
                self.v_cohesion = None
        except NotImplementedError as nie:
            print(f"[BoidsLogitsProcessor] Error pre-calculating cohesion vector due to unimplemented path: {nie}. Cohesion disabled.")
            self.v_cohesion = None
        except Exception as e:
            print(f"[BoidsLogitsProcessor] Error during cohesion vector calculation for prompt '{self.cohesion_prompt_text[:50]}...': {type(e).__name__} - {str(e)}. Cohesion disabled.")
            self.v_cohesion = None

    def _cosine_similarity(self, v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculates cosine similarity: v1 is [D], v2 is [N, D] or [D]."""
        if v1.ndim == 1 and v2.ndim == 2:
            return F.cosine_similarity(v1.unsqueeze(0), v2, dim=1, eps=eps)
        elif v1.ndim == 1 and v2.ndim == 1:
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1, eps=eps).squeeze()
        return F.cosine_similarity(v1, v2, dim=-1, eps=eps)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Assuming inputs are PyTorch tensors for HF models
        current_processing_device = self.device # self.device should be set in __init__
        input_ids_tensor = input_ids.to(current_processing_device)
        scores_tensor = scores.to(current_processing_device)

        batch_size, vocab_size = scores_tensor.shape

        if self.embedding_matrix is None:
            if not self.is_gguf_mode: # Only log warning if not GGUF mode (where it might be expected)
                print("[BoidsLogitsProcessor] Warning: Embedding matrix is None. Skipping Boids rules application.")
            # For GGUF mode, this is logged during __init__ if matrix is missing.
            return scores_tensor # Return original scores if no embedding matrix

        for i in range(batch_size):
            current_sequence_input_ids = input_ids_tensor[i]

            # Debug: Print current input context (last few tokens)
            # Using input_ids (original, possibly numpy) for simpler printing if needed, or current_sequence_input_ids
            print(f"\n[BOIDS_DEBUG] --- Processing Step for input_ids (last 15): ...{current_sequence_input_ids[-15:].tolist()} ---")

            # --- Alignment ---
            if self.w_align != 0 and self.n_align_tokens > 0 and len(current_sequence_input_ids) > 0:
                # self.embedding_matrix is already checked to be not None at the start of __call__
                align_token_ids = current_sequence_input_ids[-self.n_align_tokens:]
                if align_token_ids.numel() > 0:
                    try:
                        v_align_tokens_embeddings = self._get_token_embeddings(align_token_ids)
                        if v_align_tokens_embeddings.numel() > 0 and v_align_tokens_embeddings.shape[0] > 0:
                            # Ensure v_align_tokens_embeddings is 2D for mean calculation if it's a single token embedding
                            current_v_align_embeddings = v_align_tokens_embeddings
                            if current_v_align_embeddings.ndim == 1:
                                current_v_align_embeddings = current_v_align_embeddings.unsqueeze(0)

                            v_align = torch.mean(current_v_align_embeddings, dim=0)
                            v_align = F.normalize(v_align, p=2, dim=0) # Normalize v_align

                            align_similarities = self._cosine_similarity(v_align, self.embedding_matrix) # Sim(v_c, v_align)
                            alignment_bias = self.w_align * align_similarities
                            scores_tensor[i] += alignment_bias
                            # print(f"[BOIDS_DEBUG] Alignment: Applied. Avg bias: {alignment_bias.mean().item():.4f}")
                        else:
                            # print(f"[BOIDS_DEBUG] Alignment: Skipped (no valid embeddings for align_token_ids).")
                            pass # Silently skip if no valid embeddings
                    except Exception as e:
                        print(f"[BOIDS_DEBUG] Alignment: Skipped due to error: {type(e).__name__} - {e}")
                # else:
                    # print(f"[BOIDS_DEBUG] Alignment: Skipped (no align_token_ids).")
            # else:
                # print(f"[BOIDS_DEBUG] Alignment: Skipped (w_align is 0 or n_align_tokens is 0 or no input tokens).")

            # --- Separation ---
            if self.w_sep != 0 and self.m_sep_tokens > 0 and len(current_sequence_input_ids) > 0:
                # self.embedding_matrix is already checked to be not None
                sep_token_ids = current_sequence_input_ids[-self.m_sep_tokens:]
                if sep_token_ids.numel() > 0:
                    try:
                        history_embeddings = self._get_token_embeddings(sep_token_ids)
                        if history_embeddings.numel() > 0 and history_embeddings.shape[0] > 0:
                            current_history_embeddings = history_embeddings
                            if current_history_embeddings.ndim == 1: # Ensure 2D for transpose
                                current_history_embeddings = current_history_embeddings.unsqueeze(0)

                            # Normalize both sets of embeddings for cosine similarity via matmul
                            norm_embedding_matrix = F.normalize(self.embedding_matrix, p=2, dim=1)
                            norm_history_embeddings = F.normalize(current_history_embeddings, p=2, dim=1)

                            all_sims_to_history = torch.matmul(norm_embedding_matrix, norm_history_embeddings.transpose(0, 1))
                            max_similarity_to_history, _ = torch.max(all_sims_to_history, dim=1)

                            mask = max_similarity_to_history > self.theta_sep
                            num_penalized = mask.sum().item()
                            if num_penalized > 0:
                                # P_sep(c) = w_sep * (Sim_max(c, H) - theta_sep) / (1 - theta_sep)
                                denominator = (1.0 - self.theta_sep + 1e-8) # Add epsilon for stability
                                penalty_values = self.w_sep * ((max_similarity_to_history[mask] - self.theta_sep) / denominator)
                                penalty_values = torch.clamp(penalty_values, min=0) # Ensure penalty is not negative
                                scores_tensor[i, mask] -= penalty_values
                                # print(f"[BOIDS_DEBUG] Separation: Penalized {num_penalized} tokens. Avg penalty: {penalty_values.mean().item():.4f}")
                            # else:
                                # print(f"[BOIDS_DEBUG] Separation: No tokens exceeded theta_sep.")
                        else:
                            # print(f"[BOIDS_DEBUG] Separation: Skipped (no valid embeddings for sep_token_ids).")
                            pass
                    except Exception as e:
                        print(f"[BOIDS_DEBUG] Separation: Skipped due to error: {type(e).__name__} - {e}")
                # else:
                    # print(f"[BOIDS_DEBUG] Separation: Skipped (no sep_token_ids).")
            # else:
                # print(f"[BOIDS_DEBUG] Separation: Skipped (w_sep is 0 or m_sep_tokens is 0 or no input tokens).")

            # --- Cohesion ---
            if self.w_cohesion != 0 and self.v_cohesion is not None and self.v_cohesion.numel() > 0:
                # self.embedding_matrix is already checked
                try:
                    v_cohesion_on_device = self.v_cohesion.to(current_processing_device) # Ensure v_cohesion is on the correct device
                    cohesion_similarities = self._cosine_similarity(v_cohesion_on_device, self.embedding_matrix) # Sim(v_c, v_cohesion)
                    cohesion_bias = self.w_cohesion * cohesion_similarities
                    scores_tensor[i] += cohesion_bias
                    # print(f"[BOIDS_DEBUG] Cohesion: Applied. Avg bias: {cohesion_bias.mean().item():.4f}")
                except Exception as e:
                    print(f"[BOIDS_DEBUG] Cohesion: Skipped due to error: {type(e).__name__} - {e}")
            # else:
                # print(f"[BOIDS_DEBUG] Cohesion: Skipped (w_cohesion is 0 or v_cohesion not available).")

        # For HF models, the pipeline expects PyTorch tensors
        return scores_tensor

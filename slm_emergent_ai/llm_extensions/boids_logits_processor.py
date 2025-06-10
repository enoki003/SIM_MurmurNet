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
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        w_align: float,
        w_sep: float,
        w_cohesion: float,
        n_align_tokens: int,
        m_sep_tokens: int,
        theta_sep: float,
        cohesion_prompt_text: Optional[str] = None,
        device: str = 'cpu'
    ):
        """
        Initializes the BoidsLogitsProcessor.

        Args:
            model (PreTrainedModel): The Hugging Face model instance (used for accessing token embeddings).
            tokenizer (PreTrainedTokenizerBase): The tokenizer associated with the model.
            w_align (float): Weight for the alignment rule.
            w_sep (float): Weight for the separation rule.
            w_cohesion (float): Weight for the cohesion rule.
            n_align_tokens (int): Number of recent tokens in the input sequence to use for calculating the alignment vector.
            m_sep_tokens (int): Number of recent tokens in the input sequence to check against for separation.
            theta_sep (float): Similarity threshold for the separation rule. Penalties are applied to tokens
                               whose max similarity to recent history exceeds this threshold.
            cohesion_prompt_text (Optional[str]): Text used to generate a fixed cohesion vector (`v_cohesion`).
                                                  If None or empty, cohesion is disabled.
            device (str): The device ('cpu', 'cuda', etc.) on which to perform tensor operations.
        """
        if not isinstance(model, PreTrainedModel): # Check if it's an actual PreTrainedModel, not a wrapper
            raise ValueError("BoidsLogitsProcessor requires a Hugging Face PreTrainedModel instance.")
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            if cohesion_prompt_text: # Tokenizer is essential if cohesion_prompt_text is to be processed
                raise ValueError("BoidsLogitsProcessor requires a Hugging Face PreTrainedTokenizerBase instance if cohesion_prompt_text is provided.")
            else:
                # If no cohesion_prompt_text, tokenizer might not be strictly needed by processor itself, but good to warn.
                print("[BoidsLogitsProcessor] Warning: Tokenizer is not a PreTrainedTokenizerBase instance. Cohesion via text prompt will be disabled if text was provided.")

        self.model = model
        self.tokenizer = tokenizer
        self.w_align = w_align
        self.w_sep = w_sep
        self.w_cohesion = w_cohesion
        self.n_align_tokens = n_align_tokens
        self.m_sep_tokens = m_sep_tokens
        self.theta_sep = theta_sep
        self.cohesion_prompt_text = cohesion_prompt_text
        self.device = device
        self.v_cohesion: Optional[torch.Tensor] = None

        if self.cohesion_prompt_text and self.tokenizer:
            self._precalculate_cohesion_vector()
        elif self.cohesion_prompt_text and not self.tokenizer:
            print("[BoidsLogitsProcessor] Cohesion prompt text provided, but tokenizer is missing or invalid. Cohesion disabled.")


    def _get_token_embeddings(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Helper to get token embeddings for a list of token IDs."""
        if token_ids.numel() == 0:
            embedding_dim = self.model.get_input_embeddings().weight.shape[1]
            return torch.zeros(embedding_dim, device=self.device)
        return self.model.get_input_embeddings()(token_ids).detach()

    def _precalculate_cohesion_vector(self):
        """Pre-calculates `v_cohesion` from `cohesion_prompt_text`."""
        if not self.cohesion_prompt_text or not self.tokenizer:
            print("[BoidsLogitsProcessor] Cohesion prompt text or tokenizer not available. Cohesion disabled.")
            self.v_cohesion = None
            return
        try:
            inputs = self.tokenizer(
                self.cohesion_prompt_text, return_tensors="pt", padding=True, truncation=True,
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length else 512
            )
            input_ids_tensor = inputs.input_ids.to(self.device)
            if input_ids_tensor.numel() == 0:
                print("[BoidsLogitsProcessor] Tokenization of cohesion_prompt_text resulted in empty input_ids. Cohesion disabled.")
                self.v_cohesion = None; return
            token_embeddings = self._get_token_embeddings(input_ids_tensor.squeeze(0))
            if token_embeddings.ndim > 0 and token_embeddings.shape[0] > 0 :
                self.v_cohesion = torch.mean(token_embeddings, dim=0)
                self.v_cohesion = F.normalize(self.v_cohesion, p=2, dim=0)
                print(f"[BoidsLogitsProcessor] Pre-calculated v_cohesion (Device: {self.v_cohesion.device}, Shape: {self.v_cohesion.shape})")
            else:
                print("[BoidsLogitsProcessor] No valid token embeddings for cohesion_prompt_text. Cohesion disabled.")
                self.v_cohesion = None
        except Exception as e:
            print(f"[BoidsLogitsProcessor] Error pre-calculating cohesion vector: {e}. Cohesion disabled.")
            self.v_cohesion = None

    def _cosine_similarity(self, v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculates cosine similarity: v1 is [D], v2 is [N, D] or [D]."""
        if v1.ndim == 1 and v2.ndim == 2:
            return F.cosine_similarity(v1.unsqueeze(0), v2, dim=1, eps=eps)
        elif v1.ndim == 1 and v2.ndim == 1:
            return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1, eps=eps).squeeze()
        return F.cosine_similarity(v1, v2, dim=-1, eps=eps)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        original_type_is_numpy = isinstance(scores, np.ndarray)
        current_processing_device = self.device
        if original_type_is_numpy:
            input_ids_np = input_ids # Keep original if needed for logging
            if not isinstance(input_ids, torch.Tensor):
                # Handle list of lists or simple list for input_ids from NumPy context
                if input_ids and isinstance(input_ids, np.ndarray):
                     input_ids_tensor = torch.from_numpy(input_ids).long().to(current_processing_device)
                elif input_ids and isinstance(input_ids[0], list): # batch of lists
                     input_ids_tensor = torch.LongTensor(input_ids).to(current_processing_device)
                else: # single list for batch size 1
                     input_ids_tensor = torch.LongTensor([input_ids]).to(current_processing_device)
            else:
                input_ids_tensor = input_ids.to(current_processing_device)
            scores_tensor = torch.from_numpy(scores.astype(np.float32)).to(current_processing_device)
        else:
            input_ids_tensor = input_ids.to(current_processing_device)
            scores_tensor = scores.to(current_processing_device)

        batch_size, vocab_size = scores_tensor.shape

        if not hasattr(self.model, 'get_input_embeddings'):
            # This check is crucial if this processor is ever mistakenly used with a non-HF model type
            # that was not filtered out by LLM class's GGUF handling.
            print("[BoidsLogitsProcessor] Model does not support `get_input_embeddings`. Skipping Boids processing.")
            return scores

        embedding_matrix = self.model.get_input_embeddings().weight.detach().to(current_processing_device)

        for i in range(batch_size):
            current_sequence_input_ids = input_ids_tensor[i]

            # Debug: Print current input context (last few tokens)
            # Using input_ids (original, possibly numpy) for simpler printing if needed, or current_sequence_input_ids
            print(f"\n[BOIDS_DEBUG] --- Processing Step for input_ids (last 15): ...{current_sequence_input_ids[-15:].tolist()} ---")

            # --- Alignment ---
            if self.w_align != 0 and self.n_align_tokens > 0 and len(current_sequence_input_ids) > 0:
                align_token_ids = current_sequence_input_ids[-self.n_align_tokens:]
                if align_token_ids.numel() > 0:
                    v_align_tokens_embeddings = self._get_token_embeddings(align_token_ids)
                    if v_align_tokens_embeddings.numel() > 0 and v_align_tokens_embeddings.shape[0] > 0 :
                        v_align = torch.mean(v_align_tokens_embeddings, dim=0) if v_align_tokens_embeddings.ndim > 1 else v_align_tokens_embeddings
                        v_align = F.normalize(v_align, p=2, dim=0)
                        align_similarities = self._cosine_similarity(v_align, embedding_matrix)
                        alignment_bias = self.w_align * align_similarities
                        scores_tensor[i] += alignment_bias
                        print(f"[BOIDS_DEBUG] Alignment: Applied. Avg bias: {alignment_bias.mean().item():.4f}, Max bias: {alignment_bias.max().item():.4f}, Min bias: {alignment_bias.min().item():.4f}")
                    else:
                        print(f"[BOIDS_DEBUG] Alignment: Skipped (no valid embeddings for align_token_ids).")
                else:
                    print(f"[BOIDS_DEBUG] Alignment: Skipped (no align_token_ids).")
            else:
                print(f"[BOIDS_DEBUG] Alignment: Skipped (w_align is 0 or n_align_tokens is 0 or no input tokens).")


            # --- Separation ---
            if self.w_sep != 0 and self.m_sep_tokens > 0 and len(current_sequence_input_ids) > 0:
                sep_token_ids = current_sequence_input_ids[-self.m_sep_tokens:]
                if sep_token_ids.numel() > 0:
                    history_embeddings = self._get_token_embeddings(sep_token_ids)
                    if history_embeddings.numel() > 0 and history_embeddings.shape[0] > 0:
                        if history_embeddings.ndim == 1: history_embeddings = history_embeddings.unsqueeze(0)
                        if history_embeddings.ndim == 2:
                            norm_embedding_matrix = F.normalize(embedding_matrix, p=2, dim=1)
                            norm_history_embeddings = F.normalize(history_embeddings, p=2, dim=1)
                            all_sims_to_history = torch.matmul(norm_embedding_matrix, norm_history_embeddings.transpose(0, 1))
                            max_similarity_to_history, _ = torch.max(all_sims_to_history, dim=1)
                            mask = max_similarity_to_history > self.theta_sep
                            num_penalized = mask.sum().item()
                            if num_penalized > 0:
                                denominator = (1.0 - self.theta_sep)
                                if abs(denominator) < 1e-6 :
                                    denominator = 1e-6 * torch.sign(torch.tensor(denominator, device=current_processing_device))
                                penalty_values = self.w_sep * ((max_similarity_to_history[mask] - self.theta_sep) / denominator)
                                penalty_values = torch.clamp(penalty_values, min=0)
                                scores_tensor[i, mask] -= penalty_values
                                print(f"[BOIDS_DEBUG] Separation: Penalized {num_penalized} tokens. Avg penalty: {penalty_values.mean().item() if num_penalized > 0 else 0:.4f}, Max penalty: {penalty_values.max().item() if num_penalized > 0 else 0:.4f}")
                            else:
                                print(f"[BOIDS_DEBUG] Separation: No tokens exceeded theta_sep ({self.theta_sep:.2f}). Max similarity found: {max_similarity_to_history.max().item() if max_similarity_to_history.numel() > 0 else 'N/A':.4f}.")
                        else:
                            print(f"[BOIDS_DEBUG] Separation: Skipped (history_embeddings not 2D after processing).")
                    else:
                        print(f"[BOIDS_DEBUG] Separation: Skipped (no valid embeddings for sep_token_ids).")
                else:
                    print(f"[BOIDS_DEBUG] Separation: Skipped (no sep_token_ids).")
            else:
                print(f"[BOIDS_DEBUG] Separation: Skipped (w_sep is 0 or m_sep_tokens is 0 or no input tokens).")


            # --- Cohesion ---
            if self.w_cohesion != 0 and self.v_cohesion is not None and self.v_cohesion.numel() > 0:
                v_cohesion_on_device = self.v_cohesion.to(current_processing_device)
                cohesion_similarities = self._cosine_similarity(v_cohesion_on_device, embedding_matrix)
                cohesion_bias = self.w_cohesion * cohesion_similarities
                scores_tensor[i] += cohesion_bias
                print(f"[BOIDS_DEBUG] Cohesion: Applied. Avg bias: {cohesion_bias.mean().item():.4f}, Max bias: {cohesion_bias.max().item():.4f}, Min bias: {cohesion_bias.min().item():.4f}")
            else:
                print(f"[BOIDS_DEBUG] Cohesion: Skipped (w_cohesion is 0 or v_cohesion not available).")

        if original_type_is_numpy:
            return scores_tensor.cpu().numpy()
        else:
            return scores_tensor

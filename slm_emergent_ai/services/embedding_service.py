from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import re

class EmbeddingService:
    """
    Provides text embedding generation services.
    Currently uses a simple TF-IDF-like approach.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the EmbeddingService.

        Args:
            config: Configuration dictionary for the embedding service.
                    Currently unused, but can hold parameters for future models.
                    Expected keys might include:
                    "model_name": Name of the sentence transformer model.
                    "dimension": Output dimension of the embedding.
        """
        self.config = config if config is not None else {}
        # Default dimension, matching the one previously in BlackBoard
        self.dimension = self.config.get("dimension", 384)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for the given text using a simple TF-IDF-like approach.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not isinstance(text, str):
            return [0.0] * self.dimension

        try:
            # Simple TF-IDF-like implementation (taken from BlackBoard._update_summary)
            # 1. Normalize text: lowercase and remove punctuation
            text = re.sub(r'[^\w\s]', '', text.lower())
            words = text.split()

            if not words:
                return [0.0] * self.dimension

            word_freq: Dict[str, int] = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            # Create a fixed-size vector based on word frequency
            # Sort words to make the vector somewhat consistent for similar vocabularies
            # Note: This is a very basic approach. For robust embeddings, a pre-trained model is needed.
            vector = np.zeros(self.dimension, dtype=np.float32)

            # Use sorted keys for some consistency, take top N words that fit dimension
            sorted_unique_words = sorted(word_freq.keys())

            for i, word in enumerate(sorted_unique_words):
                if i >= self.dimension:
                    break
                vector[i] = float(word_freq[word])

            # Normalize the vector (L2 norm)
            norm = np.linalg.norm(vector)
            if norm > 0:
                normalized_vector = vector / norm
            else:
                # Avoid division by zero if vector is all zeros (e.g., empty or stopword-only text)
                normalized_vector = np.zeros(self.dimension, dtype=np.float32)

            return normalized_vector.tolist()

        except Exception as e:
            print(f"Error generating TF-IDF-like embedding: {e}")
            # Fallback to a zero vector in case of any error
            return [0.0] * self.dimension

# Example Usage:
if __name__ == "__main__":
    service = EmbeddingService()

    text1 = "This is a sample sentence for testing."
    embedding1 = service.generate_embedding(text1)
    print(f"Embedding for '{text1}':\nLength: {len(embedding1)}\nSample: {embedding1[:10]}...")

    text2 = "Another example sentence with different words."
    embedding2 = service.generate_embedding(text2)
    print(f"Embedding for '{text2}':\nLength: {len(embedding2)}\nSample: {embedding2[:10]}...")

    text_empty = ""
    embedding_empty = service.generate_embedding(text_empty)
    print(f"Embedding for empty text:\nLength: {len(embedding_empty)}\nSample: {embedding_empty[:10]}...")

    text_jp = "これはテスト用のサンプル文です。"
    embedding_jp = service.generate_embedding(text_jp)
    print(f"Embedding for '{text_jp}':\nLength: {len(embedding_jp)}\nSample: {embedding_jp[:10]}...")

    # Check if different texts produce different embeddings
    if embedding1 != embedding2:
        print("Embeddings for different texts are different (as expected).")
    else:
        print("Warning: Embeddings for different texts are the same!")

    if embedding_empty == [0.0] * 384:
        print("Embedding for empty text is a zero vector (as expected).")
    else:
        print("Warning: Embedding for empty text is not a zero vector!")

```

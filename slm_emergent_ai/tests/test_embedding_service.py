import unittest
import numpy as np
from slm_emergent_ai.services.embedding_service import EmbeddingService

class TestEmbeddingService(unittest.TestCase):

    def setUp(self):
        self.service = EmbeddingService()
        self.dimension = 384  # As defined in EmbeddingService

    def test_generate_embedding_output_type_and_dimension(self):
        """Test that generate_embedding returns a list of floats of the correct dimension."""
        text = "This is a sample sentence."
        embedding = self.service.generate_embedding(text)

        self.assertIsInstance(embedding, list, "Embedding should be a list.")
        self.assertEqual(len(embedding), self.dimension, f"Embedding dimension should be {self.dimension}.")
        for val in embedding:
            self.assertIsInstance(val, float, "Each value in embedding should be a float.")

    def test_generate_embedding_normalization(self):
        """Test that the generated embedding is L2 normalized (or close to it)."""
        text = "A sentence to test L2 normalization."
        embedding = self.service.generate_embedding(text)

        # Calculate L2 norm
        norm = np.linalg.norm(np.array(embedding))

        # Check if norm is approximately 1.0 (for non-zero vectors) or 0.0 (for zero vectors)
        is_zero_vector = all(v == 0.0 for v in embedding)
        if not is_zero_vector:
            self.assertAlmostEqual(norm, 1.0, places=5, msg="Embedding should be L2 normalized (norm approx 1.0).")
        else:
            self.assertAlmostEqual(norm, 0.0, places=5, msg="Norm of a zero vector should be 0.0.")

    def test_generate_embedding_empty_string(self):
        """Test that an empty string returns a zero vector of the correct dimension."""
        text = ""
        embedding = self.service.generate_embedding(text)

        self.assertEqual(len(embedding), self.dimension, f"Empty string embedding dimension should be {self.dimension}.")
        self.assertTrue(all(v == 0.0 for v in embedding), "Empty string embedding should be a zero vector.")

        # Also test normalization for zero vector explicitly
        norm = np.linalg.norm(np.array(embedding))
        self.assertAlmostEqual(norm, 0.0, places=5, msg="Norm of empty string embedding should be 0.0.")

    def test_generate_embedding_none_input(self):
        """Test that None input returns a zero vector."""
        embedding = self.service.generate_embedding(None)
        self.assertEqual(len(embedding), self.dimension)
        self.assertTrue(all(v == 0.0 for v in embedding))

    def test_generate_embedding_content_variance(self):
        """Test that different texts produce different embeddings."""
        text1 = "This is the first sentence."
        embedding1 = self.service.generate_embedding(text1)

        text2 = "This is the second sentence, quite different."
        embedding2 = self.service.generate_embedding(text2)

        # Check that non-zero embeddings are different.
        # It's possible for very simple TF-IDF on short texts with shared vocab to be similar,
        # but with enough difference they should diverge.
        is_zero_vector1 = all(v == 0.0 for v in embedding1)
        is_zero_vector2 = all(v == 0.0 for v in embedding2)

        if not is_zero_vector1 and not is_zero_vector2:
            self.assertNotEqual(embedding1, embedding2, "Embeddings for different texts should be different.")
        elif is_zero_vector1 and not is_zero_vector2:
            self.assertNotEqual(embedding1, embedding2, "Embeddings for different texts should be different (one zero, one non-zero).")
        elif not is_zero_vector1 and is_zero_vector2:
            self.assertNotEqual(embedding1, embedding2, "Embeddings for different texts should be different (one non-zero, one zero).")
        # If both are zero vectors (e.g. from stop-word only text), they would be equal.

    def test_generate_embedding_special_characters_and_case(self):
        """Test text normalization (lowercase, punctuation removal)."""
        text_v1 = "Hello World! 123."
        embedding_v1 = self.service.generate_embedding(text_v1)

        text_v2 = "hello world 123" # Should be same as v1 after normalization
        embedding_v2 = self.service.generate_embedding(text_v2)

        text_v3 = "HELLO WORLD 123" # Should also be same
        embedding_v3 = self.service.generate_embedding(text_v3)

        # Due to the simplicity of TF-IDF and potential sorting of limited unique words,
        # these might become identical.
        self.assertEqual(embedding_v1, embedding_v2, "Embeddings should be the same after normalization of minor punctuation.")
        self.assertEqual(embedding_v1, embedding_v3, "Embeddings should be the same after case normalization.")

    def test_generate_embedding_long_text(self):
        """Test with text that has more unique words than the embedding dimension."""
        # Create a text with more than self.dimension unique words
        long_text = " ".join([f"word{i}" for i in range(self.dimension + 50)])
        embedding = self.service.generate_embedding(long_text)

        self.assertEqual(len(embedding), self.dimension, f"Long text embedding dimension should be {self.dimension}.")
        # Check that the vector is not all zeros, assuming "wordX" are not filtered out
        self.assertFalse(all(v == 0.0 for v in embedding), "Long text embedding should not be a zero vector.")
        norm = np.linalg.norm(np.array(embedding))
        self.assertAlmostEqual(norm, 1.0, places=5, msg="Long text embedding should be L2 normalized.")

    def test_generate_embedding_all_stopwords_like(self):
        """Test with text that might be entirely filtered if stopwords were aggressively removed.
           Current implementation doesn't use a stopword list, so this tests frequency counting."""
        text = "the a of is to in and" # Common words, but no specific stopword list in current service
        embedding = self.service.generate_embedding(text)
        self.assertEqual(len(embedding), self.dimension)
        # This should produce a non-zero vector if these words are processed
        # and at least one fits into the dimension slots.
        if len(text.split()) > 0 and len(text.split()) <= self.dimension :
             self.assertFalse(all(v == 0.0 for v in embedding), "Stopword-like text embedding should not be zero if words are present.")
             norm = np.linalg.norm(np.array(embedding))
             self.assertAlmostEqual(norm, 1.0, places=5, msg="Stopword-like text embedding should be L2 normalized.")
        elif len(text.split()) == 0 : # if text becomes empty after processing
             self.assertTrue(all(v == 0.0 for v in embedding), "Text that becomes empty should result in zero vector.")


if __name__ == '__main__':
    unittest.main()
```

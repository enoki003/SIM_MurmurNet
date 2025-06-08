import unittest
from unittest.mock import AsyncMock, MagicMock
import asyncio # Required for running async methods in tests

from slm_emergent_ai.agents.boids_enhancer import BoidsPromptEnhancer
# Assuming BlackBoard is importable for type hinting, but we'll mock it.
# from slm_emergent_ai.memory.blackboard import BlackBoard

class TestBoidsPromptEnhancer(unittest.TestCase):

    def setUp(self):
        self.mock_blackboard = MagicMock() # Use MagicMock for general bb, AsyncMock for its methods
        self.mock_blackboard.pull_messages_raw = AsyncMock()

        self.enhancer_config = {
            "neighbor_count": 3,
            "max_summary_tokens": 30 # Keep summaries short for easier testing
        }
        self.enhancer = BoidsPromptEnhancer(
            blackboard=self.mock_blackboard,
            config=self.enhancer_config
        )
        self.base_prompt = "What is the meaning of life?"
        self.agent_role = "Philosopher"
        self.agent_id = 1

    def _run_async(self, coro):
        """Helper to run async methods in synchronous tests."""
        return asyncio.run(coro)

    def test_enhance_prompt_no_neighbors(self):
        """Test enhancement when there are no neighbor messages."""
        self.mock_blackboard.pull_messages_raw.return_value = []
        agent_recent_messages = []

        enhanced_prompt = self._run_async(
            self.enhancer.enhance_prompt(self.base_prompt, self.agent_role, self.agent_id, agent_recent_messages)
        )

        self.assertIn("There are no recent messages from others to align with.", enhanced_prompt)
        self.assertIn("No specific group activity to cohere with.", enhanced_prompt)
        self.assertIn("Consider starting a new line of discussion.", enhanced_prompt)
        self.assertTrue(enhanced_prompt.endswith(self.base_prompt))

    def test_enhance_prompt_with_neighbors_alignment_cohesion(self):
        """Test enhancement with neighbor messages, focusing on alignment and cohesion."""
        self.mock_blackboard.pull_messages_raw.return_value = [
            {"id": "msg1", "agent_id": 2, "text": "The sky is blue today and beautiful.", "timestamp": 100},
            {"id": "msg2", "agent_id": 3, "text": "I agree, the blue sky is calming.", "timestamp": 101},
        ]
        agent_recent_messages = []

        enhanced_prompt = self._run_async(
            self.enhancer.enhance_prompt(self.base_prompt, self.agent_role, self.agent_id, agent_recent_messages)
        )

        # Check for keywords from neighbor messages in hints
        self.assertIn("focus on: sky, blue", enhanced_prompt) # Example, depends on _summarize_texts
        self.assertIn("converging on themes around: sky, blue", enhanced_prompt)
        self.assertNotIn("raw message dump: The sky is blue today and beautiful.", enhanced_prompt)
        self.assertTrue(enhanced_prompt.endswith(self.base_prompt))

    def test_enhance_prompt_filters_own_messages(self):
        """Test that agent's own messages are filtered out from neighbor analysis."""
        self.mock_blackboard.pull_messages_raw.return_value = [
            {"id": "msg_own1", "agent_id": self.agent_id, "text": "My own previous thought.", "timestamp": 99},
            {"id": "msg1", "agent_id": 2, "text": "Neighbor's comment about weather.", "timestamp": 100},
            {"id": "msg_own2", "agent_id": self.agent_id, "text": "Another of my own ideas.", "timestamp": 101, "id":"msg_own2_id"}, # Ensure this ID is used
        ]
        # Agent's recent messages passed to enhance_prompt to be excluded
        agent_recent_messages = [
             {"id": "msg_own2_id", "text": "Another of my own ideas."} # Match by ID
        ]

        # _analyze_neighbors should filter out msg_own1 and msg_own2
        # So, neighbor_messages for alignment/cohesion should only be based on msg1

        enhanced_prompt = self._run_async(
            self.enhancer.enhance_prompt(self.base_prompt, self.agent_role, self.agent_id, agent_recent_messages)
        )

        # Alignment/Cohesion should be based on "Neighbor's comment about weather"
        # Exact keywords depend on summarizer, check for presence of "weather" or "comment"
        self.assertTrue("weather" in enhanced_prompt or "comment" in enhanced_prompt)
        self.assertNotIn("own", enhanced_prompt.split(self.base_prompt)[0]) # Check enhancement part
        self.assertNotIn("previous", enhanced_prompt.split(self.base_prompt)[0])

        # Test _analyze_neighbors more directly by checking what it would return
        # This requires making _analyze_neighbors temporarily non-private or careful mocking
        # For now, testing via enhance_prompt's output implies correct filtering.

    def test_enhance_prompt_separation_hint(self):
        """Test that separation hint is generated when agent's topic is too similar to neighbors."""
        self.mock_blackboard.pull_messages_raw.return_value = [
            {"id": "msg1", "agent_id": 2, "text": "Let's discuss AI ethics.", "timestamp": 100},
            {"id": "msg2", "agent_id": 3, "text": "Yes, AI ethics is crucial.", "timestamp": 101},
        ]
        # Agent's own recent messages are also about AI ethics
        agent_recent_messages = [
            {"id": "agent_msg1", "text": "I was just thinking about AI ethics too."}
        ]

        enhanced_prompt = self._run_async(
            self.enhancer.enhance_prompt(self.base_prompt, self.agent_role, self.agent_id, agent_recent_messages)
        )

        self.assertIn("Try to offer a new perspective", enhanced_prompt) # Or similar separation phrase
        self.assertIn("your recent contributions (ai, ethics)", enhanced_prompt.lower()) # Check for summary of own messages
        self.assertIn("current group focus (ai, ethics)", enhanced_prompt.lower()) # Check for summary of neighbor messages

    def test_summarize_texts_logic(self):
        """Directly test the _summarize_texts helper method."""
        texts_to_summarize = [
            "The main topic here is about apples and oranges.",
            "We are also discussing bananas and other fruits like apples."
        ]
        # max_length is 30 as per enhancer_config for this test instance
        summary = self.enhancer._summarize_texts(texts_to_summarize, max_length=30)

        # Keywords expected: apples, oranges, bananas, fruits, topic (or similar)
        # Max 7 unique keywords, joined by ", "
        # Example: "apples, bananas, fruits, oranges, topic" (length depends on actual keywords extracted)
        # Length should be <= 30
        self.assertTrue(len(summary) <= 30)
        self.assertIn("apples", summary)
        self.assertIn("bananas", summary)
        # Check for "..." if truncated
        if len(", ".join(sorted(list(set(self.enhancer._extract_keywords(texts_to_summarize[0],3) + self.enhancer._extract_keywords(texts_to_summarize[1],3))))[:7])) > 30:
            self.assertTrue(summary.endswith("..."))

    def test_extract_keywords_logic(self):
        """Directly test the _extract_keywords helper method."""
        text = "keyword keyword another sample sample sample text for for for for keywords"
        keywords = self.enhancer._extract_keywords(text, max_keywords=3)
        # Expected: 'keyword', 'sample', 'another' or 'text' or 'keywords' (depending on tie-breaking in sort)
        # The current implementation sorts by frequency, then alphabetically for tie-break (implicitly by sorted())
        # Frequencies: keyword:2, another:1, sample:3, text:1, keywords:1, for:4
        # Expected: ['for', 'sample', 'keyword']
        self.assertEqual(keywords, ['for', 'sample', 'keyword'])

    def test_analyze_neighbors_pull_count_and_filtering(self):
        """Test _analyze_neighbors respects neighbor_count and filters correctly."""
        # Configure enhancer for this test to ask for 2 neighbors
        enhancer = BoidsPromptEnhancer(self.mock_blackboard, config={"neighbor_count": 2})

        self.mock_blackboard.pull_messages_raw.return_value = [
            {"id": "own1", "agent_id": self.agent_id, "text": "My own 1", "timestamp": 98},
            {"id": "n1", "agent_id": 2, "text": "Neighbor 1", "timestamp": 99},
            {"id": "own2_id", "agent_id": self.agent_id, "text": "My own 2", "timestamp": 100}, # This ID will be in agent_recent_message_ids
            {"id": "n2", "agent_id": 3, "text": "Neighbor 2", "timestamp": 101},
            {"id": "n3", "agent_id": 4, "text": "Neighbor 3", "timestamp": 102}, # Should not be included due to neighbor_count=2
        ]
        agent_recent_message_ids = ["own2_id"]

        neighbor_messages = self._run_async(
            enhancer._analyze_neighbors(self.agent_id, agent_recent_message_ids)
        )

        self.assertEqual(len(neighbor_messages), 2)
        self.assertEqual(neighbor_messages[0]["id"], "n2") # Newest first assumed from current _analyze_neighbors impl.
        self.assertEqual(neighbor_messages[1]["id"], "n1")
        # Ensure own messages and those in agent_recent_message_ids are not present
        for msg in neighbor_messages:
            self.assertNotEqual(msg["agent_id"], self.agent_id)
            self.assertNotIn(msg["id"], agent_recent_message_ids)


if __name__ == '__main__':
    unittest.main()
```

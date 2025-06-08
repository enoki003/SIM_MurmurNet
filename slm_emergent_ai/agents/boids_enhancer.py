from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from slm_emergent_ai.memory.blackboard import BlackBoard


class BoidsPromptEnhancer:
    """
    Enhances prompts using Boids-inspired rules (alignment, cohesion, separation)
    derived from recent messages on a Blackboard.
    """

    def __init__(
        self,
        blackboard: BlackBoard,
        config: Optional[Dict[str, Any]] = None,
        llm_model: Optional[Any] = None, # For potential summarization
        tokenizer: Optional[Any] = None, # For token counting / truncation
    ):
        """
        Initializes the BoidsPromptEnhancer.

        Args:
            blackboard: The Blackboard instance to fetch messages from.
            config: Configuration dictionary. Expected keys:
                "neighbor_count": Number of neighbor messages to consider (default: 5).
                "max_summary_tokens": Max tokens for generated summary hints (default: 50).
                "use_llm_for_summarization": Boolean, whether to use LLM for summarizing (default: False).
        """
        self.bb = blackboard
        self.config = config if config is not None else {}
        self.neighbor_count = self.config.get("neighbor_count", 5)
        self.max_summary_tokens = self.config.get("max_summary_tokens", 50)
        # self.use_llm_for_summarization = self.config.get("use_llm_for_summarization", False) # Future use
        # self.llm_model = llm_model # Future use
        # self.tokenizer = tokenizer # Future use

        if not hasattr(self.bb, "pull_messages_raw"):
            raise ValueError(
                "Blackboard instance must have a 'pull_messages_raw' method."
            )

    async def _analyze_neighbors(
        self, agent_id: int, agent_recent_message_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes recent messages from neighbors.

        Args:
            agent_id: The ID of the current agent, to filter out its own messages.
            agent_recent_message_ids: A list of message IDs from the current agent's recent history.

        Returns:
            A list of messages from other agents (neighbors).
        """
        # Fetch slightly more messages to account for filtering
        #pull_count = self.neighbor_count + 5 # Add a buffer for agent's own messages
        #all_recent_messages = await self.bb.pull_messages_raw(k=pull_count)

        # Fetch all messages for now, then filter. This is simpler than predicting how many to pull.
        # In a very high-traffic system, pull_messages_raw might need optimization or pagination.
        all_messages = await self.bb.pull_messages_raw(k=-1) # Get all, sort by time if not already

        neighbor_messages = []
        seen_message_ids = set(agent_recent_message_ids)

        # Assuming messages are ordered newest first from pull_messages_raw
        for msg in all_messages:
            if len(neighbor_messages) >= self.neighbor_count:
                break
            if isinstance(msg, dict) and msg.get("agent_id") != agent_id:
                msg_id = msg.get("id", str(msg.get("timestamp", ""))) # Ensure unique ID
                if msg_id not in seen_message_ids:
                    neighbor_messages.append(msg)
                    seen_message_ids.add(msg_id)

        return neighbor_messages[:self.neighbor_count]


    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Simple keyword extraction (e.g., frequent words, excluding stop words - basic version)."""
        words = [word.lower() for word in text.split() if len(word) > 3] # Basic filter
        if not words:
            return []
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:max_keywords]]

    def _summarize_texts(self, texts: List[str], max_length: int = 20) -> str:
        """Generates a very simple summary by concatenating keywords or initial phrases."""
        if not texts:
            return "nothing specific."

        all_keywords = []
        for text in texts:
            all_keywords.extend(self._extract_keywords(text, 3))

        if not all_keywords:
            # Fallback: use initial snippets
            content = " ".join([txt[:30] + "..." for txt in texts[:2]])
        else:
            # Use unique keywords
            unique_keywords = sorted(list(set(all_keywords)))
            content = ", ".join(unique_keywords[:7]) # Limit number of keywords in summary

        if len(content) > max_length:
            content = content[:max_length-3] + "..."
        return content if content else "a variety of topics."


    async def _calculate_alignment(self, neighbor_messages: List[Dict[str, Any]]) -> str:
        """
        Calculates alignment hint based on neighbor messages.
        Focuses on summarizing the general topic or intent.
        """
        if not neighbor_messages:
            return "There are no recent messages from others to align with."

        texts = [msg.get("text", "") for msg in neighbor_messages if msg.get("text")]
        summary = self._summarize_texts(texts, self.max_summary_tokens)
        return f"Recent discussion seems to focus on: {summary}."

    async def _calculate_cohesion(self, neighbor_messages: List[Dict[str, Any]]) -> str:
        """
        Calculates cohesion hint.
        Aims to identify the "center of mass" or main themes in neighbor messages.
        """
        if not neighbor_messages:
            return "No specific group activity to cohere with."

        # For now, cohesion is similar to alignment in terms of summarization
        # In a more advanced system, cohesion might look for agreement/disagreement patterns
        # or emotional tone.
        texts = [msg.get("text", "") for msg in neighbor_messages if msg.get("text")]
        summary = self._summarize_texts(texts, self.max_summary_tokens)
        return f"The group seems to be converging on themes around: {summary}."


    async def _calculate_separation(
        self,
        agent_id: int, # Added agent_id
        own_recent_messages: List[Dict[str, Any]],
        neighbor_messages: List[Dict[str, Any]],
    ) -> str:
        """
        Calculates separation hint.
        Suggests deviation if the agent's recent messages are too similar to neighbors
        or if the general discussion is becoming too monolithic.
        """
        if not neighbor_messages and not own_recent_messages:
            return "Consider starting a new line of discussion."

        own_texts = [msg.get("text", "") for msg in own_recent_messages if msg.get("text")]
        neighbor_texts = [msg.get("text", "") for msg in neighbor_messages if msg.get("text")]

        own_summary = self._summarize_texts(own_texts, 20) # Short summary of own recent topics
        neighbor_summary = self._summarize_texts(neighbor_texts, 20) # Short summary of neighbor topics

        # Simple heuristic: if own summary and neighbor summary are very similar, suggest deviation.
        # This is a placeholder for a more sophisticated similarity check.
        if own_summary.strip('.').split(', ')[0] == neighbor_summary.strip('.').split(', ')[0] and own_summary != "nothing specific.":
             return f"Your recent contributions ({own_summary}) are very similar to the current group focus ({neighbor_summary}). Try to offer a new perspective or ask a clarifying question on a different aspect."

        # Another heuristic: if there are many messages and high similarity, suggest separation.
        # This requires a proper similarity metric. For now, we'll keep it simple.
        # if len(neighbor_messages) > self.neighbor_count / 2:
        #    # Advanced: check for content repetitiveness in neighbor_messages
        #    pass

        return "Consider if your next contribution should offer a contrasting viewpoint or explore a new angle."


    async def enhance_prompt(
        self, base_prompt: str, agent_role: str, agent_id: int, agent_recent_messages: List[Dict[str, Any]]
    ) -> str:
        """
        Enhances the base prompt with Boids-inspired guidance.

        Args:
            base_prompt: The original prompt for the agent.
            agent_role: The role of the agent.
            agent_id: The ID of the agent.
            agent_recent_messages: A list of the agent's own recent messages (dictionaries with 'text' and 'id').

        Returns:
            The enhanced prompt string.
        """
        agent_recent_message_ids = [msg.get("id", str(msg.get("timestamp"))) for msg in agent_recent_messages]

        # Analyze neighbors (async call)
        neighbor_messages = await self._analyze_neighbors(agent_id, agent_recent_message_ids)

        # Calculate Boids components (can be async if they do internal async calls)
        alignment_hint = await self._calculate_alignment(neighbor_messages)
        cohesion_hint = await self._calculate_cohesion(neighbor_messages)
        separation_hint = await self._calculate_separation(agent_id, agent_recent_messages, neighbor_messages)

        # Construct the enhanced prompt
        enhancements = [
            f"As a {agent_role}, consider the following social dynamics:",
            f"- Alignment: {alignment_hint}",
            f"- Cohesion: {cohesion_hint}",
            f"- Separation: {separation_hint}",
            "Based on this, refine your response to the following task:"
        ]

        enhanced_prompt = "\n".join(enhancements) + "\n\n" + base_prompt

        # Optional: Log the enhancement for debugging
        # print(f"Agent {agent_id} Enhancements: A: {alignment_hint} C: {cohesion_hint} S: {separation_hint}")

        return enhanced_prompt

# Example Usage (Illustrative - requires an async environment and a mock Blackboard)
async def main_example():
    class MockBlackboard:
        async def pull_messages_raw(self, k: int = -1) -> List[Dict[str, Any]]:
            # Simulate fetching messages
            return [
                {"id": "msg1", "agent_id": 2, "text": "I think AI safety is very important.", "timestamp": 100},
                {"id": "msg2", "agent_id": 3, "text": "Agree, AI safety protocols should be standardized.", "timestamp": 101},
                {"id": "msg3", "agent_id": 2, "text": "We should also consider the economic impact of AI.", "timestamp": 102},
                {"id": "msg4", "agent_id": 1, "text": "My previous point was about economic impact.", "timestamp": 103}, # Agent 1's own message
                {"id": "msg5", "agent_id": 4, "text": "Let's not forget ethical AI development.", "timestamp": 104},
            ]

    bb = MockBlackboard()
    enhancer_config = {"neighbor_count": 3, "max_summary_tokens": 30}
    enhancer = BoidsPromptEnhancer(blackboard=bb, config=enhancer_config)

    agent1_recent_msgs = [{"id": "msg4", "text": "My previous point was about economic impact.", "timestamp": 103}]

    enhanced_prompt = await enhancer.enhance_prompt(
        base_prompt="What are your thoughts on the future of AI?",
        agent_role="Critical Thinker",
        agent_id=1,
        agent_recent_messages=agent1_recent_msgs,
    )
    print(enhanced_prompt)

if __name__ == "__main__":
    # This example won't run directly without an event loop manager like asyncio.run()
    # asyncio.run(main_example())
    print("BoidsPromptEnhancer class defined. Run main_example() in an async context to test.")

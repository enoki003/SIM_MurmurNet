"""
Agent History Manager - Handles processing and formatting of conversation history for SLMAgents.
"""

from typing import List, Dict, Optional, Any

# Use try-except for BlackBoard and ConversationSummarizer imports for flexibility
try:
    from ..memory.blackboard import BlackBoard
    from ..summarizer.conversation_summarizer import ConversationSummarizer
except ImportError:
    # This allows the file to be potentially analyzed or imported in contexts where
    # the relative imports might not immediately resolve (e.g. some linters or tools
    # not running from the project root).
    # For runtime, the imports from the try block should work.
    BlackBoard = None
    ConversationSummarizer = None


class AgentHistoryManager:
    """
    Manages the retrieval, processing (including summarization),
    and formatting of agent conversation history.
    """

    def __init__(self, blackboard: "BlackBoard", summarizer: Optional["ConversationSummarizer"]):
        """
        Initializes the AgentHistoryManager.

        Args:
            blackboard (BlackBoard): The shared blackboard instance for message retrieval.
            summarizer (Optional[ConversationSummarizer]): Summarizer for conversation history.
        """
        if BlackBoard is None or ConversationSummarizer is None:
            # This check helps if the fallback imports were used, indicating a potential setup issue.
            # Depending on strictness, could raise an error or just warn.
            print("[WARN] AgentHistoryManager: BlackBoard or ConversationSummarizer type not available at init. Ensure imports are correct.")

        self.blackboard = blackboard
        self.summarizer = summarizer

    async def get_processed_history(
        self, history_fetch_limit: int = 20, num_recent_verbatim: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves raw messages, applies summarization if configured and applicable,
        and returns a list of message dictionaries for prompt construction.

        Args:
            history_fetch_limit (int): How many messages to pull for potential summarization.
            num_recent_verbatim (int): How many of the very latest messages to keep raw.

        Returns:
            List[Dict[str, Any]]: A list of message dictionaries, potentially including a summary.
        """
        if not self.blackboard:
            # Should not happen if constructor check is in place and enforced
            raise ValueError("AgentHistoryManager: Blackboard not initialized.")

        raw_messages_for_history = await self.blackboard.pull_messages_raw(k=history_fetch_limit)
        processed_history: List[Dict[str, Any]] = []
        summary_text = ""

        if (
            self.summarizer
            and len(raw_messages_for_history) > self.summarizer.summarize_threshold
        ):
            actual_num_recent_verbatim = min(
                num_recent_verbatim, len(raw_messages_for_history)
            )

            messages_to_summarize = (
                raw_messages_for_history[:-actual_num_recent_verbatim]
                if actual_num_recent_verbatim > 0
                else raw_messages_for_history
            )
            recent_verbatim_messages = (
                raw_messages_for_history[-actual_num_recent_verbatim:]
                if actual_num_recent_verbatim > 0
                else []
            )

            if messages_to_summarize:
                summary_text = self.summarizer.summarize(messages_to_summarize)

            if summary_text:
                processed_history.append(
                    {"agent_name": "System", "text": summary_text, "type": "summary"}
                )
            processed_history.extend(recent_verbatim_messages)
        else:
            actual_num_recent_verbatim = min(
                num_recent_verbatim, len(raw_messages_for_history)
            )
            processed_history = raw_messages_for_history[-actual_num_recent_verbatim:]

        return processed_history

    def format_history_for_prompt(
        self, processed_history: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Formats the processed history (list of message dictionaries) into a list of strings
        suitable for inclusion in an LLM prompt.

        Args:
            processed_history (List[Dict[str, Any]]): The list of message dictionaries.

        Returns:
            List[str]: A list of formatted strings for the prompt.
        """
        formatted_history_strings: List[str] = []
        if processed_history:
            for msg in processed_history:
                if isinstance(msg, dict):
                    sender_name = msg.get("agent_name", "Agent")
                    text_content = msg.get("text", "")

                    if msg.get("type") == "summary":
                        formatted_history_strings.append(text_content)
                    else:
                        if len(text_content) > 150: # Truncation limit
                            text_content = text_content[:150] + "..."
                        formatted_history_strings.append(f"{sender_name}: {text_content}")
                else:
                    # Fallback for unexpected message format
                    formatted_history_strings.append(str(msg)[:150])
        return formatted_history_strings

# slm_emergent_ai/summarizer/conversation_summarizer.py

from typing import List, Dict, Any

class ConversationSummarizer:
    def __init__(self, summarize_threshold: int = 10, num_first_messages: int = 2, num_last_messages: int = 3):
        """
        Initializes the ConversationSummarizer.

        Args:
            summarize_threshold (int): Minimum number of messages required to attempt summarization.
            num_first_messages (int): Number of initial messages to include in the digest.
            num_last_messages (int): Number of final messages to include in the digest.
        """
        self.summarize_threshold = summarize_threshold
        self.num_first_messages = num_first_messages
        self.num_last_messages = num_last_messages

    def summarize(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Creates a digest of the conversation history by selecting first and last messages.

        Args:
            conversation_history (List[Dict[str, Any]]): A list of messages.
                                                       Each message is a dict, expected to have 'agent_name' and 'text'.

        Returns:
            str: A summary string, or an empty string if no summary is generated.
        """
        num_messages = len(conversation_history)

        if num_messages < self.summarize_threshold:
            # Not enough messages to summarize, or return a specific format indicating no summary
            return "" # Or perhaps: "No summary generated (conversation too short)."

        summary_parts = []

        # Add a header for the summary
        summary_parts.append("Summary of earlier discussion:")

        # Select first messages
        if self.num_first_messages > 0:
            summary_parts.append("\n--- Start of Discussion Highlights ---")
            for i in range(min(self.num_first_messages, num_messages)):
                msg = conversation_history[i]
                summary_parts.append(f"{msg.get('agent_name', 'Agent')}: {msg.get('text', '')}")

        # Add a separator if both first and last messages are included
        if self.num_first_messages > 0 and self.num_last_messages > 0 and num_messages > self.num_first_messages + self.num_last_messages:
            summary_parts.append("\n--- End of Discussion Highlights ---")
            # Adding a visual separator for clarity when older messages are significantly ellipsed.
            if num_messages > self.num_first_messages + self.num_last_messages + 1: # an arbitrary number to decide if ellipsis is needed
                summary_parts.append("[...discussion continued...]")


        # Select last messages (ensure no overlap with first messages if history is short)
        # Start index for last messages should not be before the end of first messages
        start_index_for_last = max(self.num_first_messages, num_messages - self.num_last_messages)

        if self.num_last_messages > 0:
            # Only add this header if we haven't effectively just shown all messages via num_first + num_last
            if num_messages > self.num_first_messages + self.num_last_messages :
                 summary_parts.append("\n--- Most Recent Discussion Points ---")

            for i in range(start_index_for_last, num_messages):
                msg = conversation_history[i]
                summary_parts.append(f"{msg.get('agent_name', 'Agent')}: {msg.get('text', '')}")

        if not summary_parts or len(summary_parts) == 1 and summary_parts[0] == "Summary of earlier discussion:":
             return "" # No actual content was added

        return "\n".join(summary_parts)

if __name__ == '__main__':
    summarizer = ConversationSummarizer(summarize_threshold=5, num_first_messages=2, num_last_messages=2)

    sample_history_short = [
        {'agent_name': 'Alice', 'text': 'Hello everyone!'},
        {'agent_name': 'Bob', 'text': 'Hi Alice, how are you?'},
        {'agent_name': 'Charlie', 'text': 'Good morning!'}
    ]
    print(f"Short History (len={len(sample_history_short)}):")
    print(f"Summary: '{summarizer.summarize(sample_history_short)}'\n") # Expect: ""

    sample_history_medium = [
        {'agent_name': 'Alice', 'text': 'Initial idea about project A.'},
        {'agent_name': 'Bob', 'text': 'Response to project A.'},
        {'agent_name': 'Charlie', 'text': 'Question about project A.'},
        {'agent_name': 'David', 'text': 'Clarification for project A.'},
        {'agent_name': 'Eve', 'text': 'Final thoughts on project A for now.'}
    ]
    print(f"Medium History (len={len(sample_history_medium)}):")
    summary_medium = summarizer.summarize(sample_history_medium)
    print(f"Summary:\n{summary_medium}\n")
    # Expect: Start, Eve's message, possibly David's too if num_last_messages=2 and no overlap logic is perfect

    sample_history_long = [
        {'agent_name': 'Agent 1', 'text': 'This is the first message.'},
        {'agent_name': 'Agent 2', 'text': 'Second message, building on first.'},
        {'agent_name': 'Agent 3', 'text': 'A middle message that might be skipped.'},
        {'agent_name': 'Agent 4', 'text': 'Another middle message.'},
        {'agent_name': 'Agent 5', 'text': 'Yet another middle one.'},
        {'agent_name': 'Agent 6', 'text': 'Penultimate message.'},
        {'agent_name': 'Agent 7', 'text': 'The very last message.'}
    ]
    print(f"Long History (len={len(sample_history_long)}):")
    summary_long = summarizer.summarize(sample_history_long)
    print(f"Summary:\n{summary_long}\n")
    # Expect: Agent 1, Agent 2, ..., Agent 6, Agent 7

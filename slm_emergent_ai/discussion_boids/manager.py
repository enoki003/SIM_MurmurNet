# slm_emergent_ai/discussion_boids/manager.py

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, util

class DiscussionBoidsManager:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.9,
                 diversity_threshold: float = 0.7,
                 alignment_threshold: float = 0.8,
                 cohesion_threshold: float = 0.6):
        """
        Manages Boids-inspired rules for discussion content.

        Args:
            model_name (str): Name of the SentenceTransformer model to use for embeddings.
            similarity_threshold (float): General similarity threshold.
            diversity_threshold (float): If avg similarity among others is *above* this, encourage separation.
            alignment_threshold (float): If similarity between two messages from different other agents is *above* this, encourage alignment.
            cohesion_threshold (float): If avg similarity of discussion to topic is *below* this, encourage refocusing.
        """
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            print("Please ensure 'sentence-transformers' is installed and the model is valid.")
            self.embedding_model = None

        self.similarity_threshold = similarity_threshold
        self.diversity_threshold = diversity_threshold
        self.alignment_threshold = alignment_threshold
        self.cohesion_threshold = cohesion_threshold
        self.topic_embedding: Optional[np.ndarray] = None
        # self.message_history_embeddings: List[Dict[str, Any]] = [] # No longer storing internal history

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Helper function to get sentence embedding."""
        if not self.embedding_model:
            return None
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding text to embedding: {e}")
            return None

    def set_topic_embedding(self, topic_text: str):
        """
        Sets the topic embedding for the Cohesion rule.

        Args:
            topic_text (str): The text representing the main topic/task.
        """
        if not self.embedding_model:
            print("INFO: Embedding model not available. Cannot set topic embedding.")
            return

        embedding = self._get_embedding(topic_text)
        if embedding is not None:
            self.topic_embedding = embedding
            print(f"Topic embedding set successfully for: '{topic_text[:100]}...'")
        else:
            print(f"Failed to set topic embedding for: '{topic_text[:100]}...'")

    def get_directive_for_agent(self, agent_id_to_prompt: str, recent_messages_history: List[Dict[str, Any]]) -> str:
        """
        Returns a Boids-inspired directive based on the recent messages history,
        focusing on the diversity of messages from *other* agents.

        Args:
            agent_id_to_prompt (str): The ID of the agent for whom the directive is being generated.
                                      Messages from this agent will be excluded from the diversity calculation.
            recent_messages_history (List[Dict[str, Any]]): A list of recent messages from the blackboard.
                                                           Each dict should have at least 'text' and 'agent_id'.

        Returns:
            str: A directive string or an empty string if no specific directive.
        """
        if not self.embedding_model:
            return "INFO: Embedding model not available, Boids directives disabled."

        directives = []
        cohesion_triggered = False

        # Step 1: Collect texts from other agents and their original info
        other_agents_texts = []
        original_message_infos = [] # To store original text and agent_id for later reconstruction

        for msg in recent_messages_history:
            if msg.get('agent_id') != agent_id_to_prompt and msg.get('text'):
                other_agents_texts.append(msg.get('text'))
                original_message_infos.append({'text': msg.get('text'), 'agent_id': msg.get('agent_id')})

        if not other_agents_texts: # No messages from other agents
            return ""

        # Step 2: Batch encode all collected texts
        try:
            all_other_embeddings_list = self.embedding_model.encode(other_agents_texts, convert_to_numpy=True)
        except Exception as e:
            print(f"Error batch encoding texts: {e}")
            return "INFO: Error processing message embeddings for Boids directives."

        # Step 3: Reconstruct the list of messages with their embeddings
        other_agents_messages_with_embeddings = []
        for i, info in enumerate(original_message_infos):
            other_agents_messages_with_embeddings.append({
                'text': info['text'],
                'agent_id': info['agent_id'],
                'embedding': all_other_embeddings_list[i]
            })

        # --- Cohesion Rule ---
        if self.topic_embedding is not None and len(other_agents_messages_with_embeddings) > 0:
            # Embeddings are already numpy arrays from batch encoding
            current_discussion_avg_embedding = np.mean(all_other_embeddings_list, axis=0)

            similarity_to_topic = util.cos_sim(current_discussion_avg_embedding, self.topic_embedding)[0][0]

            if similarity_to_topic < self.cohesion_threshold:
                directive_cohesion = (
                    "The current discussion seems to be moving away from the main objective. "
                    "Let's try to bring the focus back."
                )
                directives.append(directive_cohesion)
                cohesion_triggered = True

        if not cohesion_triggered:
            if len(other_agents_messages_with_embeddings) < 2:
                return " ".join(directives)

            # --- Alignment Rule ---
            # Uses the full similarity matrix for efficiency
            alignment_found = False
            # all_other_embeddings_list is already an np.array of embeddings
            similarity_matrix_alignment = util.cos_sim(all_other_embeddings_list, all_other_embeddings_list)

            num_other_msgs = len(other_agents_messages_with_embeddings)
            for i in range(num_other_msgs):
                if alignment_found: break
                for j in range(i + 1, num_other_msgs):
                    msg1_info = other_agents_messages_with_embeddings[i]
                    msg2_info = other_agents_messages_with_embeddings[j]

                    if msg1_info['agent_id'] == msg2_info['agent_id']:
                        continue # Skip messages from the same agent

                    # Similarity already computed in matrix
                    similarity = similarity_matrix_alignment[i, j]

                    if similarity > self.alignment_threshold:
                        aligned_text_snippet = msg1_info['text'][:50]
                        directive_alignment = (
                            f"There seems to be some agreement forming around: '{aligned_text_snippet}...'. "
                            f"Perhaps you could elaborate on this or provide further support?"
                        )
                        directives.append(directive_alignment)
                        alignment_found = True
                        break

            # --- Separation (Diversity) Rule ---
            if not alignment_found:
                # The similarity_matrix_alignment is the same one needed for separation among other agents
                # No need to recalculate util.cos_sim for the same set of embeddings

                pairwise_similarities = []
                # Iterate over upper triangle of similarity_matrix_alignment
                for i in range(num_other_msgs):
                    for j in range(i + 1, num_other_msgs):
                        pairwise_similarities.append(similarity_matrix_alignment[i, j])

                if pairwise_similarities:
                    average_similarity_among_others = np.mean(pairwise_similarities)
                    if average_similarity_among_others > self.diversity_threshold:
                        directive_separation = (
                            "The recent discussion appears to lack diverse viewpoints. "
                            "Please offer a fresh perspective or a contrasting idea."
                        )
                        directives.append(directive_separation)

        return " ".join(directives)

if __name__ == '__main__':
    # Example Usage (for testing)
    manager = DiscussionBoidsManager(
        diversity_threshold=0.6,
        alignment_threshold=0.82, # Made slightly higher to differentiate from cohesion
        cohesion_threshold=0.5  # Threshold for cohesion
    )
    if manager.embedding_model:
        print("DiscussionBoidsManager initialized with embedding model.")

        topic = "Sustainable urban development and green technologies."
        manager.set_topic_embedding(topic)
        print(f"Topic for testing: '{topic}'")

        # Scenario 1: Cohesion - discussion is off-topic
        history_off_topic = [
            {"text": "Let's talk about favorite holiday destinations.", "agent_id": "Agent_1"},
            {"text": "I really enjoyed my trip to the mountains last year.", "agent_id": "Agent_2"},
            {"text": "Beaches are better, in my opinion. So relaxing.", "agent_id": "Agent_3"}
        ]
        directive_sc1 = manager.get_directive_for_agent("Agent_4", history_off_topic)
        print(f"\nScenario 1 (Off-topic discussion): Expect Cohesion prompt")
        print(f"Directive for Agent_4: '{directive_sc1}'") # Expected: Cohesion

        # Scenario 2: On-topic and Aligned (Expect Alignment)
        history_on_topic_aligned = [
            {"text": "Solar panel adoption is key for sustainable cities.", "agent_id": "Agent_1"}, # On topic
            {"text": "I agree, solar panels are crucial for urban sustainability.", "agent_id": "Agent_2"}, # On topic & Aligned with A1
            {"text": "We also need better public transport for green cities.", "agent_id": "Agent_3"}  # On topic but different
        ]
        directive_sc2 = manager.get_directive_for_agent("Agent_4", history_on_topic_aligned)
        print(f"\nScenario 2 (On-topic, strong alignment): Expect Alignment prompt")
        print(f"Directive for Agent_4: '{directive_sc2}'")

        # Scenario 3: On-topic and Similar (but not aligned enough for Alignment rule, triggers Separation)
        manager_sep_test = DiscussionBoidsManager(diversity_threshold=0.6, alignment_threshold=0.95, cohesion_threshold=0.5)
        manager_sep_test.set_topic_embedding(topic)
        history_on_topic_similar = [
            {"text": "Green roofs can significantly improve urban ecosystems.", "agent_id": "Agent_1"}, # On topic
            {"text": "Yes, installing green roofs is a great idea for city biodiversity.", "agent_id": "Agent_2"}, # On topic & similar to A1
            {"text": "Vertical gardens also contribute to greener urban spaces.", "agent_id": "Agent_3"}  # On topic & similar concept
        ]
        # Assume embeddings for these are similar enough to trigger separation but not high enough for alignment
        directive_sc3 = manager_sep_test.get_directive_for_agent("Agent_4", history_on_topic_similar)
        print(f"\nScenario 3 (On-topic, similar but not strongly aligned): Expect Separation prompt")
        print(f"Directive for Agent_4: '{directive_sc3}'")

        # Scenario 4: On-topic and Diverse (Expect No specific directive)
        history_on_topic_diverse = [
            {"text": "Improving waste management systems is vital for urban sustainability.", "agent_id": "Agent_1"}, # On topic
            {"text": "Community gardens can also play a role in greener cities.", "agent_id": "Agent_2"}, # On topic, different aspect
            {"text": "What about policy changes to incentivize sustainable construction?", "agent_id": "Agent_3"} # On topic, different angle
        ]
        directive_sc4 = manager.get_directive_for_agent("Agent_4", history_on_topic_diverse)
        print(f"\nScenario 4 (On-topic and diverse): Expect No specific directive")
        print(f"Directive for Agent_4: '{directive_sc4}'")

        # Scenario 5: No topic set, but alignment exists (Expect Alignment)
        manager_no_topic = DiscussionBoidsManager(alignment_threshold=0.8)
        # No call to manager_no_topic.set_topic_embedding()
        directive_sc5 = manager_no_topic.get_directive_for_agent("Agent_4", history_on_topic_aligned) # Reusing aligned history
        print(f"\nScenario 5 (No topic set, alignment exists): Expect Alignment prompt")
        print(f"Directive for Agent_4: '{directive_sc5}'")

    else:
        print("Failed to initialize DiscussionBoidsManager.")

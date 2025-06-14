# slm_emergent_ai/discussion_boids/manager.py

from typing import List, Dict, Any, Optional
import numpy as np
import re
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
            recent_messages_history (List[Dict[str, Any]]): A list of recent messages from the blackboard.                                                           Each dict should have at least 'text' and 'agent_id'.

        Returns:
            str: A directive string or an empty string if no specific directive.
        """
        if not self.embedding_model:
            return "INFO: Embedding model not available, Boids directives disabled."        # Apply window-based processing for scalability
        windowed_messages = self.get_recent_messages(recent_messages_history, window_size=50)

        directives = []  # Now stores structured directive dictionaries
        cohesion_triggered = False

        # Step 1: Collect texts from other agents and their original info (with meta-text filtering)
        other_agents_texts = []
        original_message_infos = [] # To store original text and agent_id for later reconstruction

        for msg in windowed_messages:
            msg_text = msg.get('text', '')
            msg_agent_id = msg.get('agent_id', '')
            
            # Filter: exclude self, empty messages, and meta-text
            if (msg_agent_id != agent_id_to_prompt and 
                msg_text and 
                self._is_user_content(msg_text)):
                
                other_agents_texts.append(msg_text)
                original_message_infos.append({'text': msg_text, 'agent_id': msg_agent_id})

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
            })        # --- Cohesion Rule ---
        if self.topic_embedding is not None and len(other_agents_messages_with_embeddings) > 0:
            # Use weighted average favoring recent messages to avoid dilution
            num_messages = len(all_other_embeddings_list)
            if num_messages > 10:  # Use exponential decay weights for large message sets
                # Recent messages have higher weights (exponential decay)
                weights = np.exp(-0.1 * np.arange(num_messages)[::-1])
                weights = weights / np.sum(weights)  # Normalize weights
                current_discussion_avg_embedding = np.average(all_other_embeddings_list, weights=weights, axis=0)
            else:
                # Use simple average for small message sets
                current_discussion_avg_embedding = np.mean(all_other_embeddings_list, axis=0)

            similarity_to_topic = util.cos_sim(current_discussion_avg_embedding, self.topic_embedding)[0][0]

            if similarity_to_topic < self.cohesion_threshold:
                directive_cohesion = {
                    'type': 'cohesion',
                    'priority': 'high',
                    'content': "The current discussion seems to be moving away from the main objective. Let's try to bring the focus back.",
                    'confidence': 1.0 - similarity_to_topic  # Higher confidence when more off-topic
                }
                directives.append(directive_cohesion)
                cohesion_triggered = True

        if not cohesion_triggered:
            if len(other_agents_messages_with_embeddings) < 2:
                return self.build_llm_optimized_prompt(directives)

            # --- Alignment Rule ---
            # Uses the full similarity matrix for efficiency
            alignment_found = False
            # all_other_embeddings_list is already an np.array of embeddings
            similarity_matrix_alignment = util.cos_sim(all_other_embeddings_list, all_other_embeddings_list)

            num_other_msgs = len(other_agents_messages_with_embeddings)
            for i in range(num_other_msgs):
                if alignment_found: 
                    break
                for j in range(i + 1, num_other_msgs):
                    msg1_info = other_agents_messages_with_embeddings[i]
                    msg2_info = other_agents_messages_with_embeddings[j]

                    if msg1_info['agent_id'] == msg2_info['agent_id']:
                        continue # Skip messages from the same agent

                    # Similarity already computed in matrix (対角セルは自動的に除外される i != j)
                    similarity = similarity_matrix_alignment[i, j]

                    if similarity > self.alignment_threshold:
                        aligned_text_snippet = msg1_info['text'][:50]
                        directive_alignment = {
                            'type': 'alignment',
                            'priority': 'medium',
                            'content': f"There seems to be some agreement forming around: '{aligned_text_snippet}...'. Perhaps you could elaborate on this or provide further support?",
                            'confidence': similarity  # Use similarity as confidence
                        }
                        directives.append(directive_alignment)
                        alignment_found = True
                        break

            # --- Separation (Diversity) Rule ---
            if not alignment_found:
                # The similarity_matrix_alignment is the same one needed for separation among other agents
                # 対角セル除去：上三角行列のみを使用して自己類似度を除外
                pairwise_similarities = []
                # Iterate over upper triangle of similarity_matrix_alignment (対角セル除外)
                for i in range(num_other_msgs):
                    for j in range(i + 1, num_other_msgs):  # i + 1 により対角セル (i == j) を除外
                        # 同一エージェントのメッセージ間の類似度は除外
                        if other_agents_messages_with_embeddings[i]['agent_id'] != other_agents_messages_with_embeddings[j]['agent_id']:
                            pairwise_similarities.append(similarity_matrix_alignment[i, j])

                if pairwise_similarities:
                    average_similarity_among_others = np.mean(pairwise_similarities)
                    print(f"[BOIDS_DEBUG] Separation rule - Average similarity among different agents: {average_similarity_among_others:.3f}")
                    print(f"[BOIDS_DEBUG] Separation rule - Diversity threshold: {self.diversity_threshold:.3f}")
                    
                    # 判定ロジック修正：閾値より高い場合に多様性不足と判定
                    if average_similarity_among_others > self.diversity_threshold:
                        directive_separation = {
                            'type': 'separation',
                            'priority': 'medium',
                            'content': "The recent discussion appears to lack diverse viewpoints. Please offer a fresh perspective or a contrasting idea.",
                            'confidence': average_similarity_among_others  # Use average similarity as confidence
                        }
                        directives.append(directive_separation)
                        print(f"[BOIDS_DEBUG] Separation directive triggered - High similarity ({average_similarity_among_others:.3f}) > threshold ({self.diversity_threshold:.3f})")
                    else:
                        print(f"[BOIDS_DEBUG] Separation rule not triggered - Sufficient diversity ({average_similarity_among_others:.3f}) <= threshold ({self.diversity_threshold:.3f})")
                else:
                    print(f"[BOIDS_DEBUG] Separation rule - No valid pairwise similarities found")

        # Convert structured directives to string format for backward compatibility
        if not directives:
            return ""
        
        # Use LLM-optimized prompt building for better results
        return self.build_llm_optimized_prompt(directives)

    def _is_user_content(self, msg_text: str) -> bool:
        """
        Check if message text is actual user content (not meta/system text).
        
        Args:
            msg_text (str): Message text to check
            
        Returns:
            bool: True if it's user content, False if it's meta/system text
        """
        if not msg_text or not msg_text.strip():
            return False
            
        # Patterns for meta/system content to exclude
        meta_patterns = [
            r'^Boids\s+Suggestion\s*:',         # Boids suggestions
            r'^\[.*\]',                         # Log messages like [DEBUG], [INFO]
            r'^System\s*:',                     # System messages
            r'^DEBUG\s*:',                      # Debug messages
            r'^INFO\s*:',                       # Info messages
            r'^ERROR\s*:',                      # Error messages
            r'^WARNING\s*:',                    # Warning messages
            r'^METRICS\s*:',                    # Metrics messages
            r'^<!--.*-->',                      # HTML-style comments
            r'^\s*#\s*meta\s*:',               # Meta tags
            r'^\s*<meta>.*</meta>\s*$',        # XML-style meta tags
        ]
        
        # Check if text matches any meta pattern
        for pattern in meta_patterns:
            if re.match(pattern, msg_text.strip(), re.IGNORECASE):
                return False
                
        return True

    def calibrate_thresholds(self, sample_texts: List[str], percentiles: Dict[str, int] = None) -> Dict[str, float]:
        """
        Calibrate thresholds based on statistical analysis of sample texts.
        
        Args:
            sample_texts (List[str]): Sample texts for threshold calibration
            percentiles (Dict[str, int]): Custom percentiles for each threshold type
            
        Returns:
            Dict[str, float]: Calibrated thresholds
        """
        if not self.embedding_model or len(sample_texts) < 10:
            print("INFO: Insufficient data for threshold calibration, using defaults.")
            return {
                'cohesion_threshold': self.cohesion_threshold,
                'alignment_threshold': self.alignment_threshold,
                'diversity_threshold': self.diversity_threshold
            }
        
        # Default percentiles if not provided
        if percentiles is None:
            percentiles = {
                'cohesion_threshold': 40,    # 40th percentile
                'alignment_threshold': 85,   # 85th percentile  
                'diversity_threshold': 70    # 70th percentile
            }
        
        try:
            # Encode sample texts
            sample_embeddings = self.embedding_model.encode(sample_texts, convert_to_numpy=True)
            
            # Calculate similarity matrix
            similarities = util.cos_sim(sample_embeddings, sample_embeddings).numpy()
            
            # Exclude diagonal elements (self-similarity = 1.0)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            sim_values = similarities[mask]
            
            # Calculate percentile-based thresholds
            calibrated_thresholds = {}
            for threshold_name, percentile in percentiles.items():
                calibrated_thresholds[threshold_name] = float(np.percentile(sim_values, percentile))
            
            print(f"INFO: Thresholds calibrated from {len(sample_texts)} samples:")
            for name, value in calibrated_thresholds.items():
                print(f"  {name}: {value:.3f}")
                
            return calibrated_thresholds
            
        except Exception as e:
            print(f"ERROR: Threshold calibration failed: {e}")
            return {
                'cohesion_threshold': self.cohesion_threshold,
                'alignment_threshold': self.alignment_threshold,
                'diversity_threshold': self.diversity_threshold
            }

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update thresholds with new values.
        
        Args:
            new_thresholds (Dict[str, float]): New threshold values
        """
        if 'cohesion_threshold' in new_thresholds:
            self.cohesion_threshold = new_thresholds['cohesion_threshold']
        if 'alignment_threshold' in new_thresholds:
            self.alignment_threshold = new_thresholds['alignment_threshold']
        if 'diversity_threshold' in new_thresholds:
            self.diversity_threshold = new_thresholds['diversity_threshold']
            
        print(f"INFO: Thresholds updated - Cohesion: {self.cohesion_threshold:.3f}, "
              f"Alignment: {self.alignment_threshold:.3f}, Diversity: {self.diversity_threshold:.3f}")

    def get_recent_messages(self, messages: List[Dict[str, Any]], window_size: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recent N messages for processing to improve scalability.
        
        Args:
            messages (List[Dict[str, Any]]): All messages
            window_size (int): Maximum number of recent messages to consider
            
        Returns:
            List[Dict[str, Any]]: Recent messages within the window
        """
        if len(messages) <= window_size:
            return messages
        
        # Return the most recent messages
        recent_messages = messages[-window_size:]
        print(f"INFO: Using recent {len(recent_messages)} messages (window size: {window_size}) "
              f"from total {len(messages)} messages for scalability.")
        return recent_messages

    def hierarchical_similarity_check(self, embeddings: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Perform hierarchical similarity checking for large datasets.
        
        Args:
            embeddings (np.ndarray): Embedding vectors
            threshold (float): Clustering threshold
            
        Returns:
            Dict[str, Any]: Results of hierarchical analysis
        """
        num_embeddings = len(embeddings)
        
        if num_embeddings <= 20:
            # Use full similarity matrix for small datasets
            similarity_matrix = util.cos_sim(embeddings, embeddings).numpy()
            return {
                'method': 'full_matrix',
                'similarity_matrix': similarity_matrix,
                'num_comparisons': num_embeddings * (num_embeddings - 1) // 2
            }
        else:
            # Use sampling for large datasets
            sample_size = min(20, num_embeddings // 2)
            sample_indices = np.random.choice(num_embeddings, size=sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            # Calculate similarity on sampled data
            similarity_matrix = util.cos_sim(sample_embeddings, sample_embeddings).numpy()
            
            print(f"INFO: Using hierarchical similarity check with {sample_size} samples "
                  f"from {num_embeddings} embeddings for performance.")
            
            return {
                'method': 'sampled',
                'similarity_matrix': similarity_matrix,
                'sample_indices': sample_indices,
                'num_comparisons': sample_size * (sample_size - 1) // 2
            }

    def format_structured_directive(self, directives: List[Dict[str, Any]]) -> str:
        """
        Generate structured directive format for better LLM understanding.
        
        Args:
            directives (List[Dict[str, Any]]): List of directive dictionaries
            
        Returns:
            str: Formatted directive string
        """
        if not directives:
            return ""
        
        # Sort by priority (high -> medium -> low)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_directives = sorted(directives, 
                                 key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        formatted = []
        for i, directive in enumerate(sorted_directives, 1):
            rule_type = directive.get('type', 'general')
            priority = directive.get('priority', 'medium')
            content = directive.get('content', '')
            confidence = directive.get('confidence', 0.0)
            
            # 会話口調の指示に変更
            if rule_type == 'cohesion':
                conversation_content = f"話題が少し逸れているようですね。メインのテーマに沿った発言をしていただけると、みんなで建設的な議論ができそうです。"
            elif rule_type == 'alignment':
                conversation_content = f"いい感じで意見が一致してきていますね。この流れをさらに発展させて、より深い議論にしていきませんか？"
            elif rule_type == 'separation':
                conversation_content = f"みなさん似たような視点が多いようです。違った角度からの意見や、新しいアイデアを聞かせていただけませんか？"
            else:
                conversation_content = content
            
            if priority == 'high':
                formatted.append(f"### 重要な提案：{conversation_content}")
            else:
                formatted.append(f"### 議論のヒント：{conversation_content}")
            
            if confidence > 0:
                confidence_text = "とても確信" if confidence > 0.8 else "まあまあ確信" if confidence > 0.5 else "少し確信"
                formatted.append(f"    （{confidence_text}しています：{confidence:.2f}）")
        
        return "\n".join(formatted)

    def build_llm_optimized_prompt(self, directives: List[Dict[str, Any]]) -> str:
        """
        Build LLM-optimized prompt from directives.
        
        Args:
            directives (List[Dict[str, Any]]): List of directive dictionaries
            
        Returns:
            str: Optimized prompt string
        """
        if not directives:
            return ""
        
        # Filter high priority directives
        high_priority = [d for d in directives if d.get('priority') == 'high']
        
        if high_priority:
            # Use structured format for high priority
            return self.format_structured_directive(high_priority)
        else:
            # Use simple format for normal priority with conversational tone
            content = directives[0].get('content', '')
            rule_type = directives[0].get('type', 'general')
            
            if rule_type == 'cohesion':
                return "そろそろメインの話題に戻って、みんなで建設的な議論をしませんか？"
            elif rule_type == 'alignment':
                return "いい方向で議論が進んでいますね。この調子でさらに深めていきましょう。"
            elif rule_type == 'separation':
                return "違った視点からの意見も聞いてみたいです。新しいアイデアはありませんか？"
            else:
                return f"議論への提案：{content}"

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

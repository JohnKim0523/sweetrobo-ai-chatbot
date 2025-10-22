"""
Context Manager for Conversation History
Maintains and enriches queries with conversation context
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ConversationContextManager:
    """
    Manages conversation context to make queries context-aware.

    Example:
    User: "My machine has error 4012"
    Bot: "Check the nozzle..."
    User: "What about temperature?" ← needs context from previous messages
    """

    def __init__(self, max_context_messages: int = 10):
        self.max_context_messages = max_context_messages

    def build_contextualized_query(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]],
        machine_type: str
    ) -> tuple[str, str]:
        """
        Enriches query with conversation context.

        Returns:
            (enriched_query, context_summary)
        """

        # Get recent history (last 10 messages)
        recent_history = conversation_history[-self.max_context_messages:]

        if not recent_history:
            return current_query, ""

        # Check if query is short and likely needs context
        is_short_query = len(current_query.split()) <= 5
        has_pronouns = any(word in current_query.lower() for word in [
            'it', 'that', 'this', 'what about', 'how about', 'also', 'and'
        ])

        if not is_short_query and not has_pronouns:
            return current_query, ""  # Long, specific query doesn't need enrichment

        # Extract key topics from history
        context_topics = self._extract_topics_from_history(recent_history)

        # Build context summary
        context_summary = self._build_context_summary(recent_history, context_topics)

        # Enrich query
        enriched = self._enrich_query(current_query, context_topics, machine_type)

        logger.info(f"Original query: '{current_query}'")
        logger.info(f"Enriched query: '{enriched}'")
        logger.info(f"Context: {context_summary}")

        return enriched, context_summary

    def _extract_topics_from_history(self, history: List[Dict]) -> Dict[str, List[str]]:
        """Extract error codes, symptoms, parts mentioned in conversation."""

        topics = {
            'error_codes': set(),
            'symptoms': set(),
            'parts': set(),
            'actions_tried': set()
        }

        # Common parts vocabulary
        parts_vocab = {
            'nozzle', 'sensor', 'motor', 'heating', 'element', 'circuit', 'board',
            'furnace', 'burner', 'pump', 'valve', 'wire', 'cable', 'antenna',
            'screen', 'display', 'button', 'fan', 'belt', 'drum'
        }

        symptoms_vocab = {
            'noise', 'spinning', 'heat', 'cold', 'stuck', 'jam', 'leak', 'error',
            'broken', 'not working', 'failing', 'intermittent', 'slow'
        }

        actions_vocab = {
            'cleaned', 'replaced', 'checked', 'reset', 'restarted', 'tested',
            'adjusted', 'tightened', 'removed', 'installed'
        }

        for entry in history:
            content = entry.get('content', '').lower()

            # Extract error codes
            import re
            error_codes = re.findall(r'(40\d{2})', content)
            topics['error_codes'].update(error_codes)

            # Extract parts
            for part in parts_vocab:
                if part in content:
                    topics['parts'].add(part)

            # Extract symptoms
            for symptom in symptoms_vocab:
                if symptom in content:
                    topics['symptoms'].add(symptom)

            # Extract actions
            for action in actions_vocab:
                if action in content:
                    topics['actions_tried'].add(action)

        # Convert sets to lists
        return {k: list(v) for k, v in topics.items()}

    def _build_context_summary(self, history: List[Dict], topics: Dict) -> str:
        """Build a human-readable context summary."""

        summary_parts = []

        if topics['error_codes']:
            summary_parts.append(f"Error codes discussed: {', '.join(topics['error_codes'])}")

        if topics['parts']:
            summary_parts.append(f"Parts mentioned: {', '.join(topics['parts'][:3])}")

        if topics['symptoms']:
            summary_parts.append(f"Symptoms: {', '.join(topics['symptoms'][:3])}")

        if topics['actions_tried']:
            summary_parts.append(f"Actions tried: {', '.join(topics['actions_tried'][:3])}")

        return " | ".join(summary_parts)

    def _enrich_query(self, query: str, topics: Dict, machine_type: str) -> str:
        """
        Enrich short/ambiguous queries with context.

        Examples:
        - "what about temperature?" → "what about temperature for error 4012?"
        - "how do I check it?" → "how do I check the nozzle for error 4012?"
        """

        query_lower = query.lower()

        # Don't enrich if already specific
        if len(query.split()) > 8:
            return query

        enrichments = []

        # Add error code context if missing
        if topics['error_codes'] and not any(code in query for code in topics['error_codes']):
            enrichments.append(f"error {topics['error_codes'][0]}")

        # Add part context if query uses pronouns
        if any(pronoun in query_lower for pronoun in ['it', 'this', 'that']):
            if topics['parts']:
                enrichments.append(f"the {topics['parts'][0]}")

        # Add symptom context if relevant
        if 'why' in query_lower and topics['symptoms']:
            enrichments.append(f"({topics['symptoms'][0]})")

        if enrichments:
            # Append context to query
            enriched = f"{query} [Context: {', '.join(enrichments)}]"
            return enriched

        return query

    def should_ask_clarification(
        self,
        query: str,
        confidence: float,
        history: List[Dict]
    ) -> Optional[str]:
        """
        Determine if clarification is needed based on context.

        Returns clarification question or None.
        """

        if confidence > 0.7:
            return None  # High confidence, no clarification needed

        query_lower = query.lower()

        # Check if this is a vague follow-up
        if len(history) > 0:
            if any(word in query_lower for word in ['what about', 'how about', 'also']):
                if confidence < 0.5:
                    return (
                        "I want to make sure I understand. Are you asking about:\n"
                        "1. The same issue we discussed\n"
                        "2. A different problem\n"
                        "3. A specific part or error code\n\n"
                        "Please clarify so I can help better."
                    )

        # Check for ambiguous pronouns without context
        if any(word in query_lower for word in ['it', 'this', 'that']) and not history:
            return "Could you be more specific about what you're referring to?"

        return None

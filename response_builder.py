"""
Multi-Answer Response Builder
Intelligently combines multiple relevant solutions
"""

import logging
from typing import List, Dict, Optional
from openai import OpenAI
import os
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

config = dotenv_values(".env")


class ResponseBuilder:
    """
    Builds rich responses from multiple answer candidates.

    Handles:
    - Single high-confidence answers
    - Multiple complementary solutions
    - Step-by-step troubleshooting
    """

    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY") or config.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_key) if openai_key else None

    def build_response(
        self,
        user_query: str,
        top_matches: List[tuple],
        max_answers: int = 3
    ) -> Dict[str, any]:
        """
        Build a response from top matches.

        Args:
            user_query: User's question
            top_matches: List of (match, usefulness, confidence) tuples
            max_answers: Maximum number of solutions to include

        Returns:
            {
                'answer': str,
                'confidence': float,
                'num_sources': int,
                'response_type': 'single'|'multi'|'clarification'
            }
        """

        if not top_matches:
            return self._build_no_match_response(user_query)

        # Check if top match is very confident
        best_match = top_matches[0]
        best_score = best_match[0].score
        best_answer = best_match[0].metadata.get('a', '')

        # Single high-confidence answer
        if best_score >= 0.85:
            return {
                'answer': self._format_single_answer(best_answer),
                'confidence': best_score,
                'num_sources': 1,
                'response_type': 'single',
                'sources': [best_match[0].metadata.get('q', '')]
            }

        # Multiple potential solutions (medium confidence)
        if 0.6 <= best_score < 0.85 and len(top_matches) >= 2:
            return self._build_multi_answer_response(user_query, top_matches[:max_answers])

        # Low confidence - ask for clarification
        if best_score < 0.6:
            return self._build_clarification_response(user_query, top_matches[:3])

        # Fallback: single answer with medium confidence
        return {
            'answer': self._format_single_answer(best_answer),
            'confidence': best_score,
            'num_sources': 1,
            'response_type': 'single',
            'sources': [best_match[0].metadata.get('q', '')]
        }

    def _build_multi_answer_response(
        self,
        user_query: str,
        matches: List[tuple]
    ) -> Dict[str, any]:
        """
        Build response with multiple complementary solutions.

        Example output:
        "I found 3 potential solutions for error 4012:

        **Solution 1: Check Nozzle**
        [answer 1]

        **Solution 2: Verify Heating Element**
        [answer 2]

        **Solution 3: Reset System**
        [answer 3]

        Try these in order. Let me know which one works!"
        """

        answers = []
        for match, usefulness, confidence in matches:
            metadata = match.metadata
            answer = metadata.get('a', '').strip()
            question = metadata.get('q', '').strip()

            if answer and answer not in [a['text'] for a in answers]:
                # Extract title from answer
                title = self._extract_solution_title(answer, question)
                answers.append({
                    'title': title,
                    'text': answer,
                    'score': match.score
                })

        if not answers:
            return self._build_no_match_response(user_query)

        # Format multi-answer response
        if len(answers) == 1:
            return {
                'answer': self._format_single_answer(answers[0]['text']),
                'confidence': answers[0]['score'],
                'num_sources': 1,
                'response_type': 'single',
                'sources': [matches[0][0].metadata.get('q', '')]
            }

        # Build numbered list of solutions
        response_parts = [
            f"I found {len(answers)} potential solutions for your issue:\n"
        ]

        for i, ans in enumerate(answers, 1):
            response_parts.append(f"\n**Solution {i}: {ans['title']}**")
            # Truncate long answers for multi-solution view
            truncated = self._truncate_answer(ans['text'], max_length=300)
            response_parts.append(truncated)
            response_parts.append("")  # Blank line

        response_parts.append(
            "\n💡 **Try these in order.** If one doesn't work, try the next. "
            "Let me know which solution worked or if you need more details on any step."
        )

        avg_confidence = sum(a['score'] for a in answers) / len(answers)

        return {
            'answer': '\n'.join(response_parts),
            'confidence': avg_confidence,
            'num_sources': len(answers),
            'response_type': 'multi',
            'sources': [matches[i][0].metadata.get('q', '') for i in range(len(answers))]
        }

    def _build_clarification_response(
        self,
        user_query: str,
        matches: List[tuple]
    ) -> Dict[str, any]:
        """
        Build a clarification request when confidence is low.

        Example:
        "I'm not sure exactly what you're asking. Did you mean:
        1. How to fix error 4012?
        2. How to reset the nozzle?
        3. How to clean the heating element?

        Please select one or rephrase your question."
        """

        if not matches:
            return self._build_no_match_response(user_query)

        # Extract potential interpretations
        options = []
        for match, _, _ in matches[:3]:
            q = match.metadata.get('q', '').strip()
            if q and q not in options:
                options.append(q)

        if not options:
            return self._build_no_match_response(user_query)

        response_parts = [
            "I found a few possible matches. Did you mean:\n"
        ]

        for i, option in enumerate(options, 1):
            response_parts.append(f"{i}. {option}")

        response_parts.append(
            "\nPlease select a number or rephrase your question with more details."
        )

        return {
            'answer': '\n'.join(response_parts),
            'confidence': matches[0][0].score if matches else 0.0,
            'num_sources': len(options),
            'response_type': 'clarification',
            'sources': options
        }

    def _build_no_match_response(self, user_query: str) -> Dict[str, any]:
        """Build response when no matches found."""

        return {
            'answer': (
                "I couldn't find a specific answer for your question in my knowledge base. "
                "\n\n**What you can do:**\n"
                "• Try rephrasing with more details (error codes, symptoms, parts)\n"
                "• Contact support at support@sweetrobo.com\n"
                "• Call our support line for immediate help\n\n"
                "I've logged this question so our team can add an answer to help future users."
            ),
            'confidence': 0.0,
            'num_sources': 0,
            'response_type': 'no_match',
            'sources': []
        }

    def _extract_solution_title(self, answer: str, question: str) -> str:
        """
        Extract a short title for the solution.

        Priority:
        1. First line if it's short and descriptive
        2. Extract from "Problem:" line
        3. First 50 chars of answer
        4. Derive from question
        """

        lines = answer.split('\n')

        # Check first line
        if lines and len(lines[0]) < 60 and not lines[0].startswith('Problem'):
            return lines[0].strip('*#: ')

        # Check for "Problem:" line
        for line in lines[:3]:
            if line.strip().startswith('Problem'):
                title = line.replace('Problem:', '').replace('Problem Summary:', '').strip()
                if len(title) < 60:
                    return title

        # Check for error codes in answer
        import re
        error_match = re.search(r'error\s+(\d{4})', answer.lower())
        if error_match:
            return f"Error {error_match.group(1)} Solution"

        # Fallback: use question
        if len(question) < 60:
            return question

        # Last resort: truncate answer
        return answer[:50].strip() + "..."

    def _truncate_answer(self, answer: str, max_length: int = 300) -> str:
        """Truncate long answers while preserving meaning."""

        if len(answer) <= max_length:
            return answer

        # Try to truncate at sentence boundary
        truncated = answer[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')

        boundary = max(last_period, last_newline)

        if boundary > max_length * 0.7:  # If we can keep 70%+ of text
            return answer[:boundary + 1] + "\n\n[Click 'More Details' to see full solution]"

        return truncated + "...\n\n[Click 'More Details' to see full solution]"

    def _format_single_answer(self, answer: str) -> str:
        """Format a single answer for display."""

        # Remove "Problem:" prefix if present
        lines = answer.strip().split('\n')
        if lines and lines[0].strip().startswith('Problem:'):
            lines = lines[1:]

        formatted = '\n'.join(lines).strip()

        # Add footer
        formatted += "\n\n---\n💬 **Need more help?** Let me know if this didn't resolve the issue."

        return formatted


# Test the response builder
if __name__ == "__main__":
    print("Testing Response Builder")
    print("=" * 60)

    # Mock matches for testing
    class MockMatch:
        def __init__(self, score, q, a):
            self.score = score
            self.metadata = {'q': q, 'a': a}

    builder = ResponseBuilder()

    # Test 1: High confidence single answer
    print("\n1. HIGH CONFIDENCE (score 0.9):")
    matches = [
        (MockMatch(0.9, "How to fix error 4012?", "Problem: Error 4012\n\nCheck the nozzle sensor..."), 9, 0.9)
    ]
    result = builder.build_response("error 4012", matches)
    print(f"Type: {result['response_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Answer preview: {result['answer'][:100]}...")

    # Test 2: Multiple solutions
    print("\n2. MULTIPLE SOLUTIONS (scores 0.7, 0.68, 0.65):")
    matches = [
        (MockMatch(0.7, "Error 4012", "Check nozzle sensor for blockages..."), 8, 0.7),
        (MockMatch(0.68, "4012 error", "Verify heating element connection..."), 8, 0.68),
        (MockMatch(0.65, "Machine error 4012", "Reset the system by powering off..."), 7, 0.65),
    ]
    result = builder.build_response("how to fix 4012", matches)
    print(f"Type: {result['response_type']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Num sources: {result['num_sources']}")
    print(f"\nAnswer:\n{result['answer']}")

    # Test 3: Clarification needed
    print("\n3. LOW CONFIDENCE - CLARIFICATION (score 0.45):")
    matches = [
        (MockMatch(0.45, "How to fix error 4012?", "Check nozzle..."), 6, 0.45),
        (MockMatch(0.43, "How to clean machine?", "Clean with warm water..."), 6, 0.43),
    ]
    result = builder.build_response("machine issue", matches)
    print(f"Type: {result['response_type']}")
    print(f"\nAnswer:\n{result['answer']}")

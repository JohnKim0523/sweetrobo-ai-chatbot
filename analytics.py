"""
Analytics and Telemetry for Chatbot Performance Monitoring

Tracks:
- Query success/failure rates
- Response confidence distributions
- Common unanswered questions
- User satisfaction (if feedback implemented)
- API costs
- Latency metrics
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict, Counter
import csv

logger = logging.getLogger(__name__)


class ChatbotAnalytics:
    """
    Tracks and analyzes chatbot performance.

    Data stored in JSON files for simplicity.
    Can be upgraded to database (SQLite, PostgreSQL) later.
    """

    def __init__(self, data_dir: str = "./analytics_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.interactions_file = self.data_dir / "interactions.jsonl"
        self.daily_stats_file = self.data_dir / "daily_stats.json"
        self.failed_queries_file = self.data_dir / "failed_queries.json"

    def log_interaction(
        self,
        thread_id: str,
        user_query: str,
        response: str,
        confidence: float,
        num_matches: int,
        response_type: str,
        latency_ms: float,
        machine_type: str,
        error_code: Optional[str] = None,
        context_used: bool = False
    ):
        """
        Log a single user interaction.

        Args:
            thread_id: Conversation ID
            user_query: User's question
            response: Bot's response
            confidence: Confidence score (0-1)
            num_matches: Number of dataset matches found
            response_type: 'single'|'multi'|'clarification'|'no_match'
            latency_ms: Response time in milliseconds
            machine_type: Which machine
            error_code: Error code if mentioned
            context_used: Whether conversation context was used
        """

        interaction = {
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'user_query': user_query,
            'response_preview': response[:200],  # Truncate for storage
            'confidence': confidence,
            'num_matches': num_matches,
            'response_type': response_type,
            'latency_ms': latency_ms,
            'machine_type': machine_type,
            'error_code': error_code,
            'context_used': context_used,
            'user_feedback': None  # Updated later if user provides feedback
        }

        # Append to JSONL file (one JSON per line)
        with open(self.interactions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction) + '\n')

        # Update daily stats
        self._update_daily_stats(interaction)

        # Log failed queries separately
        if response_type == 'no_match' or confidence < 0.5:
            self._log_failed_query(interaction)

        logger.info(
            f"Logged interaction - Confidence: {confidence:.2f}, "
            f"Type: {response_type}, Latency: {latency_ms:.0f}ms"
        )

    def log_user_feedback(
        self,
        thread_id: str,
        was_helpful: bool,
        feedback_text: Optional[str] = None
    ):
        """
        Log user feedback on a response.

        Args:
            thread_id: Conversation ID
            was_helpful: True if user clicked "Helpful", False if "Not Helpful"
            feedback_text: Optional text feedback
        """

        feedback = {
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'was_helpful': was_helpful,
            'feedback_text': feedback_text
        }

        feedback_file = self.data_dir / "user_feedback.jsonl"
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback) + '\n')

        logger.info(f"Logged user feedback - Thread: {thread_id}, Helpful: {was_helpful}")

    def _update_daily_stats(self, interaction: Dict):
        """Update aggregated daily statistics."""

        today = datetime.now().strftime('%Y-%m-%d')

        # Load existing stats
        if self.daily_stats_file.exists():
            with open(self.daily_stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}

        if today not in stats:
            stats[today] = {
                'total_queries': 0,
                'successful': 0,
                'failed': 0,
                'avg_confidence': 0.0,
                'avg_latency_ms': 0.0,
                'response_types': defaultdict(int),
                'machine_types': defaultdict(int),
                'error_codes': defaultdict(int)
            }

        day_stats = stats[today]
        day_stats['total_queries'] += 1

        if interaction['response_type'] != 'no_match' and interaction['confidence'] >= 0.5:
            day_stats['successful'] += 1
        else:
            day_stats['failed'] += 1

        # Update averages
        total = day_stats['total_queries']
        day_stats['avg_confidence'] = (
            (day_stats['avg_confidence'] * (total - 1) + interaction['confidence']) / total
        )
        day_stats['avg_latency_ms'] = (
            (day_stats['avg_latency_ms'] * (total - 1) + interaction['latency_ms']) / total
        )

        # Update counts
        day_stats['response_types'][interaction['response_type']] = \
            day_stats['response_types'].get(interaction['response_type'], 0) + 1
        day_stats['machine_types'][interaction['machine_type']] = \
            day_stats['machine_types'].get(interaction['machine_type'], 0) + 1

        if interaction['error_code']:
            day_stats['error_codes'][interaction['error_code']] = \
                day_stats['error_codes'].get(interaction['error_code'], 0) + 1

        # Save updated stats
        with open(self.daily_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def _log_failed_query(self, interaction: Dict):
        """Log queries that failed to find good answers."""

        # Load existing failed queries
        if self.failed_queries_file.exists():
            with open(self.failed_queries_file, 'r') as f:
                failed = json.load(f)
        else:
            failed = {'queries': []}

        failed['queries'].append({
            'timestamp': interaction['timestamp'],
            'query': interaction['user_query'],
            'confidence': interaction['confidence'],
            'machine_type': interaction['machine_type'],
            'error_code': interaction['error_code']
        })

        # Keep only last 1000 failed queries
        failed['queries'] = failed['queries'][-1000:]

        with open(self.failed_queries_file, 'w') as f:
            json.dump(failed, f, indent=2)

    def get_daily_report(self, date: Optional[str] = None) -> Dict:
        """
        Get performance report for a specific date.

        Args:
            date: Date string 'YYYY-MM-DD' (default: today)

        Returns:
            Daily statistics dict
        """

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        if not self.daily_stats_file.exists():
            return {'error': 'No data available'}

        with open(self.daily_stats_file, 'r') as f:
            stats = json.load(f)

        return stats.get(date, {'error': f'No data for {date}'})

    def get_weekly_report(self) -> Dict:
        """Get aggregated stats for the past 7 days."""

        if not self.daily_stats_file.exists():
            return {'error': 'No data available'}

        with open(self.daily_stats_file, 'r') as f:
            all_stats = json.load(f)

        # Get last 7 days
        today = datetime.now()
        last_week = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

        weekly = {
            'date_range': f"{last_week[-1]} to {last_week[0]}",
            'total_queries': 0,
            'successful': 0,
            'failed': 0,
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'avg_latency_ms': 0.0,
            'top_error_codes': {},
            'top_machines': {}
        }

        days_with_data = 0
        for date in last_week:
            if date in all_stats:
                day = all_stats[date]
                weekly['total_queries'] += day['total_queries']
                weekly['successful'] += day['successful']
                weekly['failed'] += day['failed']
                weekly['avg_confidence'] += day['avg_confidence']
                weekly['avg_latency_ms'] += day['avg_latency_ms']

                # Aggregate error codes and machines
                for code, count in day.get('error_codes', {}).items():
                    weekly['top_error_codes'][code] = \
                        weekly['top_error_codes'].get(code, 0) + count

                for machine, count in day.get('machine_types', {}).items():
                    weekly['top_machines'][machine] = \
                        weekly['top_machines'].get(machine, 0) + count

                days_with_data += 1

        if days_with_data > 0:
            weekly['avg_confidence'] /= days_with_data
            weekly['avg_latency_ms'] /= days_with_data

        if weekly['total_queries'] > 0:
            weekly['success_rate'] = weekly['successful'] / weekly['total_queries']

        # Sort top items
        weekly['top_error_codes'] = dict(
            sorted(weekly['top_error_codes'].items(), key=lambda x: -x[1])[:10]
        )
        weekly['top_machines'] = dict(
            sorted(weekly['top_machines'].items(), key=lambda x: -x[1])[:10]
        )

        return weekly

    def get_most_common_failed_queries(self, limit: int = 20) -> List[Dict]:
        """Get the most common queries that failed to find answers."""

        if not self.failed_queries_file.exists():
            return []

        with open(self.failed_queries_file, 'r') as f:
            failed = json.load(f)

        # Count query frequencies
        query_counts = Counter(q['query'] for q in failed['queries'])

        # Get top N
        top_failed = [
            {'query': query, 'count': count}
            for query, count in query_counts.most_common(limit)
        ]

        return top_failed

    def export_to_csv(self, output_file: str = "analytics_export.csv"):
        """Export all interactions to CSV for analysis."""

        if not self.interactions_file.exists():
            logger.warning("No interactions to export")
            return

        interactions = []
        with open(self.interactions_file, 'r') as f:
            for line in f:
                interactions.append(json.loads(line))

        if not interactions:
            logger.warning("No interactions to export")
            return

        keys = interactions[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(interactions)

        logger.info(f"Exported {len(interactions)} interactions to {output_file}")


# Global singleton
_analytics_instance = None


def get_analytics() -> ChatbotAnalytics:
    """Get or create singleton analytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = ChatbotAnalytics()
    return _analytics_instance


# Test analytics
if __name__ == "__main__":
    print("Testing Analytics System")
    print("=" * 60)

    analytics = ChatbotAnalytics(data_dir="./test_analytics")

    # Simulate some interactions
    print("\n1. Logging 10 test interactions...")
    import time
    for i in range(10):
        analytics.log_interaction(
            thread_id=f"thread_{i % 3}",
            user_query=f"Test query {i}",
            response=f"Test response {i}",
            confidence=0.5 + (i % 5) * 0.1,
            num_matches=i % 4,
            response_type=['single', 'multi', 'no_match'][i % 3],
            latency_ms=100 + i * 10,
            machine_type=['COTTON_CANDY', 'ICE_CREAM'][i % 2],
            error_code='4012' if i % 3 == 0 else None
        )
        time.sleep(0.01)

    # Get daily report
    print("\n2. Daily Report:")
    daily = analytics.get_daily_report()
    print(json.dumps(daily, indent=2))

    # Get weekly report
    print("\n3. Weekly Report:")
    weekly = analytics.get_weekly_report()
    print(json.dumps(weekly, indent=2))

    # Get failed queries
    print("\n4. Most Common Failed Queries:")
    failed = analytics.get_most_common_failed_queries(limit=5)
    for item in failed:
        print(f"  - '{item['query']}' (failed {item['count']} times)")

    # Export to CSV
    print("\n5. Exporting to CSV...")
    analytics.export_to_csv("test_export.csv")
    print("✓ Exported to test_export.csv")

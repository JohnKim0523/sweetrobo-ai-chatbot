"""
Hybrid Search: Vector + Keyword (BM25)

Solves the problem where pure vector search fails on:
- Exact matches (part numbers, error codes)
- Rare technical terms
- Specific model names
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Combines vector search (Pinecone) with keyword search (BM25).

    Use cases:
    - User searches "PN-4012-X" → BM25 finds exact match
    - User searches "machine not heating" → Vector search finds semantic matches
    - Combined: Best of both worlds
    """

    def __init__(self, dataset_path: str = "cleaned_qa_JSON.json"):
        self.dataset_path = dataset_path
        self.dataset = None
        self.bm25 = None
        self.tokenized_corpus = None
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset and build BM25 index."""

        logger.info(f"Loading dataset from {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

        # Build corpus for BM25 (combine question + answer for better matching)
        corpus = []
        for item in self.dataset:
            q = item.get('q', '').lower()
            a = item.get('a', '').lower()
            # Combine with extra weight on question
            combined = f"{q} {q} {a}"  # Double the question for more weight
            corpus.append(combined)

        # Tokenize corpus
        self.tokenized_corpus = [doc.split() for doc in corpus]

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built with {len(corpus)} documents")

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        machine_type: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """
        Perform keyword search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return
            machine_type: Filter by machine type (optional)

        Returns:
            List of (index, score) tuples
        """

        if not self.bm25:
            logger.error("BM25 index not initialized")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Filter by machine type if specified
        if machine_type:
            filtered_scores = []
            for idx, score in enumerate(scores):
                if self.dataset[idx].get('machine_type') == machine_type:
                    filtered_scores.append((idx, score))
        else:
            filtered_scores = [(idx, score) for idx, score in enumerate(scores)]

        # Sort by score
        filtered_scores.sort(key=lambda x: -x[1])

        # Return top K
        results = filtered_scores[:top_k]

        logger.info(
            f"Keyword search: '{query[:50]}...' → {len(results)} results "
            f"(top score: {results[0][1]:.3f if results else 0})"
        )

        return results

    def hybrid_search(
        self,
        query: str,
        vector_results: List[Dict],
        top_k: int = 10,
        machine_type: Optional[str] = None,
        alpha: float = 0.7
    ) -> List[Dict]:
        """
        Combine vector search results with keyword search.

        Args:
            query: Search query
            vector_results: Results from Pinecone (list of dicts with 'id', 'score', 'metadata')
            top_k: Number of final results
            machine_type: Filter by machine type
            alpha: Weight for vector search (0=only keyword, 1=only vector, 0.5=equal)

        Returns:
            Re-ranked list of results
        """

        # Perform keyword search
        keyword_results = self.keyword_search(query, top_k=top_k*2, machine_type=machine_type)

        # Build mapping of dataset index to keyword score
        keyword_scores = {idx: score for idx, score in keyword_results}

        # Normalize keyword scores to 0-1 range
        if keyword_scores:
            max_kw_score = max(keyword_scores.values())
            if max_kw_score > 0:
                keyword_scores = {
                    idx: score / max_kw_score
                    for idx, score in keyword_scores.items()
                }

        # Build mapping of ID to vector result
        vector_map = {r['id']: r for r in vector_results}

        # Combine scores
        combined_results = {}

        # Add vector results
        for r in vector_results:
            result_id = r['id']
            vector_score = r['score']

            # Try to find corresponding keyword score
            # Assuming ID format like "thread_1" where 1 is the dataset index
            try:
                dataset_idx = int(result_id.split('_')[-1]) if '_' in result_id else int(result_id)
            except:
                dataset_idx = None

            keyword_score = keyword_scores.get(dataset_idx, 0.0)

            # Combine scores using weighted average
            combined_score = alpha * vector_score + (1 - alpha) * keyword_score

            combined_results[result_id] = {
                **r,
                'vector_score': vector_score,
                'keyword_score': keyword_score,
                'combined_score': combined_score
            }

        # Add keyword-only results (not in vector results)
        for idx, kw_score in keyword_scores.items():
            # Create pseudo ID
            result_id = f"thread_{idx}"

            if result_id not in combined_results:
                # This result was found by keyword search but not vector search
                # Give it a chance if keyword score is high
                if kw_score > 0.5:  # Threshold for keyword-only results
                    combined_results[result_id] = {
                        'id': result_id,
                        'score': kw_score * (1 - alpha),  # Adjusted by alpha
                        'metadata': self.dataset[idx],
                        'vector_score': 0.0,
                        'keyword_score': kw_score,
                        'combined_score': kw_score * (1 - alpha)
                    }

        # Sort by combined score
        ranked_results = sorted(
            combined_results.values(),
            key=lambda x: -x['combined_score']
        )

        # Log top results
        logger.info(f"Hybrid search: '{query[:50]}...'")
        for i, r in enumerate(ranked_results[:3], 1):
            logger.info(
                f"  #{i}: Combined={r['combined_score']:.3f} "
                f"(Vector={r['vector_score']:.3f}, Keyword={r['keyword_score']:.3f}) "
                f"- {r['metadata'].get('q', '')[:50]}..."
            )

        return ranked_results[:top_k]

    def should_use_hybrid(self, query: str) -> bool:
        """
        Determine if hybrid search should be used for this query.

        Use hybrid when:
        - Query contains exact codes/part numbers
        - Query is very specific (long, technical terms)
        - Query mentions rare technical terms
        """

        query_lower = query.lower()

        # Check for part numbers (PN-XXXX, Model-XXXX, etc.)
        import re
        if re.search(r'[a-z]{2,4}-\d{4}', query_lower):
            return True

        # Check for error codes with context (not just "4012" alone)
        if re.search(r'(error|code|alert)\s*\d{4}', query_lower):
            return True

        # Check for very specific technical terms
        technical_terms = [
            'circuit board', 'motherboard', 'pcb', 'firmware',
            'calibration', 'voltage', 'amperage', 'resistance',
            'model number', 'serial number', 'part number'
        ]
        if any(term in query_lower for term in technical_terms):
            return True

        # Check query length (long queries are often specific)
        if len(query.split()) > 10:
            return True

        return False


# Global singleton
_hybrid_search_instance = None


def get_hybrid_search() -> HybridSearchEngine:
    """Get or create singleton hybrid search instance."""
    global _hybrid_search_instance
    if _hybrid_search_instance is None:
        _hybrid_search_instance = HybridSearchEngine()
    return _hybrid_search_instance


# Test hybrid search
if __name__ == "__main__":
    print("Testing Hybrid Search Engine")
    print("=" * 60)

    engine = HybridSearchEngine()

    test_queries = [
        "error 4012",  # Should favor BM25 for exact match
        "machine not heating properly",  # Should favor vector search
        "PN-4012-X replacement part",  # Should strongly favor BM25
        "nozzle stuck",  # Both should work
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Should use hybrid: {engine.should_use_hybrid(query)}")
        print(f"{'='*60}")

        # Keyword search only
        keyword_results = engine.keyword_search(query, top_k=3)
        print("\nTop 3 Keyword Results:")
        for idx, (dataset_idx, score) in enumerate(keyword_results[:3], 1):
            item = engine.dataset[dataset_idx]
            print(f"  {idx}. Score: {score:.3f} - Q: {item.get('q', '')[:60]}...")

        # Note: Can't test vector search here without Pinecone access
        # But you can see how keyword search performs

    print("\n" + "="*60)
    print("Hybrid search combines these keyword results with Pinecone's")
    print("vector search results for best accuracy!")

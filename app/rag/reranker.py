import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
from sentence_transformers import CrossEncoder

from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)


@dataclass
class RerankedResult:
    """Result after reranking."""
    id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float
    reranked_score: float
    ranking_change: int  # Positive if improved, negative if worsened


class CrossEncoderReranker:
    """Production-grade cross-encoder reranker for improved relevance."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.model = None
        self.batch_size = 32
        self.max_length = 512

        # Cache for frequent query-document pairs
        self.rerank_cache = {}
        self.cache_size = 5000

    async def initialize(self):
        """Initialize the cross-encoder model."""
        if self.model is None:
            try:
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self.model = CrossEncoder(self.model_name, max_length=self.max_length)
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise

    async def rerank(self,
                     query: str,
                     documents: List[str],
                     metadata_list: Optional[List[Dict[str, Any]]] = None,
                     top_k: Optional[int] = None) -> List[RerankedResult]:
        """Rerank documents based on relevance to query."""
        await self.initialize()

        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents for query: {query[:100]}...")

        try:
            # Prepare query-document pairs
            pairs = [(query, doc) for doc in documents]

            # Get similarity scores
            scores = await self._batch_predict(pairs)

            # Create reranked results
            results = self._create_reranked_results(
                documents, scores, metadata_list
            )

            # Apply top_k if specified
            if top_k:
                results = results[:top_k]

            logger.info(f"Reranking completed. Top score: {results[0].reranked_score:.3f}")
            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Return documents in original order as fallback
            return self._create_fallback_results(documents, metadata_list)

    async def rerank_with_scores(self,
                                 query: str,
                                 documents_with_scores: List[Tuple[str, float]],
                                 metadata_list: Optional[List[Dict]] = None) -> List[RerankedResult]:
        """Rerank documents that already have initial scores."""
        await self.initialize()

        if not documents_with_scores:
            return []

        # Extract documents and original scores
        documents = [doc for doc, _ in documents_with_scores]
        original_scores = [score for _, score in documents_with_scores]

        # Rerank
        reranked_results = await self.rerank(query, documents, metadata_list)

        # Update with original scores
        for result, original_score in zip(reranked_results, original_scores):
            result.original_score = original_score
            result.ranking_change = 0  # Will be recalculated

        # Sort by reranked score
        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)

        # Calculate ranking changes
        self._calculate_ranking_changes(reranked_results)

        return reranked_results

    async def _batch_predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Batch prediction for efficiency."""
        # Check cache first
        cached_scores = []
        uncached_pairs = []
        uncached_indices = []

        for i, (query, doc) in enumerate(pairs):
            cache_key = self._get_cache_key(query, doc)
            if cache_key in self.rerank_cache:
                cached_scores.append((i, self.rerank_cache[cache_key]))
            else:
                uncached_pairs.append((query, doc))
                uncached_indices.append(i)

        # Initialize result array
        scores = [0.0] * len(pairs)

        # Fill cached scores
        for idx, score in cached_scores:
            scores[idx] = score

        # Predict for uncached pairs
        if uncached_pairs:
            try:
                # Split into batches
                batch_scores = []
                for i in range(0, len(uncached_pairs), self.batch_size):
                    batch = uncached_pairs[i:i + self.batch_size]

                    # Prepare batch for model
                    batch_pairs = list(batch)

                    # Predict
                    batch_result = self.model.predict(
                        batch_pairs,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )

                    batch_scores.extend(batch_result.tolist())

                # Update scores and cache
                for idx, pair_idx, score in zip(uncached_indices, range(len(uncached_pairs)), batch_scores):
                    scores[idx] = score

                    # Cache result
                    query, doc = uncached_pairs[pair_idx]
                    cache_key = self._get_cache_key(query, doc)
                    self._update_cache(cache_key, score)

            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                # Use fallback scores for failed predictions
                for idx in uncached_indices:
                    scores[idx] = 0.5  # Neutral score

        return scores

    def _create_reranked_results(self,
                                 documents: List[str],
                                 scores: List[float],
                                 metadata_list: Optional[List[Dict]]) -> List[RerankedResult]:
        """Create reranked results from documents and scores."""
        results = []

        for i, (doc, score) in enumerate(zip(documents, scores)):
            metadata = {}
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]

            # Generate document ID if not present
            doc_id = metadata.get("id", f"doc_{i}")

            results.append(RerankedResult(
                id=doc_id,
                content=doc,
                metadata=metadata,
                original_score=0.0,  # Will be updated if available
                reranked_score=float(score),
                ranking_change=0
            ))

        # Sort by reranked score
        results.sort(key=lambda x: x.reranked_score, reverse=True)

        # Assign ranking positions
        for rank, result in enumerate(results):
            result.metadata["rerank_position"] = rank + 1

        return results

    def _create_fallback_results(self,
                                 documents: List[str],
                                 metadata_list: Optional[List[Dict]]) -> List[RerankedResult]:
        """Create fallback results when reranking fails."""
        results = []

        for i, doc in enumerate(documents):
            metadata = {}
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]

            doc_id = metadata.get("id", f"doc_{i}")

            results.append(RerankedResult(
                id=doc_id,
                content=doc,
                metadata=metadata,
                original_score=0.5,
                reranked_score=0.5,
                ranking_change=0
            ))

        return results

    def _calculate_ranking_changes(self, results: List[RerankedResult]):
        """Calculate how much each result moved in ranking."""
        # Sort by original score to get original ranking
        original_order = sorted(
            results,
            key=lambda x: x.original_score,
            reverse=True
        )

        # Create mapping from ID to original position
        original_positions = {}
        for pos, result in enumerate(original_order):
            original_positions[result.id] = pos

        # Calculate changes for current order
        for current_pos, result in enumerate(results):
            original_pos = original_positions.get(result.id, current_pos)
            result.ranking_change = original_pos - current_pos  # Positive if improved

    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair."""
        import hashlib

        # Use first 100 chars of each for key (balance between accuracy and performance)
        query_key = query[:100]
        doc_key = document[:100]

        key_string = f"{query_key}_{doc_key}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _update_cache(self, key: str, score: float):
        """Update rerank cache."""
        if len(self.rerank_cache) >= self.cache_size:
            # Remove random entry (simple cache eviction)
            import random
            if self.rerank_cache:
                random_key = random.choice(list(self.rerank_cache.keys()))
                del self.rerank_cache[random_key]

        self.rerank_cache[key] = score

    async def evaluate_reranking(self,
                                 query: str,
                                 original_results: List[Dict[str, Any]],
                                 ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate reranking performance."""
        await self.initialize()

        if not original_results:
            return {"error": "No results to evaluate"}

        # Extract documents
        documents = [result.get("content", "") for result in original_results]
        original_scores = [result.get("score", 0.0) for result in original_results]

        # Rerank
        reranked = await self.rerank(query, documents)

        # Calculate metrics
        metrics = {
            "query": query[:100],
            "num_documents": len(documents),
            "score_statistics": {
                "original_mean": float(np.mean(original_scores)),
                "original_std": float(np.std(original_scores)),
                "reranked_mean": float(np.mean([r.reranked_score for r in reranked])),
                "reranked_std": float(np.std([r.reranked_score for r in reranked]))
            }
        }

        # Calculate ranking changes
        ranking_changes = [r.ranking_change for r in reranked]
        metrics["ranking_changes"] = {
            "mean_change": float(np.mean(ranking_changes)),
            "max_improvement": max(ranking_changes, default=0),
            "max_decline": min(ranking_changes, default=0),
            "improved_count": sum(1 for change in ranking_changes if change > 0),
            "declined_count": sum(1 for change in ranking_changes if change < 0)
        }

        # Compare with ground truth if available
        if ground_truth:
            relevance_scores = await self._calculate_relevance_scores(
                query, ground_truth
            )
            metrics["ground_truth_comparison"] = {
                "average_relevance": float(np.mean(relevance_scores)),
                "top_5_relevance": float(np.mean(relevance_scores[:5]) if relevance_scores else 0)
            }

        # Add top reranked results
        metrics["top_reranked"] = [
            {
                "id": r.id,
                "reranked_score": r.reranked_score,
                "ranking_change": r.ranking_change
            }
            for r in reranked[:5]
        ]

        return metrics

    async def _calculate_relevance_scores(self, query: str, documents: List[str]) -> List[float]:
        """Calculate relevance scores for evaluation."""
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = await self._batch_predict(pairs)

        return scores

    def clear_cache(self):
        """Clear reranking cache."""
        self.rerank_cache.clear()
        logger.info("Reranking cache cleared")

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model."""
        await self.initialize()

        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "cache_size": len(self.rerank_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified)."""
        # This would track hits/misses in production
        return 0.0  # Placeholder
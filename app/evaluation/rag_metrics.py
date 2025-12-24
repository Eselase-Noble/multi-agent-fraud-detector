import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGEvaluation:
    """RAG evaluation results."""
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    ground_truth: Optional[str] = None

    # Metrics
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None

    # Additional metrics
    hallucination_score: Optional[float] = None
    citation_precision: Optional[float] = None
    citation_recall: Optional[float] = None

    # Timestamps
    evaluation_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "hallucination_score": self.hallucination_score,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "evaluation_time": self.evaluation_time.isoformat() if self.evaluation_time else None,
            "retrieved_count": len(self.retrieved_documents),
            "has_ground_truth": self.ground_truth is not None
        }


class RAGMetricsCollector:
    """Production-grade RAG metrics collection and evaluation."""

    def __init__(self):
        self.evaluations = []
        self.metrics_history = []

        # Thresholds for alerting
        self.thresholds = {
            "faithfulness": 0.7,
            "answer_relevance": 0.6,
            "context_relevance": 0.5,
            "hallucination_score": 0.3
        }

    async def evaluate_response(self,
                                query: str,
                                response: str,
                                retrieved_documents: List[Dict[str, Any]],
                                ground_truth: Optional[str] = None) -> RAGEvaluation:
        """Evaluate a RAG response."""
        logger.info(f"Evaluating RAG response for query: {query[:100]}...")

        evaluation = RAGEvaluation(
            query=query,
            response=response,
            retrieved_documents=retrieved_documents,
            ground_truth=ground_truth,
            evaluation_time=datetime.now()
        )

        try:
            # Calculate metrics
            evaluation.faithfulness = await self._calculate_faithfulness(
                response, retrieved_documents
            )

            evaluation.answer_relevance = await self._calculate_answer_relevance(
                query, response
            )

            evaluation.context_relevance = await self._calculate_context_relevance(
                query, retrieved_documents
            )

            if ground_truth:
                evaluation.context_recall = await self._calculate_context_recall(
                    ground_truth, retrieved_documents
                )
                evaluation.context_precision = await self._calculate_context_precision(
                    query, retrieved_documents, ground_truth
                )

            evaluation.hallucination_score = await self._calculate_hallucination_score(
                response, retrieved_documents
            )

            # Calculate citation metrics if citations are present
            evaluation.citation_precision, evaluation.citation_recall = \
                await self._calculate_citation_metrics(response, retrieved_documents)

            # Store evaluation
            self.evaluations.append(evaluation)

            # Check thresholds and alert if needed
            await self._check_thresholds(evaluation)

            logger.info(f"RAG evaluation completed: faithfulness={evaluation.faithfulness:.2f}")

            return evaluation

        except Exception as e:
            logger.error(f"RAG evaluation failed: {e}")
            # Return evaluation with None metrics
            return evaluation

    async def _calculate_faithfulness(self,
                                      response: str,
                                      documents: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness (how well response is grounded in documents)."""
        # Simple implementation - in production would use NLI model

        # Extract claims from response
        claims = self._extract_claims(response)

        if not claims:
            return 0.0

        # Check each claim against documents
        supported_claims = 0

        for claim in claims:
            if self._claim_supported(claim, documents):
                supported_claims += 1

        return supported_claims / len(claims)

    async def _calculate_answer_relevance(self, query: str, response: str) -> float:
        """Calculate how relevant the answer is to the query."""
        # Simple cosine similarity of bag-of-words
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(response_words)
        return len(intersection) / len(query_words)

    async def _calculate_context_relevance(self,
                                           query: str,
                                           documents: List[Dict[str, Any]]) -> float:
        """Calculate how relevant retrieved documents are to query."""
        if not documents:
            return 0.0

        query_words = set(query.lower().split())
        total_relevance = 0.0

        for doc in documents:
            content = doc.get("content", "").lower()
            doc_words = set(content.split())

            if query_words and doc_words:
                intersection = query_words.intersection(doc_words)
                relevance = len(intersection) / len(query_words)
                total_relevance += relevance

        return total_relevance / len(documents)

    async def _calculate_context_recall(self,
                                        ground_truth: str,
                                        documents: List[Dict[str, Any]]) -> float:
        """Calculate how much of ground truth is covered by documents."""
        gt_words = set(ground_truth.lower().split())

        if not gt_words:
            return 0.0

        # Combine all document content
        all_doc_content = " ".join([d.get("content", "").lower() for d in documents])
        doc_words = set(all_doc_content.split())

        intersection = gt_words.intersection(doc_words)
        return len(intersection) / len(gt_words)

    async def _calculate_context_precision(self,
                                           query: str,
                                           documents: List[Dict[str, Any]],
                                           ground_truth: str) -> float:
        """Calculate precision of retrieved documents."""
        if not documents:
            return 0.0

        # This is simplified - would need labeled relevance data in production
        query_words = set(query.lower().split())
        gt_words = set(ground_truth.lower().split())
        relevant_words = query_words.union(gt_words)

        relevant_docs = 0

        for doc in documents:
            content = doc.get("content", "").lower()
            doc_words = set(content.split())

            # Check if document contains relevant words
            if relevant_words.intersection(doc_words):
                relevant_docs += 1

        return relevant_docs / len(documents)

    async def _calculate_hallucination_score(self,
                                             response: str,
                                             documents: List[Dict[str, Any]]) -> float:
        """Calculate hallucination score."""
        claims = self._extract_claims(response)

        if not claims:
            return 0.0

        hallucinated_claims = 0

        for claim in claims:
            if not self._claim_supported(claim, documents):
                hallucinated_claims += 1

        return hallucinated_claims / len(claims)

    async def _calculate_citation_metrics(self,
                                          response: str,
                                          documents: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Calculate citation precision and recall."""
        # Extract citations from response
        citations = self._extract_citations(response)

        if not citations:
            return 0.0, 0.0

        # Get document IDs
        doc_ids = [doc.get("id") for doc in documents]

        # Calculate precision (citations that match documents)
        correct_citations = [c for c in citations if c in doc_ids]
        precision = len(correct_citations) / len(citations) if citations else 0.0

        # Calculate recall (documents that are cited)
        cited_docs = len(set(citations).intersection(set(doc_ids)))
        recall = cited_docs / len(doc_ids) if doc_ids else 0.0

        return precision, recall

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from text."""
        # Simple sentence splitting
        sentences = text.split('. ')
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _claim_supported(self, claim: str, documents: List[Dict[str, Any]]) -> bool:
        """Check if claim is supported by any document."""
        claim_lower = claim.lower()

        for doc in documents:
            content = doc.get("content", "").lower()
            # Simple containment check
            if claim_lower in content:
                return True

        return False

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citation IDs from text."""
        import re

        # Look for patterns like [1], [doc_123], etc.
        patterns = [
            r'\[([A-Za-z0-9_-]+)\]',  # [id]
            r'\(([A-Za-z0-9_-]+)\)',  # (id)
            r'source[:\s]+([A-Za-z0-9_-]+)',  # source: id
        ]

        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        return list(set(citations))

    async def _check_thresholds(self, evaluation: RAGEvaluation):
        """Check if metrics fall below thresholds."""
        alerts = []

        for metric_name, threshold in self.thresholds.items():
            metric_value = getattr(evaluation, metric_name, None)

            if metric_value is not None and metric_value < threshold:
                alerts.append({
                    "metric": metric_name,
                    "value": metric_value,
                    "threshold": threshold,
                    "query": evaluation.query[:100]
                })

        if alerts:
            logger.warning(f"RAG metrics below thresholds: {alerts}")

            # Store alert
            self.metrics_history.append({
                "timestamp": datetime.now(),
                "type": "threshold_alert",
                "alerts": alerts,
                "evaluation": evaluation.to_dict()
            })

    async def get_aggregated_metrics(self,
                                     time_window_hours: int = 24) -> Dict[str, Any]:
        """Get aggregated metrics over time window."""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)

        recent_evaluations = [
            e for e in self.evaluations
            if e.evaluation_time and e.evaluation_time > cutoff
        ]

        if not recent_evaluations:
            return {"error": "No evaluations in time window"}

        metrics_summary = {}

        # Aggregate each metric
        metric_fields = [
            "faithfulness", "answer_relevance", "context_relevance",
            "context_recall", "context_precision", "hallucination_score",
            "citation_precision", "citation_recall"
        ]

        for field in metric_fields:
            values = [
                getattr(e, field) for e in recent_evaluations
                if getattr(e, field) is not None
            ]

            if values:
                metrics_summary[field] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }

        # Overall statistics
        metrics_summary["overall"] = {
            "total_evaluations": len(recent_evaluations),
            "time_window_hours": time_window_hours,
            "evaluations_with_ground_truth": sum(
                1 for e in recent_evaluations if e.ground_truth
            ),
            "average_retrieved_docs": np.mean([
                len(e.retrieved_documents) for e in recent_evaluations
            ]) if recent_evaluations else 0
        }

        # Threshold compliance
        threshold_compliance = {}
        for metric, threshold in self.thresholds.items():
            if metric in metrics_summary:
                below_threshold = sum(
                    1 for e in recent_evaluations
                    if getattr(e, metric, 1.0) is not None
                    and getattr(e, metric) < threshold
                )

                threshold_compliance[metric] = {
                    "threshold": threshold,
                    "below_threshold": below_threshold,
                    "compliance_rate": 1 - (below_threshold / len(recent_evaluations))
                }

        metrics_summary["threshold_compliance"] = threshold_compliance

        return metrics_summary

    async def export_evaluations(self,
                                 format: str = "json",
                                 limit: int = 1000) -> str:
        """Export evaluations in specified format."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_evaluations": len(self.evaluations),
            "evaluations": [e.to_dict() for e in self.evaluations[-limit:]]
        }

        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            import io

            output = io.StringIO()
            fieldnames = [
                "timestamp", "query", "faithfulness", "answer_relevance",
                "context_relevance", "hallucination_score", "retrieved_count"
            ]

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for eval_data in export_data["evaluations"]:
                writer.writerow({
                    "timestamp": eval_data.get("evaluation_time"),
                    "query": eval_data.get("query", "")[:100],
                    "faithfulness": eval_data.get("faithfulness"),
                    "answer_relevance": eval_data.get("answer_relevance"),
                    "context_relevance": eval_data.get("context_relevance"),
                    "hallucination_score": eval_data.get("hallucination_score"),
                    "retrieved_count": eval_data.get("retrieved_count")
                })

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def reset(self):
        """Reset all collected metrics."""
        self.evaluations = []
        self.metrics_history = []
        logger.info("RAG metrics collector reset")

    async def save_state(self, filepath: str):
        """Save current state to file."""
        try:
            state = {
                "evaluations": [e.to_dict() for e in self.evaluations],
                "metrics_history": self.metrics_history,
                "saved_at": datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"RAG metrics state saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save RAG metrics state: {e}")

    async def load_state(self, filepath: str):
        """Load state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Recreate evaluations
            self.evaluations = []
            for eval_data in state.get("evaluations", []):
                evaluation = RAGEvaluation(
                    query=eval_data.get("query"),
                    response=eval_data.get("response"),
                    retrieved_documents=[],
                    ground_truth=None
                )

                # Set metrics
                for field in ["faithfulness", "answer_relevance", "context_relevance",
                              "context_recall", "context_precision", "hallucination_score",
                              "citation_precision", "citation_recall"]:
                    if field in eval_data:
                        setattr(evaluation, field, eval_data[field])

                if eval_data.get("evaluation_time"):
                    evaluation.evaluation_time = datetime.fromisoformat(
                        eval_data["evaluation_time"]
                    )

                self.evaluations.append(evaluation)

            self.metrics_history = state.get("metrics_history", [])

            logger.info(f"RAG metrics state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load RAG metrics state: {e}")


# Global instance
rag_metrics_collector = RAGMetricsCollector()
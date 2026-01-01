import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass

from app.rag.retriever import HybridRetriever
from app.rag.reranker import CrossEncoderReranker
from app.models.openai_client import OpenAIClient
from app.utils.logger import get_logger
from app.utils.config import settings


#author: Noble Eselase Vulley
#version: 1.0.0


logger = get_logger(__name__)


@dataclass
class RetrievedDocument:
    """Structure for retrieved policy documents."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    section: Optional[str] = None


@dataclass
class PolicyContext:
    """Context for policy analysis."""
    query: str
    documents: List[RetrievedDocument]
    relevant_sections: List[str]
    confidence: float
    gaps: List[str]


class PolicyRAGAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.openai_client = OpenAIClient()

        self.system_prompt = """You are a Policy Retrieval Agent.

        Task:
        - Retrieve relevant compliance and fraud policy sections
        - Use only retrieved documents
        - Do NOT infer beyond source text
        - Cite document ID and section
        - If insufficient context, return "INSUFFICIENT_POLICY_CONTEXT"

        Constraints:
        - Always cite sources with format: [doc_id:section]
        - Be precise and factual
        - Return both answer and citations

        Query: {policy_question}

        Retrieved Context:
        {retrieved_chunks}"""

        # Policy types for filtering
        self.policy_types = ['AML', 'KYC', 'Fraud', 'Compliance', 'Risk', 'Monitoring']

    async def search(self, query: str, limit: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Search for relevant policies."""
        try:
            logger.info(f"Searching policies for: {query[:100]}...")

            # Step 1: Retrieve relevant documents
            retrieved = await self.retrieve_documents(query, limit * 2, filters)

            if not retrieved:
                return self._create_no_results_response(query)

            # Step 2: Rerank for relevance
            reranked = await self.rerank_documents(query, retrieved, limit)

            # Step 3: Generate grounded response
            response = await self.generate_grounded_response(query, reranked)

            # Step 4: Check confidence
            if response.get("confidence", 0) < settings.CONFIDENCE_THRESHOLD:
                response["insufficient_context"] = True
                response["answer"] = "INSUFFICIENT_POLICY_CONTEXT"

            # Add metadata
            response.update({
                "query": query,
                "retrieval_count": len(retrieved),
                "reranked_count": len(reranked),
                "filters_applied": filters or {},
                "timestamp": asyncio.get_event_loop().time()
            })

            logger.info(f"Policy search completed: found {len(reranked)} relevant documents")
            return response

        except Exception as e:
            logger.error(f"Policy search failed: {e}", exc_info=True)
            return self._create_error_response(query, str(e))

    async def retrieve_documents(self, query: str, limit: int,
                                 filters: Optional[Dict] = None) -> List[RetrievedDocument]:
        """Retrieve policy documents using hybrid search."""
        try:
            # Combine vector and keyword search
            vector_results = await self.retriever.vector_search(
                query=query,
                limit=limit,
                filters=filters,
                # score_threshold=0.7
            )

            keyword_results = await self.retriever.keyword_search(
                query=query,
                limit=limit,
                filters=filters
            )

            # Combine and deduplicate
            all_results = self._combine_results(vector_results, keyword_results)

            # Sort by score
            all_results.sort(key=lambda x: x.score, reverse=True)

            # Take top results
            return all_results[:limit * 2]  # Get extra for reranking

        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []

    async def rerank_documents(self, query: str, documents: List[RetrievedDocument],
                               limit: int) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []

        try:
            # Prepare documents for reranking
            doc_texts = [doc.content for doc in documents]

            # Get reranking scores
            scores = await self.reranker.rerank(query, doc_texts)

            # Apply scores to documents
            for doc, score in zip(documents, scores):
                doc.score = (doc.score * 0.3 + score * 0.7)  # Weighted combination

            # Resort by new scores
            documents.sort(key=lambda x: x.score, reverse=True)

            return documents[:limit]

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return documents[:limit]

    async def generate_grounded_response(self, query: str,
                                         documents: List[RetrievedDocument]) -> Dict[str, Any]:
        """Generate response grounded in retrieved documents."""
        if not documents:
            return {
                "answer": "No relevant policies found.",
                "sources": [],
                "confidence": 0.0,
                "document_count": 0
            }

        # Prepare context for LLM
        context = self._prepare_context(documents)

        # Generate response with OpenAI
        messages = [
            {"role": "system", "content": self.system_prompt.format(
                policy_question=query,
                retrieved_chunks=context
            )},
            {"role": "user", "content": query}
        ]

        response = await self.openai_client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )

        # Extract answer and citations
        answer, citations = self._parse_response(response, documents)

        # Calculate confidence
        confidence = self._calculate_confidence(documents, citations)

        return {
            "answer": answer,
            "sources": citations,
            "confidence": confidence,
            "document_count": len(documents),
            "top_documents": [self._format_document(doc) for doc in documents[:3]]
        }

    def _combine_results(self, vector_results: List, keyword_results: List) -> List[RetrievedDocument]:
        """Combine and deduplicate search results."""
        seen_ids = set()
        combined = []

        # Add vector results first (higher precision)
        for result in vector_results:
            if result.id not in seen_ids:
                combined.append(RetrievedDocument(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    score=result.score,
                    source="vector"
                ))
                seen_ids.add(result.id)

        # Add keyword results
        for result in keyword_results:
            if result.id not in seen_ids:
                combined.append(RetrievedDocument(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    score=result.score * 0.8,  # Slightly lower weight for keyword
                    source="keyword"
                ))
                seen_ids.add(result.id)

        return combined

    def _prepare_context(self, documents: List[RetrievedDocument]) -> str:
        """Prepare context string for LLM."""
        context_parts = []

        for i, doc in enumerate(documents[:10]):  # Limit context
            metadata_str = json.dumps(doc.metadata, default=str)
            context_parts.append(
                f"[Document {i + 1} - ID: {doc.id}]\n"
                f"Metadata: {metadata_str}\n"
                f"Content: {doc.content[:500]}...\n"
                f"Score: {doc.score:.3f}\n"
                f"Source: {doc.source}\n"
                f"{'-' * 50}"
            )

        return "\n".join(context_parts)

    def _parse_response(self, response: str, documents: List[RetrievedDocument]) -> Tuple[str, List[Dict]]:
        """Parse LLM response to extract answer and citations."""
        # Extract citations
        import re

        citation_pattern = r'\[([A-Za-z0-9_-]+):([A-Za-z0-9._-]+)\]'
        citations_found = re.findall(citation_pattern, response)

        # Map to document information
        citations = []
        for doc_id, section in citations_found:
            doc = next((d for d in documents if d.id == doc_id), None)
            if doc:
                citations.append({
                    "document_id": doc_id,
                    "section": section,
                    "title": doc.metadata.get("title", "Unknown"),
                    "policy_type": doc.metadata.get("policy_type", "Unknown"),
                    "relevance_score": doc.score
                })

        # Clean answer (remove citation markers for display)
        clean_answer = re.sub(citation_pattern, '', response).strip()

        return clean_answer, citations

    def _calculate_confidence(self, documents: List[RetrievedDocument],
                              citations: List[Dict]) -> float:
        """Calculate confidence based on document relevance and citation coverage."""
        if not documents or not citations:
            return 0.0

        # Average score of cited documents
        cited_scores = []
        for citation in citations:
            doc = next((d for d in documents if d.id == citation["document_id"]), None)
            if doc:
                cited_scores.append(doc.score)

        if not cited_scores:
            return 0.0

        avg_score = sum(cited_scores) / len(cited_scores)

        # Coverage ratio
        coverage = len(cited_scores) / min(len(documents), 5)

        # Combined confidence
        confidence = (avg_score * 0.6 + min(coverage, 1.0) * 0.4)

        return min(confidence, 1.0)

    def _format_document(self, doc: RetrievedDocument) -> Dict[str, Any]:
        """Format document for response."""
        return {
            "id": doc.id,
            "title": doc.metadata.get("title", "Unknown"),
            "policy_type": doc.metadata.get("policy_type", "Unknown"),
            "region": doc.metadata.get("region", "Global"),
            "version": doc.metadata.get("version", "1.0"),
            "score": doc.score,
            "source": doc.source,
            "excerpt": doc.content[:200] + "..."
        }

    def _create_no_results_response(self, query: str) -> Dict[str, Any]:
        return {
            "answer": "No relevant policies found for the query.",
            "sources": [],
            "confidence": 0.0,
            "document_count": 0,
            "query": query,
            "suggestion": "Try broadening your search terms or check policy categories."
        }

    def _create_error_response(self, query: str, error: str) -> Dict[str, Any]:
        return {
            "answer": f"Error searching policies: {error}",
            "sources": [],
            "confidence": 0.0,
            "document_count": 0,
            "query": query,
            "error": True,
            "error_details": error
        }

    async def get_policy_by_id(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get specific policy by ID."""
        try:
            document = await self.retriever.get_document_by_id(policy_id)
            if document:
                return self._format_document(document)
            return None
        except Exception as e:
            logger.error(f"Failed to get policy {policy_id}: {e}")
            return None

    async def list_policies(self, policy_type: Optional[str] = None,
                            region: Optional[str] = None) -> List[Dict[str, Any]]:
        """List policies with filtering."""
        try:
            filters = {}
            if policy_type:
                filters["policy_type"] = policy_type
            if region:
                filters["region"] = region

            documents = await self.retriever.list_documents(filters=filters, limit=50)

            return [self._format_document(doc) for doc in documents]

        except Exception as e:
            logger.error(f"Failed to list policies: {e}")
            return []

    async def add_policy(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add a new policy document."""
        try:
            # Validate required metadata
            required_fields = ["title", "policy_type", "region", "version"]
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required field: {field}")

            # Add to vector store
            doc_id = await self.retriever.add_document(content, metadata)

            logger.info(f"Added policy: {metadata.get('title')} (ID: {doc_id})")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add policy: {e}")
            raise

    async def update_policy(self, policy_id: str, content: Optional[str] = None,
                            metadata: Optional[Dict] = None) -> bool:
        """Update an existing policy."""
        try:
            success = await self.retriever.update_document(policy_id, content, metadata)

            if success:
                logger.info(f"Updated policy: {policy_id}")
            else:
                logger.warning(f"Policy not found: {policy_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to update policy {policy_id}: {e}")
            return False

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy document."""
        try:
            success = await self.retriever.delete_document(policy_id)

            if success:
                logger.info(f"Deleted policy: {policy_id}")
            else:
                logger.warning(f"Policy not found: {policy_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete policy {policy_id}: {e}")
            return False
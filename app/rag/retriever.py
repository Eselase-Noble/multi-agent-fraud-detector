import json
from typing import List, Dict, Any, Optional, Tuple, Union
import asyncio
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.embeddings import EmbeddingManager
from app.db.vector_store import VectorStore
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str
    embedding: Optional[List[float]] = None


class HybridRetriever:
    """Production-grade hybrid retriever with vector + keyword search."""

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.bm25_index = None
        self.documents = []
        self.document_map = {}  # id -> document

        # Retrieval weights
        self.weights = {
            "vector": 0.7,
            "keyword": 0.3,
            "recency": 0.1,
            "relevance": 0.9
        }

        # Cache for frequent queries
        self.query_cache = {}
        self.cache_size = 1000

    async def initialize(self):
        """Initialize retriever with existing documents."""
        try:
            # Load documents from vector store
            self.documents = await self.vector_store.get_all_documents()

            # Build BM25 index
            await self._build_bm25_index()

            logger.info(f"Retriever initialized with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    async def search(self,
                     query: str,
                     limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None,
                     hybrid: bool = True) -> List[RetrievalResult]:
        """Search for relevant documents."""
        logger.info(f"Searching for: {query[:100]}...")

        # Check cache
        cache_key = self._get_cache_key(query, filters, limit)
        if cache_key in self.query_cache:
            logger.debug("Cache hit for query")
            return self.query_cache[cache_key]

        try:
            if hybrid:
                results = await self._hybrid_search(query, limit, filters)
            else:
                results = await self._vector_search(query, limit, filters)

            # Apply post-processing
            results = await self._post_process_results(results, query, filters)

            # Cache results
            self._update_cache(cache_key, results)

            logger.info(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    async def _hybrid_search(self,
                             query: str,
                             limit: int,
                             filters: Optional[Dict]) -> List[RetrievalResult]:
        """Perform hybrid vector + keyword search."""
        # Run searches in parallel
        vector_task = self._vector_search(query, limit * 2, filters)
        keyword_task = self._keyword_search(query, limit * 2, filters)

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )

        # Combine and rerank
        combined = await self._combine_results(
            vector_results, keyword_results, query
        )

        # Return top results
        return combined[:limit]

    async def _vector_search(self,
                             query: str,
                             limit: int,
                             filters: Optional[Dict]) -> List[RetrievalResult]:
        """Perform vector similarity search."""
        try:
            # Get query embedding
            embedding_result = await self.embedding_manager.embed_text(query)
            query_embedding = embedding_result.embedding

            # Search vector store
            vector_results = await self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit,
                filters=filters,
                score_threshold=0.7
            )

            # Convert to RetrievalResult format
            results = []
            for doc, score in vector_results:
                results.append(RetrievalResult(
                    id=doc.get("id", ""),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=score,
                    retrieval_method="vector",
                    embedding=doc.get("embedding")
                ))

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _keyword_search(self,
                              query: str,
                              limit: int,
                              filters: Optional[Dict]) -> List[RetrievalResult]:
        """Perform keyword search using BM25."""
        if not self.bm25_index or not self.documents:
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top documents
            top_indices = np.argsort(scores)[::-1][:limit]

            # Apply filters
            filtered_results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]

                    # Apply filters if provided
                    if filters and not self._apply_filters(doc, filters):
                        continue

                    # Normalize BM25 score to 0-1 range
                    normalized_score = scores[idx] / (scores[idx] + 1.0)

                    filtered_results.append(RetrievalResult(
                        id=doc.get("id", f"doc_{idx}"),
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        score=normalized_score,
                        retrieval_method="keyword"
                    ))

            return filtered_results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    async def _combine_results(self,
                               vector_results: List[RetrievalResult],
                               keyword_results: List[RetrievalResult],
                               query: str) -> List[RetrievalResult]:
        """Combine and rerank results from both methods."""
        # Create dictionary of all results by ID
        all_results = {}

        # Add vector results
        for result in vector_results:
            all_results[result.id] = {
                "result": result,
                "vector_score": result.score * self.weights["vector"]
            }

        # Add keyword results
        for result in keyword_results:
            if result.id in all_results:
                # Update existing result
                all_results[result.id]["keyword_score"] = result.score * self.weights["keyword"]
            else:
                all_results[result.id] = {
                    "result": result,
                    "keyword_score": result.score * self.weights["keyword"]
                }

        # Calculate combined scores
        combined_results = []
        for data in all_results.values():
            result = data["result"]

            # Calculate weighted score
            vector_score = data.get("vector_score", 0)
            keyword_score = data.get("keyword_score", 0)

            # Apply additional scoring factors
            relevance_score = self._calculate_relevance_score(result, query)
            recency_score = self._calculate_recency_score(result)

            combined_score = (
                    vector_score + keyword_score +
                    relevance_score * self.weights["relevance"] +
                    recency_score * self.weights["recency"]
            )

            # Update result with combined score
            result.score = combined_score
            combined_results.append(result)

        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results

    async def _post_process_results(self,
                                    results: List[RetrievalResult],
                                    query: str,
                                    filters: Optional[Dict]) -> List[RetrievalResult]:
        """Post-process results for quality."""
        if not results:
            return results

        processed = []

        for result in results:
            # Apply confidence threshold
            if result.score < 0.3:
                continue

            # Check for duplicates (similar content)
            if self._is_duplicate(result, processed):
                continue

            # Enhance metadata
            result.metadata = self._enhance_metadata(result.metadata, query)

            processed.append(result)

        # Deduplicate by content (semantic deduplication)
        processed = await self._semantic_deduplication(processed)

        # Ensure diversity in results
        processed = self._ensure_diversity(processed)

        return processed

    def _calculate_relevance_score(self, result: RetrievalResult, query: str) -> float:
        """Calculate additional relevance score."""
        # Check for query terms in content
        content_lower = result.content.lower()
        query_terms = query.lower().split()

        # Count term matches
        term_matches = sum(1 for term in query_terms if term in content_lower)

        # Normalize by query length
        if query_terms:
            return min(term_matches / len(query_terms), 1.0)

        return 0.0

    def _calculate_recency_score(self, result: RetrievalResult) -> float:
        """Calculate recency score based on metadata."""
        metadata = result.metadata

        # Check for date in metadata
        date_str = metadata.get("date") or metadata.get("created_at") or metadata.get("timestamp")

        if date_str:
            try:
                if isinstance(date_str, str):
                    doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    doc_date = date_str

                # Calculate recency (more recent = higher score)
                days_old = (datetime.now() - doc_date).days
                recency_score = max(0, 1.0 - (days_old / 365))  # Decay over year

                return recency_score
            except Exception:
                pass

        return 0.5  # Default score for undated documents

    def _is_duplicate(self, result: RetrievalResult, existing: List[RetrievalResult]) -> bool:
        """Check if result is duplicate of existing results."""
        # Simple content-based deduplication
        result_content = result.content[:200]  # First 200 chars

        for existing_result in existing:
            existing_content = existing_result.content[:200]

            # Check for high overlap
            if result_content == existing_content:
                return True

            # Check for very similar scores and metadata
            if (abs(result.score - existing_result.score) < 0.05 and
                    result.metadata.get("source") == existing_result.metadata.get("source")):
                return True

        return False

    async def _semantic_deduplication(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove semantically similar results."""
        if len(results) <= 1:
            return results

        # Use embeddings for semantic deduplication
        embeddings = []
        valid_results = []

        for result in results:
            if result.embedding:
                embeddings.append(result.embedding)
                valid_results.append(result)
            else:
                # Generate embedding if missing
                try:
                    embedding_result = await self.embedding_manager.embed_text(
                        result.content[:1000]  # Limit for efficiency
                    )
                    embeddings.append(embedding_result.embedding)
                    result.embedding = embedding_result.embedding
                    valid_results.append(result)
                except Exception:
                    # Skip if embedding fails
                    continue

        if len(valid_results) <= 1:
            return results

        # Calculate similarity matrix
        embedding_matrix = np.array(embeddings)
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        # Remove duplicates
        deduplicated = []
        used_indices = set()

        for i in range(len(valid_results)):
            if i in used_indices:
                continue

            deduplicated.append(valid_results[i])
            used_indices.add(i)

            # Find similar results to exclude
            for j in range(i + 1, len(valid_results)):
                if j not in used_indices and similarity_matrix[i, j] > 0.9:
                    used_indices.add(j)

        return deduplicated

    def _ensure_diversity(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Ensure diversity in results."""
        if len(results) <= 3:
            return results

        # Group by source/metadata to ensure diversity
        grouped = {}
        for result in results:
            source = result.metadata.get("source", "unknown")
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(result)

        # Take top result from each group, then second, etc.
        diverse_results = []
        max_per_group = 2  # Maximum results per source

        for i in range(max_per_group):
            for source, group_results in grouped.items():
                if i < len(group_results):
                    diverse_results.append(group_results[i])

                if len(diverse_results) >= 10:  # Limit total results
                    break

            if len(diverse_results) >= 10:
                break

        return diverse_results

    def _enhance_metadata(self, metadata: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Enhance metadata with additional information."""
        enhanced = metadata.copy()

        # Add retrieval timestamp
        enhanced["retrieved_at"] = datetime.now().isoformat()

        # Add query context
        enhanced["query_context"] = query[:100]

        # Calculate content statistics
        content = metadata.get("content", "")
        if content:
            enhanced["content_length"] = len(content)
            enhanced["word_count"] = len(content.split())

        return enhanced

    async def _build_bm25_index(self):
        """Build BM25 index from documents."""
        if not self.documents:
            return

        # Extract text content
        texts = []
        for doc in self.documents:
            content = doc.get("content", "")
            if content:
                tokens = self._tokenize(content)
                texts.append(tokens)
            else:
                texts.append([])  # Empty document

        # Build BM25 index
        self.bm25_index = BM25Okapi(texts)
        logger.info(f"Built BM25 index with {len(texts)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # Convert to lowercase and split
        tokens = text.lower().split()

        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]

        return tokens

    def _apply_filters(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to document."""
        metadata = document.get("metadata", {})

        for key, value in filters.items():
            if key in metadata:
                if isinstance(value, list):
                    # Check if document value is in filter list
                    if metadata[key] not in value:
                        return False
                else:
                    # Exact match
                    if metadata[key] != value:
                        return False

        return True

    def _get_cache_key(self, query: str, filters: Optional[Dict], limit: int) -> str:
        """Generate cache key for query."""
        import hashlib

        key_parts = [query, str(limit)]

        if filters:
            # Sort filters for consistent key
            sorted_filters = json.dumps(filters, sort_keys=True)
            key_parts.append(sorted_filters)

        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _update_cache(self, key: str, results: List[RetrievalResult]):
        """Update query cache."""
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[key] = results

    async def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add document to retriever."""
        try:
            # Add to vector store
            doc_id = await self.vector_store.add_document(content, metadata)

            # Add to local collections
            document = {
                "id": doc_id,
                "content": content,
                "metadata": metadata
            }

            self.documents.append(document)
            self.document_map[doc_id] = document

            # Rebuild BM25 index
            await self._build_bm25_index()

            # Clear cache (since documents changed)
            self.query_cache.clear()

            logger.info(f"Added document {doc_id} to retriever")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from retriever."""
        try:
            # Remove from vector store
            success = await self.vector_store.delete_document(doc_id)

            if success:
                # Remove from local collections
                self.documents = [doc for doc in self.documents if doc.get("id") != doc_id]
                self.document_map.pop(doc_id, None)

                # Rebuild BM25 index
                await self._build_bm25_index()

                # Clear cache
                self.query_cache.clear()

                logger.info(f"Deleted document {doc_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def get_document_by_id(self, doc_id: str) -> Optional[RetrievalResult]:
        """Get document by ID."""
        if doc_id in self.document_map:
            doc = self.document_map[doc_id]
            return RetrievalResult(
                id=doc_id,
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=1.0,
                retrieval_method="direct"
            )

        # Try vector store
        doc = await self.vector_store.get_document(doc_id)
        if doc:
            return RetrievalResult(
                id=doc_id,
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=1.0,
                retrieval_method="direct"
            )

        return None

    async def list_documents(self,
                             filters: Optional[Dict] = None,
                             limit: int = 50) -> List[RetrievalResult]:
        """List documents with optional filtering."""
        results = []

        for doc in self.documents[:limit]:
            if filters and not self._apply_filters(doc, filters):
                continue

            results.append(RetrievalResult(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=1.0,
                retrieval_method="list"
            ))

        return results

    async def vector_search(self, query: str, limit: int, filters: Optional[Dict] = None):
        return await self._vector_search(query, limit, filters)

    async def keyword_search(self, query: str, limit: int, filters: Optional[Dict] = None):
        return await self._keyword_search(query, limit, filters)
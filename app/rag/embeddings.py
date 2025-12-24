import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
from dataclasses import dataclass
import json
import hashlib
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import openai

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    token_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingManager:
    """Production-grade embedding management with multiple providers."""

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.local_models = {}
        self.embedding_cache = {}

        # Model configurations
        self.models = {
            "openai": {
                "name": "text-embedding-3-small",
                "dimensions": 1536,
                "max_tokens": 8191,
                "rate_limit": 1000  # requests per minute
            },
            "local": {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "max_tokens": 256,
                "rate_limit": None
            }
        }

    async def embed_text(self,
                         text: Union[str, List[str]],
                         model_type: str = "openai",
                         metadata: Optional[Dict] = None) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Embed text using specified model."""
        logger.info(f"Embedding text with {model_type} model")

        # Check cache first
        cache_key = self._get_cache_key(text, model_type)
        if cache_key in self.embedding_cache:
            logger.debug("Cache hit for embedding")
            return self.embedding_cache[cache_key]

        try:
            if isinstance(text, list):
                return await self._embed_batch(text, model_type, metadata)
            else:
                return await self._embed_single(text, model_type, metadata)

        except Exception as e:
            logger.error(f"Embedding failed: {e}", exc_info=True)
            raise

    async def _embed_single(self, text: str, model_type: str, metadata: Optional[Dict]) -> EmbeddingResult:
        """Embed single text."""
        # Pre-process text
        processed_text = self._preprocess_text(text)

        if model_type == "openai":
            result = await self._openai_embedding(processed_text)
        else:
            result = await self._local_embedding(processed_text, model_type)

        # Add metadata
        if metadata:
            result.metadata = metadata

        # Cache result
        cache_key = self._get_cache_key(text, model_type)
        self.embedding_cache[cache_key] = result

        return result

    async def _embed_batch(self,
                           texts: List[str],
                           model_type: str,
                           metadata: Optional[Dict]) -> List[EmbeddingResult]:
        """Embed batch of texts efficiently."""
        logger.info(f"Embedding batch of {len(texts)} texts")

        # Check cache for each text
        results = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, model_type)
            if cache_key in self.embedding_cache:
                results.append(self.embedding_cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                results.append(None)  # Placeholder

        if not texts_to_embed:
            return results

        # Process batch
        try:
            if model_type == "openai":
                batch_results = await self._openai_batch_embedding(texts_to_embed)
            else:
                batch_results = await self._local_batch_embedding(texts_to_embed, model_type)

            # Update results and cache
            for idx, batch_result in zip(indices_to_embed, batch_results):
                # Add metadata if provided
                if metadata:
                    batch_result.metadata = metadata.copy()

                results[idx] = batch_result

                # Cache
                cache_key = self._get_cache_key(texts[idx], model_type)
                self.embedding_cache[cache_key] = batch_result

            return results

        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fallback to sequential embedding
            return await self._fallback_embedding(texts, model_type, metadata, results, indices_to_embed)

    async def _openai_embedding(self, text: str) -> EmbeddingResult:
        """Get embedding from OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.models["openai"]["name"],
                input=text,
                encoding_format="float"
            )

            return EmbeddingResult(
                text=text,
                embedding=response.data[0].embedding,
                model=self.models["openai"]["name"],
                dimensions=self.models["openai"]["dimensions"],
                token_count=response.usage.total_tokens
            )

        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit, waiting...")
            await asyncio.sleep(60)  # Wait 1 minute
            return await self._openai_embedding(text)  # Retry

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    async def _openai_batch_embedding(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embedding with OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.models["openai"]["name"],
                input=texts,
                encoding_format="float"
            )

            results = []
            for i, data in enumerate(response.data):
                results.append(EmbeddingResult(
                    text=texts[i],
                    embedding=data.embedding,
                    model=self.models["openai"]["name"],
                    dimensions=self.models["openai"]["dimensions"],
                    token_count=response.usage.total_tokens // len(texts)
                ))

            return results

        except openai.RateLimitError:
            logger.warning("OpenAI batch rate limit hit, splitting batch...")
            # Split batch and retry
            half = len(texts) // 2
            batch1 = await self._openai_batch_embedding(texts[:half])
            batch2 = await self._openai_batch_embedding(texts[half:])
            return batch1 + batch2

        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise

    async def _local_embedding(self, text: str, model_type: str) -> EmbeddingResult:
        """Get embedding from local model."""
        model = await self._get_local_model(model_type)

        try:
            embedding = model.encode(text, convert_to_numpy=True).tolist()

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=model_type,
                dimensions=self.models["local"]["dimensions"]
            )

        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise

    async def _local_batch_embedding(self, texts: List[str], model_type: str) -> List[EmbeddingResult]:
        """Batch embedding with local model."""
        model = await self._get_local_model(model_type)

        try:
            embeddings = model.encode(texts, convert_to_numpy=True)

            results = []
            for i, embedding in enumerate(embeddings):
                results.append(EmbeddingResult(
                    text=texts[i],
                    embedding=embedding.tolist(),
                    model=model_type,
                    dimensions=self.models["local"]["dimensions"]
                ))

            return results

        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")
            raise

    async def _get_local_model(self, model_type: str):
        """Get or load local embedding model."""
        if model_type not in self.local_models:
            logger.info(f"Loading local model: {model_type}")

            if model_type == "default":
                model_name = self.models["local"]["name"]
            else:
                model_name = model_type

            self.local_models[model_type] = SentenceTransformer(model_name)

        return self.local_models[model_type]

    async def _fallback_embedding(self,
                                  texts: List[str],
                                  model_type: str,
                                  metadata: Optional[Dict],
                                  current_results: List[Optional[EmbeddingResult]],
                                  failed_indices: List[int]) -> List[EmbeddingResult]:
        """Fallback embedding when batch fails."""
        logger.warning("Using fallback sequential embedding")

        for idx in failed_indices:
            try:
                result = await self._embed_single(texts[idx], model_type, metadata)
                current_results[idx] = result

                # Add small delay to avoid rate limits
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to embed text at index {idx}: {e}")
                # Create error placeholder
                current_results[idx] = EmbeddingResult(
                    text=texts[idx],
                    embedding=[0.0] * self.models[model_type]["dimensions"],
                    model=model_type,
                    dimensions=self.models[model_type]["dimensions"],
                    metadata={"error": str(e)}
                )

        return current_results

    def _preprocess_text(self, text: str) -> str:
        """Pre-process text before embedding."""
        # Truncate if too long
        max_tokens = self.models["openai"]["max_tokens"]
        if len(text.split()) > max_tokens:
            logger.warning(f"Text too long ({len(text.split())} tokens), truncating...")
            words = text.split()[:max_tokens]
            text = ' '.join(words)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _get_cache_key(self, text: str, model_type: str) -> str:
        """Generate cache key for text embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{model_type}_{text_hash}"

    async def similarity_search(self,
                                query_embedding: List[float],
                                document_embeddings: List[List[float]],
                                top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar embeddings."""
        if not document_embeddings:
            return []

        # Convert to numpy arrays
        query_vec = np.array(query_embedding).reshape(1, -1)
        doc_matrix = np.array(document_embeddings)

        # Calculate cosine similarity
        similarities = np.dot(doc_matrix, query_vec.T).flatten()

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(idx, float(similarities[idx])) for idx in top_indices]

    async def validate_embedding(self, embedding: List[float], expected_dimensions: int) -> bool:
        """Validate embedding quality."""
        if not embedding:
            return False

        # Check dimensions
        if len(embedding) != expected_dimensions:
            logger.error(f"Embedding dimension mismatch: {len(embedding)} != {expected_dimensions}")
            return False

        # Check for NaN or infinity
        embedding_array = np.array(embedding)
        if np.any(np.isnan(embedding_array)):
            logger.error("Embedding contains NaN values")
            return False

        if np.any(np.isinf(embedding_array)):
            logger.error("Embedding contains infinite values")
            return False

        # Check magnitude (should be normalized for cosine similarity)
        magnitude = np.linalg.norm(embedding_array)
        if magnitude < 0.1 or magnitude > 10.0:
            logger.warning(f"Embedding magnitude unusual: {magnitude}")

        return True

    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")

    async def get_model_info(self, model_type: str = "openai") -> Dict[str, Any]:
        """Get information about embedding model."""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        info = self.models[model_type].copy()
        info["cache_size"] = len(self.embedding_cache)
        info["local_models_loaded"] = list(self.local_models.keys())

        return info
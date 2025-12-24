import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.rag.embeddings import EmbeddingManager
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Production-grade vector store manager using ChromaDB."""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_manager = EmbeddingManager()
        self.collection_name = "fraud_policies"

        # Default embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL
        )

    async def initialize(self, collection_name: Optional[str] = None):
        """Initialize vector store connection and collection."""
        try:
            # TEMPORARY: Use in-memory mode (no disk access)
            self.client = chromadb.EphemeralClient()

            logger.info("Using in-memory ChromaDB (no persistence)")

            # Get or create collection
            self.collection_name = collection_name or self.collection_name

            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )

            logger.info(f"Vector store ready: {self.collection.count()} documents")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def add_documents(self,
                            documents: List[Dict[str, Any]],
                            batch_size: int = 100) -> List[str]:
        """Add documents to vector store in batches."""
        if not documents:
            return []

        logger.info(f"Adding {len(documents)} documents to vector store...")

        try:
            document_ids = []

            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Prepare batch data
                ids = []
                texts = []
                metadatas = []

                for doc in batch:
                    # Generate unique ID
                    doc_id = doc.get("id", str(uuid.uuid4()))

                    # Extract content
                    content = doc.get("content", "")
                    if not content:
                        logger.warning(f"Document {doc_id} has no content, skipping")
                        continue

                    # Prepare metadata
                    metadata = doc.get("metadata", {}).copy()
                    metadata.update({
                        "source": doc.get("source", "unknown"),
                        "added_at": datetime.now().isoformat(),
                        "content_length": len(content)
                    })

                    ids.append(doc_id)
                    texts.append(content)
                    metadatas.append(metadata)

                # Add to collection
                if ids:
                    self.collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas
                    )
                    document_ids.extend(ids)

                    logger.debug(f"Added batch {i // batch_size + 1}: {len(ids)} documents")

            logger.info(f"Successfully added {len(document_ids)} documents")
            return document_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    async def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add single document to vector store."""
        document = {
            "id": metadata.get("id", str(uuid.uuid4())),
            "content": content,
            "metadata": metadata
        }

        doc_ids = await self.add_documents([document], batch_size=1)
        return doc_ids[0] if doc_ids else ""

    async def search(self,
                     query_embedding: List[float],
                     limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None,
                     score_threshold: float = 0.7) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents using vector similarity."""
        if not self.collection:
            await self.initialize()

        try:
            # Convert embedding to list if it's numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get extra for filtering
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            matched_documents = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                )):
                    # Convert distance to similarity score (Chroma uses distance, not similarity)
                    similarity = 1.0 - distance  # Assuming cosine distance

                    if similarity >= score_threshold:
                        matched_documents.append((
                            {
                                "id": results["ids"][0][i],
                                "content": doc,
                                "metadata": metadata or {}
                            },
                            similarity
                        ))

            # Sort by similarity and limit
            matched_documents.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Vector search found {len(matched_documents)} matches")
            return matched_documents[:limit]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def search_by_text(self,
                             query_text: str,
                             limit: int = 10,
                             filters: Optional[Dict] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search using text query (auto-embeds the query)."""
        try:
            # Get embedding for query text
            embedding_result = await self.embedding_manager.embed_text(query_text)
            query_embedding = embedding_result.embedding

            # Perform vector search
            return await self.search(
                query_embedding=query_embedding,
                limit=limit,
                filters=filters
            )

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def hybrid_search(self,
                            query: str,
                            limit: int = 10,
                            filters: Optional[Dict] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Hybrid search combining vector and keyword search."""
        # This is a simplified version - full hybrid search would be in retriever.py
        return await self.search_by_text(query, limit, filters)

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        if not self.collection:
            await self.initialize()

        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )

            if results["documents"]:
                return {
                    "id": doc_id,
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {}
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None

    async def update_document(self,
                              doc_id: str,
                              content: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> bool:
        """Update document in vector store."""
        if not self.collection:
            await self.initialize()

        try:
            # Get existing document
            existing = await self.get_document(doc_id)
            if not existing:
                return False

            # Prepare update data
            update_content = content if content is not None else existing["content"]
            update_metadata = existing["metadata"].copy()

            if metadata:
                update_metadata.update(metadata)

            update_metadata["updated_at"] = datetime.now().isoformat()

            # Update document
            self.collection.update(
                ids=[doc_id],
                documents=[update_content],
                metadatas=[update_metadata]
            )

            logger.info(f"Updated document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector store."""
        if not self.collection:
            await self.initialize()

        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def get_all_documents(self,
                                filters: Optional[Dict] = None,
                                limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all documents from vector store."""
        if not self.collection:
            await self.initialize()

        try:
            results = self.collection.get(
                where=filters,
                limit=limit,
                include=["documents", "metadatas"]
            )

            documents = []
            for i, doc_id in enumerate(results["ids"]):
                documents.append({
                    "id": doc_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self.collection:
            await self.initialize()

        try:
            count = self.collection.count()

            # Sample documents to get metadata statistics
            sample_docs = await self.get_all_documents(limit=100)

            # Calculate metadata statistics
            metadata_fields = {}
            for doc in sample_docs:
                for key, value in doc["metadata"].items():
                    if key not in metadata_fields:
                        metadata_fields[key] = {
                            "count": 0,
                            "sample_values": set()
                        }
                    metadata_fields[key]["count"] += 1
                    if len(metadata_fields[key]["sample_values"]) < 5:
                        metadata_fields[key]["sample_values"].add(str(value))

            # Convert sets to lists for JSON serialization
            for key in metadata_fields:
                metadata_fields[key]["sample_values"] = list(metadata_fields[key]["sample_values"])

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "sample_size": len(sample_docs),
                "metadata_fields": metadata_fields,
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "embedding_model": settings.EMBEDDING_MODEL
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    async def create_index(self, index_type: str = "hnsw"):
        """Create index for faster searching."""
        if not self.collection:
            await self.initialize()

        try:
            # Chroma handles indexing automatically
            logger.info(f"Index type '{index_type}' is managed automatically by ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    async def cleanup_old_documents(self, days_old: int = 90) -> int:
        """Clean up documents older than specified days."""
        if not self.collection:
            await self.initialize()

        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

            # Get all documents with timestamps
            all_docs = await self.get_all_documents()

            # Identify old documents
            old_doc_ids = []
            for doc in all_docs:
                metadata = doc.get("metadata", {})
                added_at = metadata.get("added_at")

                if added_at:
                    try:
                        doc_date = datetime.fromisoformat(added_at.replace('Z', '+00:00'))
                        if doc_date.timestamp() < cutoff_date:
                            old_doc_ids.append(doc["id"])
                    except:
                        pass

            # Delete old documents
            if old_doc_ids:
                self.collection.delete(ids=old_doc_ids)
                logger.info(f"Cleaned up {len(old_doc_ids)} old documents")

            return len(old_doc_ids)

        except Exception as e:
            logger.error(f"Failed to cleanup old documents: {e}")
            return 0

    async def backup_collection(self, backup_path: str) -> bool:
        """Create backup of collection."""
        if not self.collection:
            await self.initialize()

        try:
            # Get all documents
            all_docs = await self.get_all_documents(limit=10000)

            # Prepare backup data
            backup_data = {
                "collection_name": self.collection_name,
                "backup_timestamp": datetime.now().isoformat(),
                "document_count": len(all_docs),
                "documents": all_docs
            }

            # Save to file
            import os
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)

            logger.info(f"Created backup at {backup_path} with {len(all_docs)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            return False

    async def restore_from_backup(self, backup_path: str) -> bool:
        """Restore collection from backup."""
        try:
            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)

            # Clear existing collection
            await self.delete_collection()

            # Reinitialize collection
            await self.initialize(collection_name=backup_data["collection_name"])

            # Add documents from backup
            documents = []
            for doc in backup_data["documents"]:
                documents.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })

            await self.add_documents(documents)

            logger.info(f"Restored collection from {backup_path} with {len(documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    async def delete_collection(self) -> bool:
        """Delete the entire collection."""
        if not self.client:
            await self.initialize()

        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            logger.info(f"Deleted collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False


# Global instance
vector_store = VectorStore()


async def init_vector_store():
    """Initialize vector store."""
    await vector_store.initialize()


async def get_vector_store():
    """Get vector store instance."""
    if not vector_store.collection:
        await vector_store.initialize()
    return vector_store
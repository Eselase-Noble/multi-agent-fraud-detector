import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from enum import Enum
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingMethod(str, Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    FIXED = "fixed"
    RECURSIVE = "recursive"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    chunk_id: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    section: Optional[str] = None
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata,
            "section": self.section,
            "parent_id": self.parent_id,
            "text_hash": hashlib.md5(self.text.encode()).hexdigest()[:16]
        }


class DocumentChunker:
    """Production-grade document chunking with multiple strategies."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Structural patterns for different document types
        self.patterns = {
            "markdown": [
                (r'#{1,6}\s+.+', 'heading'),  # Markdown headings
                (r'\n\n+', 'paragraph'),  # Paragraph breaks
                (r'```[\s\S]*?```', 'code'),  # Code blocks
                (r'\*{3,}\n', 'separator'),  # Separators
            ],
            "policy": [
                (r'^(Article|Section|Clause)\s+\d+\.', 'section'),
                (r'^\d+\.\d+\.', 'subsection'),
                (r'^[A-Z][A-Z\s]+:$', 'header'),
                (r'\([a-z]\)', 'list_item'),
            ],
            "legal": [
                (r'WHEREAS', 'whereas'),
                (r'NOW THEREFORE', 'therefore'),
                (r'IN WITNESS WHEREOF', 'witness'),
                (r'^§\s*\d+\.', 'section_symbol'),
            ]
        }

    def chunk_document(self,
                       text: str,
                       method: ChunkingMethod = ChunkingMethod.STRUCTURAL,
                       document_type: str = "policy",
                       metadata: Optional[Dict] = None) -> List[TextChunk]:
        """Chunk document using specified method."""
        logger.info(f"Chunking document using {method} method")

        metadata = metadata or {}

        try:
            if method == ChunkingMethod.SEMANTIC:
                chunks = self._semantic_chunking(text, metadata)
            elif method == ChunkingMethod.STRUCTURAL:
                chunks = self._structural_chunking(text, document_type, metadata)
            elif method == ChunkingMethod.RECURSIVE:
                chunks = self._recursive_chunking(text, metadata)
            else:  # FIXED
                chunks = self._fixed_size_chunking(text, metadata)

            # Add chunk IDs and positions
            for i, chunk in enumerate(chunks):
                chunk.chunk_id = self._generate_chunk_id(chunk, i)

            logger.info(f"Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Chunking failed: {e}", exc_info=True)
            # Fallback to fixed size chunking
            return self._fixed_size_chunking(text, metadata)

    def _semantic_chunking(self, text: str, metadata: Dict) -> List[TextChunk]:
        """Semantic chunking based on content meaning."""
        chunks = []

        # Split by sentences first
        sentences = self._split_sentences(text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would exceed chunk size, start new chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id="",
                    start_pos=0,  # Will be updated
                    end_pos=0,
                    metadata=metadata.copy(),
                    section=self._detect_section(chunk_text)
                ))

                # Keep overlap sentences for next chunk
                overlap_count = len(current_chunk) // 3  # Keep 1/3 for overlap
                current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_len

        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_id="",
                start_pos=0,
                end_pos=0,
                metadata=metadata.copy(),
                section=self._detect_section(chunk_text)
            ))

        # Update positions
        self._update_positions(chunks, text)

        return chunks

    def _structural_chunking(self, text: str, doc_type: str, metadata: Dict) -> List[TextChunk]:
        """Structural chunking based on document format."""
        patterns = self.patterns.get(doc_type, self.patterns["policy"])

        # First, identify structural boundaries
        boundaries = []

        for pattern, boundary_type in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                boundaries.append((match.start(), boundary_type, match.group(0)))

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])

        chunks = []
        start_pos = 0

        for i, (boundary_pos, boundary_type, boundary_text) in enumerate(boundaries):
            # Get chunk from previous boundary to this one
            chunk_text = text[start_pos:boundary_pos].strip()

            if chunk_text and len(chunk_text) > 50:  # Minimum chunk size
                # If chunk is too large, split it
                if len(chunk_text) > self.chunk_size * 2:
                    sub_chunks = self._fixed_size_chunking(chunk_text, metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_id="",
                        start_pos=start_pos,
                        end_pos=boundary_pos,
                        metadata=metadata.copy(),
                        section=boundary_type
                    ))

            start_pos = boundary_pos

        # Add remaining text
        if start_pos < len(text):
            remaining = text[start_pos:].strip()
            if remaining:
                chunks.append(TextChunk(
                    text=remaining,
                    chunk_id="",
                    start_pos=start_pos,
                    end_pos=len(text),
                    metadata=metadata.copy(),
                    section="remaining"
                ))

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _fixed_size_chunking(self, text: str, metadata: Dict) -> List[TextChunk]:
        """Fixed-size chunking with overlap."""
        chunks = []

        # Clean and normalize text
        text = self._clean_text(text)

        words = text.split()
        total_words = len(words)

        i = 0
        while i < total_words:
            # Calculate chunk boundaries with overlap
            chunk_end = min(i + self.chunk_size, total_words)

            # Try to end at sentence boundary
            if chunk_end < total_words:
                # Look for sentence end within lookahead window
                lookahead = min(50, total_words - chunk_end)
                for j in range(chunk_end, chunk_end + lookahead):
                    if words[j].endswith(('.', '!', '?', ';')) or \
                            (j < total_words - 1 and words[j + 1][0].isupper()):
                        chunk_end = j + 1
                        break

            chunk_words = words[max(0, i - self.chunk_overlap):chunk_end]
            chunk_text = ' '.join(chunk_words)

            if chunk_text.strip():
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id="",
                    start_pos=text.find(chunk_text) if chunk_text in text else 0,
                    end_pos=text.find(chunk_text) + len(chunk_text) if chunk_text in text else 0,
                    metadata=metadata.copy(),
                    section=self._detect_section(chunk_text)
                ))

            i = chunk_end - self.chunk_overlap  # Apply overlap

        return chunks

    def _recursive_chunking(self, text: str, metadata: Dict) -> List[TextChunk]:
        """Recursive chunking that splits by different separators."""
        separators = [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words
        ]

        def recursive_split(chunk: str, separator_idx: int) -> List[str]:
            if separator_idx >= len(separators):
                return [chunk]

            separator = separators[separator_idx]
            parts = chunk.split(separator)

            # Filter empty parts
            parts = [p.strip() for p in parts if p.strip()]

            # If parts are still too large, try next separator
            if any(len(p) > self.chunk_size for p in parts) and separator_idx < len(separators) - 1:
                smaller_parts = []
                for part in parts:
                    smaller_parts.extend(recursive_split(part, separator_idx + 1))
                return smaller_parts

            return parts

        # Start recursive splitting
        parts = recursive_split(text, 0)

        chunks = []
        current_chunk = []
        current_length = 0

        for part in parts:
            part_len = len(part)

            if current_length + part_len > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_id="",
                    start_pos=text.find(chunk_text) if chunk_text in text else 0,
                    end_pos=text.find(chunk_text) + len(chunk_text) if chunk_text in text else 0,
                    metadata=metadata.copy()
                ))

                # Keep overlap
                overlap_words = ' '.join(current_chunk).split()[-self.chunk_overlap // 5:]
                current_chunk = overlap_words if overlap_words else []
                current_length = sum(len(w) for w in current_chunk)

            current_chunk.append(part)
            current_length += part_len

        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_id="",
                start_pos=text.find(chunk_text) if chunk_text in text else 0,
                end_pos=text.find(chunk_text) + len(chunk_text) if chunk_text in text else 0,
                metadata=metadata.copy()
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect section type from text."""
        # Check for headings
        heading_match = re.match(r'^(#{1,6}|[A-Z][A-Z\s]+:|§\s*\d+\.)', text.strip())
        if heading_match:
            return "heading"

        # Check for lists
        if re.match(r'^(\d+\.|[-*•])\s', text.strip()):
            return "list"

        # Check for code
        if '```' in text:
            return "code"

        # Default to paragraph
        return "paragraph"

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple whitespaces
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters except newlines
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace("'", "'")

        return text.strip()

    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for chunk in chunks[1:]:
            # Merge if both are small and same section type
            if (len(current.text) < self.chunk_size // 2 and
                    len(chunk.text) < self.chunk_size // 2 and
                    current.section == chunk.section):

                # Merge chunks
                merged_text = current.text + " " + chunk.text
                current = TextChunk(
                    text=merged_text,
                    chunk_id="",
                    start_pos=current.start_pos,
                    end_pos=chunk.end_pos,
                    metadata=current.metadata,
                    section=current.section
                )
            else:
                merged.append(current)
                current = chunk

        merged.append(current)
        return merged

    def _update_positions(self, chunks: List[TextChunk], original_text: str):
        """Update start and end positions based on original text."""
        for chunk in chunks:
            if not chunk.start_pos and not chunk.end_pos:
                # Find position in original text
                start = original_text.find(chunk.text)
                if start != -1:
                    chunk.start_pos = start
                    chunk.end_pos = start + len(chunk.text)

    def _generate_chunk_id(self, chunk: TextChunk, index: int) -> str:
        """Generate unique chunk ID."""
        # Use metadata + hash for unique ID
        source_id = chunk.metadata.get("document_id", "unknown")
        text_hash = hashlib.md5(chunk.text.encode()).hexdigest()[:8]

        return f"{source_id}_chunk{index:04d}_{text_hash}"

    def calculate_optimal_chunk_size(self, documents: List[str]) -> Tuple[int, int]:
        """Calculate optimal chunk size based on document statistics."""
        if not documents:
            return self.chunk_size, self.chunk_overlap

        # Analyze document lengths
        lengths = [len(doc) for doc in documents]
        avg_length = sum(lengths) / len(lengths)

        # Adjust chunk size based on average length
        if avg_length < 500:
            optimal_size = 500
            optimal_overlap = 100
        elif avg_length < 2000:
            optimal_size = 1000
            optimal_overlap = 200
        elif avg_length < 10000:
            optimal_size = 1500
            optimal_overlap = 300
        else:
            optimal_size = 2000
            optimal_overlap = 400

        logger.info(f"Optimal chunk size: {optimal_size}, overlap: {optimal_overlap}")
        return optimal_size, optimal_overlap
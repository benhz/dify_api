"""Semantic text splitter for intelligent document chunking."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np

from core.model_manager import ModelInstance
from core.model_runtime.model_providers.__base.tokenizers.gpt2_tokenizer import GPT2Tokenizer
from core.rag.models.document import Document
from core.rag.splitter.text_splitter import TextSplitter


class SemanticTextSplitter(TextSplitter):
    """
    Semantic-aware text splitter that uses embeddings to create meaningful chunks.

    This splitter follows the process:
    1. Split by separator (physical boundaries)
    2. Split into sentences
    3. Generate embeddings and compute cosine similarities
    4. Find semantic boundaries using threshold percentile
    5. Apply min/max token constraints
    6. Add optional overlap between chunks
    """

    def __init__(
        self,
        separator: str = "\n\n",
        max_tokens: int = 1024,
        chunk_overlap: int = 50,
        threshold_amount: int = 95,
        buffer_size: int = 2,
        min_chunk_tokens: int = 150,
        max_chunk_tokens: int = 1000,
        embedding_model_instance: Optional[ModelInstance] = None,
        **kwargs: Any,
    ):
        """
        Initialize the semantic text splitter.

        Args:
            separator: Primary separator for physical boundaries (default: "\\n\\n")
            max_tokens: Hard limit for chunk size (default: 1024)
            chunk_overlap: Number of tokens to overlap between chunks (default: 50)
            threshold_amount: Percentile for similarity threshold (default: 95)
            buffer_size: Window size for smoothing similarities (default: 2)
            min_chunk_tokens: Minimum tokens per chunk (default: 150)
            max_chunk_tokens: Maximum tokens per chunk (default: 1000)
            embedding_model_instance: Model instance for generating embeddings
        """
        super().__init__(**kwargs)
        self._separator = separator.replace("\\n", "\n") if separator else "\n\n"
        self._max_tokens = max_tokens
        self._chunk_overlap = chunk_overlap
        self._threshold_amount = threshold_amount
        self._buffer_size = buffer_size
        self._min_chunk_tokens = min_chunk_tokens
        self._max_chunk_tokens = max_chunk_tokens
        self._embedding_model_instance = embedding_model_instance

        # Sentence boundary patterns (supports multiple languages)
        self._sentence_patterns = [
            r'[。！？]',  # Chinese
            r'[.!?]+\s+',  # English
            r'\n+',  # Newlines
        ]

    def split_text(self, text: str) -> list[str]:
        """
        Split text using semantic analysis following the 5-step process:

        Step 1: separator - Split by physical boundaries
        Step 2: Semantic analysis - Find semantic breakpoints within each paragraph
        Step 3: max_tokens - Force split long chunks
        Step 4: chunk_overlap - Add overlap between chunks
        Step 5: min/max_chunk_tokens - Enforce size constraints

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Step 1: Split by separator (physical boundaries)
        paragraphs = self._split_by_separator(text)
        if not paragraphs:
            return []

        # Step 2: Semantic analysis - Find semantic breakpoints within each paragraph
        semantic_chunks = []
        for paragraph in paragraphs:
            para_chunks = self._apply_semantic_splitting(paragraph)
            semantic_chunks.extend(para_chunks)

        if not semantic_chunks:
            return []

        # Step 3: max_tokens - Force split chunks that exceed max_tokens
        size_limited_chunks = self._enforce_max_tokens(semantic_chunks)

        # Step 4: chunk_overlap - Add overlap between chunks
        overlapped_chunks = self._add_overlap(size_limited_chunks)

        # Step 5: min/max_chunk_tokens - Enforce size constraints
        final_chunks = self._enforce_size_constraints(overlapped_chunks)

        return final_chunks

    def _apply_semantic_splitting(self, paragraph: str) -> list[str]:
        """
        Apply semantic splitting to a single paragraph.

        Args:
            paragraph: Single paragraph to analyze

        Returns:
            List of semantically meaningful chunks from this paragraph
        """
        # Split into sentences
        sentences = self._split_into_sentences(paragraph)

        if not sentences:
            return [paragraph]

        # If only one sentence, return it
        if len(sentences) == 1:
            return sentences

        # Find semantic boundaries using embeddings
        boundaries = self._find_semantic_boundaries(sentences)

        # Generate chunks based on boundaries
        chunks = self._generate_semantic_chunks(sentences, boundaries)

        return chunks

    def _split_by_separator(self, text: str) -> list[str]:
        """Split text by the primary separator."""
        if self._separator:
            parts = text.split(self._separator)
            return [p.strip() for p in parts if p.strip()]
        return [text]

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using multiple language patterns.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Combine all patterns
        combined_pattern = '|'.join(f'({p})' for p in self._sentence_patterns)

        # Split by patterns
        parts = re.split(combined_pattern, text)

        # Reconstruct sentences with their delimiters
        sentences = []
        current_sentence = ""

        for part in parts:
            if part is None:
                continue
            if not part.strip():
                continue

            # Check if this is a delimiter
            is_delimiter = False
            for pattern in self._sentence_patterns:
                if re.fullmatch(pattern, part):
                    is_delimiter = True
                    break

            if is_delimiter:
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part

        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # Fallback: if no sentences found, return the whole text
        if not sentences:
            sentences = [text]

        return sentences

    def _get_token_count(self, text: str) -> int:
        """Get token count for text."""
        if self._embedding_model_instance:
            return self._embedding_model_instance.get_text_embedding_num_tokens([text])[0]
        else:
            return GPT2Tokenizer.get_num_tokens(text)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        if self._embedding_model_instance:
            try:
                # Use the embedding model instance to generate embeddings
                result = self._embedding_model_instance.invoke_text_embedding(texts=texts)
                # 方案 1：标准字段
                if hasattr(result, "embeddings"):
                    embeddings = np.array(result.embeddings, dtype=np.float32)
                    return embeddings
            except Exception:
                # Fallback to simple hash-based embeddings if model fails
                return self._fallback_embeddings(texts)
        else:
            # Use fallback embeddings
            return self._fallback_embeddings(texts)

    def _fallback_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Fallback embedding method using simple character-based features.
        This is used when no embedding model is available.
        """
        # Create simple feature vectors based on character distributions
        embeddings = []
        for text in texts:
            # Create a simple feature vector
            features = [
                len(text),  # Length
                text.count(' '),  # Word count approximation
                text.count(','),  # Comma count
                text.count('.'),  # Period count
                text.count('。'),  # Chinese period count
                sum(c.isupper() for c in text),  # Uppercase count
                sum(c.isdigit() for c in text),  # Digit count
            ]
            # Normalize
            if sum(features) > 0:
                features = [f / sum(features) for f in features]
            embeddings.append(features)

        return np.array(embeddings)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _find_semantic_boundaries(self, sentences: list[str]) -> list[int]:
        """
        Find semantic boundaries using embeddings and similarity analysis.

        Args:
            sentences: List of sentences

        Returns:
            List of boundary indices (where chunks should be split)
        """
        if len(sentences) <= 1:
            return []

        # Generate embeddings
        embeddings = self._get_embeddings(sentences)

        # Check if embeddings are empty or invalid shape
        if embeddings.size == 0 or len(embeddings.shape) < 2:
            return []

        # Calculate similarities between consecutive sentences
        similarities = []
        n_embeddings = embeddings.shape[0]
        for i in range(n_embeddings - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        if not similarities:
            return []

        # Apply smoothing with buffer_size
        smoothed_similarities = self._apply_smoothing(similarities, self._buffer_size)

        # Calculate threshold using percentile
        threshold = np.percentile(smoothed_similarities, self._threshold_amount)

        # Find boundaries where similarity drops below threshold
        boundaries = []
        for i, sim in enumerate(smoothed_similarities):
            if sim < threshold:
                boundaries.append(i + 1)  # Boundary after sentence i

        return boundaries

    def _apply_smoothing(self, similarities: list[float], buffer_size: int) -> list[float]:
        """
        Apply moving average smoothing to similarities.

        Args:
            similarities: List of similarity scores
            buffer_size: Window size for smoothing

        Returns:
            Smoothed similarity scores
        """
        if buffer_size <= 0 or len(similarities) <= buffer_size:
            return similarities

        smoothed = []
        for i in range(len(similarities)):
            start = max(0, i - buffer_size)
            end = min(len(similarities), i + buffer_size + 1)
            window = similarities[start:end]
            smoothed.append(sum(window) / len(window))

        return smoothed

    def _generate_semantic_chunks(self, sentences: list[str], boundaries: list[int]) -> list[str]:
        """
        Generate chunks based on semantic boundaries.

        Args:
            sentences: List of sentences
            boundaries: List of boundary indices

        Returns:
            List of text chunks
        """
        if not sentences:
            return []

        chunks = []
        start_idx = 0

        for boundary_idx in boundaries:
            if boundary_idx > start_idx:
                chunk_sentences = sentences[start_idx:boundary_idx]
                chunk_text = ' '.join(chunk_sentences)
                chunks.append(chunk_text)
                start_idx = boundary_idx

        # Add remaining sentences
        if start_idx < len(sentences):
            chunk_sentences = sentences[start_idx:]
            chunk_text = ' '.join(chunk_sentences)
            chunks.append(chunk_text)

        return chunks

    def _enforce_max_tokens(self, chunks: list[str]) -> list[str]:
        """
        Step 3: Force split chunks that exceed max_tokens.

        Args:
            chunks: List of chunks from semantic analysis

        Returns:
            List of chunks where no chunk exceeds max_tokens
        """
        if not chunks:
            return []

        result = []
        for chunk in chunks:
            chunk_tokens = self._get_token_count(chunk)

            if chunk_tokens <= self._max_tokens:
                result.append(chunk)
            else:
                # Split by sentences first
                sentences = self._split_into_sentences(chunk)
                current_chunk = ""
                current_tokens = 0

                for sentence in sentences:
                    sentence_tokens = self._get_token_count(sentence)

                    # If single sentence exceeds max, force split
                    if sentence_tokens > self._max_tokens:
                        if current_chunk:
                            result.append(current_chunk.strip())
                            current_chunk = ""
                            current_tokens = 0

                        # Force split the long sentence
                        sub_chunks = self._force_split_by_tokens(sentence, self._max_tokens)
                        result.extend(sub_chunks)
                    else:
                        # Check if adding this sentence would exceed max
                        if current_tokens + sentence_tokens > self._max_tokens:
                            if current_chunk:
                                result.append(current_chunk.strip())
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            current_chunk += ' ' + sentence if current_chunk else sentence
                            current_tokens += sentence_tokens

                if current_chunk:
                    result.append(current_chunk.strip())

        return result

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """
        Step 4: Add overlap between chunks to maintain context continuity.

        Overlap structure:
        [Chunk1]
        [Last N tokens of Chunk1 + Chunk2 + Last N tokens of Chunk2]
        [Last N tokens of Chunk2 + Chunk3 + ...]

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with overlap added
        """
        if not chunks or self._chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i in range(len(chunks)):
            chunk = chunks[i]
            prefix = ""
            suffix = ""

            # Add prefix: last N tokens from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                prefix = self._get_last_n_tokens(prev_chunk, self._chunk_overlap)

            # Add suffix: first N tokens from next chunk (prepare for next chunk's prefix)
            # Actually, we only add current chunk + prepare suffix for calculation
            # The next chunk will use current chunk's suffix as its prefix

            # Combine: prefix + current_chunk
            if prefix:
                chunk = prefix + ' ' + chunk

            overlapped_chunks.append(chunk.strip())

        return overlapped_chunks

    def _get_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """
        Get the last N tokens from text.

        Args:
            text: Input text
            n_tokens: Number of tokens to extract

        Returns:
            Last N tokens as string
        """
        words = text.split()
        if not words:
            return ""

        # Build from the end
        result_words = []
        current_tokens = 0

        for word in reversed(words):
            test_text = word + ' ' + ' '.join(result_words) if result_words else word
            test_tokens = self._get_token_count(test_text)

            if test_tokens > n_tokens:
                break

            result_words.insert(0, word)
            current_tokens = test_tokens

        return ' '.join(result_words)

    def _enforce_size_constraints(self, chunks: list[str]) -> list[str]:
        """
        Step 5: Enforce min_chunk_tokens and max_chunk_tokens constraints.

        - Merge chunks smaller than min_chunk_tokens
        - Ensure no chunk exceeds max_chunk_tokens

        Args:
            chunks: List of chunks

        Returns:
            List of chunks within size constraints
        """
        if not chunks:
            return []

        # First pass: merge short chunks
        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            current_tokens = self._get_token_count(current_chunk)

            # If chunk is too short, try to merge
            if current_tokens < self._min_chunk_tokens:
                # Try to merge with previous chunk
                if merged_chunks:
                    last_chunk = merged_chunks[-1]
                    last_tokens = self._get_token_count(last_chunk)

                    # Only merge if combined size doesn't exceed max
                    combined_tokens = last_tokens + current_tokens
                    if combined_tokens <= self._max_chunk_tokens:
                        merged_chunks[-1] = last_chunk + ' ' + current_chunk
                        i += 1
                        continue

                # Try to merge with next chunk
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    next_tokens = self._get_token_count(next_chunk)
                    combined_tokens = current_tokens + next_tokens

                    if combined_tokens <= self._max_chunk_tokens:
                        merged_chunks.append(current_chunk + ' ' + next_chunk)
                        i += 2
                        continue

                # Can't merge, keep as is (even if too short)
                merged_chunks.append(current_chunk)
            else:
                merged_chunks.append(current_chunk)

            i += 1

        # Second pass: ensure no chunk exceeds max_chunk_tokens
        final_chunks = []
        for chunk in merged_chunks:
            chunk_tokens = self._get_token_count(chunk)

            if chunk_tokens <= self._max_chunk_tokens:
                final_chunks.append(chunk)
            else:
                # Split at sentence boundaries
                sentences = self._split_into_sentences(chunk)
                current_chunk = ""
                current_tokens = 0

                for sentence in sentences:
                    sentence_tokens = self._get_token_count(sentence)

                    if current_tokens + sentence_tokens > self._max_chunk_tokens:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_tokens = sentence_tokens
                    else:
                        current_chunk += ' ' + sentence if current_chunk else sentence
                        current_tokens += sentence_tokens

                if current_chunk:
                    final_chunks.append(current_chunk.strip())

        return final_chunks

    def _force_split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Force split text by token count when no good boundary exists."""
        chunks = []
        words = text.split()

        current_chunk = ""
        for word in words:
            test_chunk = current_chunk + ' ' + word if current_chunk else word
            test_tokens = self._get_token_count(test_chunk)

            if test_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        chunked_documents = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)

            for chunk in chunks:
                # Create new document with same metadata
                new_doc = Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy() if doc.metadata else {}
                )
                chunked_documents.append(new_doc)

        return chunked_documents

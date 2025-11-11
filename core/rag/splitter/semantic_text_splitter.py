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

    def _is_table(self, text: str) -> bool:
        """
        Check if text contains a table structure.

        Detects:
        - Markdown tables with separator line: |---|---|
        - Markdown tables without separator: multiple rows with | delimiters
        - HTML tables: <table>...</table>

        Args:
            text: Text to check

        Returns:
            True if text contains a table
        """
        # Check for HTML table tags
        if '<table' in text.lower() and '</table>' in text.lower():
            return True

        lines = text.split('\n')
        table_like_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if line starts and ends with |
            if stripped.startswith('|') and stripped.endswith('|'):
                # Check if it's a separator line (|---|---|)
                check_str = stripped.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')
                if len(check_str) == 0 and stripped.count('-') >= 2:
                    # Found separator line, definitely a table
                    return True

                # Count how many | delimiters (at least 3 for a valid table row)
                pipe_count = stripped.count('|')
                if pipe_count >= 3:  # At least 2 columns (3 pipes: |col1|col2|)
                    table_like_lines.append(line)

        # If we have 3+ consecutive lines with | delimiters, it's a table
        # Even without a separator line
        if len(table_like_lines) >= 3:
            return True

        return False

    def _extract_tables(self, text: str) -> list[tuple[int, int]]:
        """
        Extract table positions (start, end) from text.

        Args:
            text: Text to analyze

        Returns:
            List of (start_pos, end_pos) tuples for each table
        """
        lines = text.split('\n')
        tables = []
        current_table_start = None
        current_table_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check if this line is part of a table
            is_table_line = False
            if stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 3:
                is_table_line = True

            if is_table_line:
                if current_table_start is None:
                    current_table_start = i
                current_table_lines.append(i)
            else:
                # End of table
                if current_table_start is not None and len(current_table_lines) >= 3:
                    # Valid table (3+ rows)
                    tables.append((current_table_start, current_table_lines[-1]))
                current_table_start = None
                current_table_lines = []

        # Check last table
        if current_table_start is not None and len(current_table_lines) >= 3:
            tables.append((current_table_start, current_table_lines[-1]))

        return tables

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using multiple language patterns.

        Tables are treated as single sentences and kept intact.
        Regular text is split by sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of sentences (tables are single items)
        """
        if not text:
            return []

        lines = text.split('\n')

        # Find all table regions
        table_regions = self._extract_tables(text)

        # Build a set of table line indices for quick lookup
        table_line_indices = set()
        for start, end in table_regions:
            for i in range(start, end + 1):
                table_line_indices.add(i)

        # Split text into segments: [text, table, text, table, ...]
        segments = []
        current_text_lines = []

        for i, line in enumerate(lines):
            if i in table_line_indices:
                # This line is part of a table
                if current_text_lines:
                    # Save accumulated text
                    segments.append(('text', '\n'.join(current_text_lines)))
                    current_text_lines = []

                # Check if this is the start of a new table
                is_table_start = i not in table_line_indices or (i > 0 and i - 1 not in table_line_indices)
                if is_table_start or not segments or segments[-1][0] != 'table':
                    # Start a new table segment
                    table_lines = []
                    # Collect all consecutive table lines
                    for j in range(i, len(lines)):
                        if j in table_line_indices:
                            table_lines.append(lines[j])
                        else:
                            break
                    if table_lines:
                        segments.append(('table', '\n'.join(table_lines)))
            else:
                # Regular text line
                if segments and segments[-1][0] == 'table':
                    # Previous segment was a table, start new text segment
                    current_text_lines = [line]
                else:
                    current_text_lines.append(line)

        # Add remaining text
        if current_text_lines:
            segments.append(('text', '\n'.join(current_text_lines)))

        # Now split each segment
        sentences = []
        for seg_type, seg_text in segments:
            if seg_type == 'table':
                # Table as a single sentence
                sentences.append(seg_text)
            else:
                # Split text by sentence patterns
                text_sentences = self._split_text_by_patterns(seg_text)
                sentences.extend(text_sentences)

        # Fallback
        if not sentences:
            sentences = [text]

        return sentences

    def _split_text_by_patterns(self, text: str) -> list[str]:
        """
        Split regular text by sentence patterns.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text or not text.strip():
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

        return sentences

    def _get_token_count(self, text: str) -> int:
        """
        Estimate token count for text using character count.

        Uses a simple estimation: length / 4 for English, length / 1.5 for Chinese.
        This avoids frequent network requests to the embedding API.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Count Chinese characters (CJK)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))

        # Estimate tokens:
        # - Chinese: ~1.5 tokens per character (more accurate for GPT tokenizers)
        # - English: ~1 token per 4 characters
        total_chars = len(text)
        english_chars = total_chars - chinese_chars

        estimated_tokens = int(chinese_chars / 1.5 + english_chars / 4)

        return max(1, estimated_tokens)  # At least 1 token

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with dynamic batching.

        Uses dual thresholds to avoid timeout and improve performance:
        - max_sentences_per_batch: 1000 sentences
        - max_bytes_per_batch: 48KB UTF-8 bytes

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        if self._embedding_model_instance:
            try:
                # Use dynamic batching to avoid timeout and improve performance
                result = self._get_embeddings_with_batching(texts)
                # result is a list of embeddings, convert to numpy array
                if result:
                    embeddings = np.array(result, dtype=np.float32)
                    return embeddings
                else:
                    # Empty result, use fallback
                    return self._fallback_embeddings(texts)
            except Exception:
                # Fallback to simple hash-based embeddings if model fails
                return self._fallback_embeddings(texts)
        else:
            # Use fallback embeddings
            return self._fallback_embeddings(texts)

    def _get_embeddings_with_batching(self, texts: list[str]) -> list:
        """
        Get embeddings with dynamic batching to handle large text lists safely.

        Uses dual thresholds:
        - max_sentences_per_batch: 1000 sentences
        - max_bytes_per_batch: 48KB UTF-8 bytes

        Accumulates texts into batches until either threshold is reached,
        then processes the batch and continues with remaining texts.

        Args:
            texts: List of text strings

        Returns:
            List of embeddings (list[list[float]]) in original order
        """
        MAX_SENTENCES_PER_BATCH = 64
        MAX_BYTES_PER_BATCH = 24 * 1024  # 48 KiB

        all_embeddings = []
        current_batch = []
        current_bytes = 0

        for text in texts:
            text_bytes = len(text.encode('utf-8'))

            # Check if adding this text would exceed either threshold
            would_exceed_count = len(current_batch) >= MAX_SENTENCES_PER_BATCH
            would_exceed_bytes = current_bytes + text_bytes > MAX_BYTES_PER_BATCH

            # If batch is not empty and would exceed, process current batch
            if current_batch and (would_exceed_count or would_exceed_bytes):
                batch_result = self._embedding_model_instance.invoke_text_embedding(
                    texts=current_batch
                )
                # Extract embeddings from TextEmbeddingResult object
                all_embeddings.extend(batch_result.embeddings)

                # Reset batch
                current_batch = []
                current_bytes = 0

            # Add current text to batch
            current_batch.append(text)
            current_bytes += text_bytes

        # Process remaining batch
        if current_batch:
            batch_result = self._embedding_model_instance.invoke_text_embedding(
                texts=current_batch
            )
            # Extract embeddings from TextEmbeddingResult object
            all_embeddings.extend(batch_result.embeddings)

        return all_embeddings

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

        # Check if embeddings are None, empty or invalid shape
        if embeddings is None or embeddings.size == 0 or len(embeddings.shape) < 2:
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

        Tables are split by rows to preserve structure.

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
                # Check if this is a table - if so, split by rows
                if self._is_table(chunk):
                    sub_chunks = self._split_table_by_rows(chunk, self._max_tokens)
                    result.extend(sub_chunks)
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

    def _split_table_by_rows(self, table: str, max_tokens: int) -> list[str]:
        """
        Split a table by rows while preserving table structure.

        Each chunk includes the table header plus data rows.
        Header can be:
        - 1 line (column names only)
        - 2 lines (column names + separator |---|)

        Args:
            table: Table text to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of table chunks, each with header + data rows
        """
        lines = [line for line in table.split('\n') if line.strip()]

        if len(lines) <= 1:
            # Table too small to split
            return [table]

        # Detect header: first line + optional separator line
        header_lines = []
        data_lines = []

        # First line is always header
        header_lines.append(lines[0])

        # Check if second line is separator (|---|---|)
        header_end_index = 1
        if len(lines) > 1:
            stripped = lines[1].strip()
            if stripped.startswith('|') and stripped.endswith('|'):
                check_str = stripped.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')
                if len(check_str) == 0 and stripped.count('-') >= 2:
                    # Second line is separator
                    header_lines.append(lines[1])
                    header_end_index = 2

        # Everything else is data
        data_lines = lines[header_end_index:]

        # If no data rows, return as is
        if not data_lines:
            return [table]

        # Check if table is small enough to keep as one chunk
        total_tokens = self._get_token_count(table)
        if total_tokens <= max_tokens:
            return [table]

        # Split into chunks: each chunk = header + some data rows
        header_text = '\n'.join(header_lines)
        header_tokens = self._get_token_count(header_text)

        chunks = []
        current_chunk_lines = header_lines.copy()
        current_tokens = header_tokens

        for line in data_lines:
            line_tokens = self._get_token_count(line)

            # Check if adding this line would exceed max
            if current_tokens + line_tokens > max_tokens and len(current_chunk_lines) > len(header_lines):
                # Save current chunk and start new one with header
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = header_lines.copy()
                current_tokens = header_tokens

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        # Add remaining chunk
        if len(current_chunk_lines) > len(header_lines):
            chunks.append('\n'.join(current_chunk_lines))

        return chunks if chunks else [table]

    def _split_lines_by_tokens(self, lines: list[str], max_tokens: int) -> list[str]:
        """
        Split lines by token count when no clear structure exists.

        Args:
            lines: List of lines
            max_tokens: Maximum tokens per chunk

        Returns:
            List of chunks
        """
        chunks = []
        current_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._get_token_count(line)

            if current_tokens + line_tokens > max_tokens and current_lines:
                chunks.append('\n'.join(current_lines))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            chunks.append('\n'.join(current_lines))

        return chunks if chunks else ['\n'.join(lines)]

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
                # Check if this is a table - if so, split by rows
                if self._is_table(chunk):
                    sub_chunks = self._split_table_by_rows(chunk, self._max_chunk_tokens)
                    final_chunks.extend(sub_chunks)
                else:
                    # Split at sentence boundaries
                    sentences = self._split_into_sentences(chunk)
                    current_chunk = ""
                    current_tokens = 0

                    for sentence in sentences:
                        sentence_tokens = self._get_token_count(sentence)

                        # If a single sentence exceeds max_chunk_tokens, force split it
                        if sentence_tokens > self._max_chunk_tokens:
                            # Append current chunk if exists
                            if current_chunk:
                                final_chunks.append(current_chunk.strip())
                                current_chunk = ""
                                current_tokens = 0

                            # Force split the long sentence
                            split_sentences = self._force_split_by_tokens(sentence, self._max_chunk_tokens)
                            final_chunks.extend(split_sentences)
                        elif current_tokens + sentence_tokens > self._max_chunk_tokens:
                            # Adding this sentence would exceed max, start new chunk
                            if current_chunk:
                                final_chunks.append(current_chunk.strip())
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            # Add sentence to current chunk
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

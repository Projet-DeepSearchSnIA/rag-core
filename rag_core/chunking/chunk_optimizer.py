from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import replace
import re

from rag_core.chunking.chunk_schemas import ChunkMetadata, DocumentChunk
from rag_core.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkOptimizer:
    """optimise les chunks pour améliorer la qualité du RAG"""

    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        target_chunk_size: int,
        merge_small_chunks: bool,
        split_large_chunks: bool,
        remove_duplicates: bool,
        similarity_threshold: float,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        self.merge_small_chunks = merge_small_chunks
        self.split_large_chunks = split_large_chunks
        self.remove_duplicates = remove_duplicates
        self.similarity_threshold = similarity_threshold

    def optimize_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], Dict]:
        logger.info("optimisation de %d chunks...", len(chunks))

        original_count = len(chunks)
        optimized = chunks.copy()
        stats = {
            'original_count': original_count,
            'operations': []
        }

        optimized = self._remove_empty_chunks(optimized)
        if len(optimized) < original_count:
            stats['operations'].append({'type': 'remove_empty', 'removed': original_count - len(optimized)})

        if self.remove_duplicates:
            before = len(optimized)
            optimized = self._remove_duplicate_chunks(optimized)
            if len(optimized) < before:
                stats['operations'].append({'type': 'remove_duplicates', 'removed': before - len(optimized)})

        if self.merge_small_chunks:
            before = len(optimized)
            optimized = self._merge_small_chunks(optimized)
            if len(optimized) < before:
                stats['operations'].append({'type': 'merge_small', 'merged': before - len(optimized)})

        if self.split_large_chunks:
            before = len(optimized)
            optimized = self._split_large_chunks(optimized)
            if len(optimized) > before:
                stats['operations'].append({'type': 'split_large', 'split': len(optimized) - before})

        optimized = self._reindex_chunks(optimized)

        stats['final_count'] = len(optimized)
        stats['size_stats'] = self._compute_size_stats(optimized)

        logger.info("%d chunks finaux", stats["final_count"])
        return optimized, stats

    def _remove_empty_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        return [c for c in chunks if c.content.strip() and len(c.content.strip()) > 10]

    def _remove_duplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        unique_chunks = []
        seen_contents = []

        for chunk in chunks:
            normalized = self._normalize_text(chunk.content)
            is_duplicate = any(
                self._text_similarity(normalized, seen) >= self.similarity_threshold
                for seen in seen_contents
            )
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_contents.append(normalized)

        return unique_chunks

    def _merge_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        if not chunks:
            return chunks

        merged = []
        buffer = []
        buffer_size = 0

        for chunk in chunks:
            if chunk.char_count >= self.min_chunk_size:
                if buffer:
                    merged.append(self._merge_chunk_list(buffer))
                    buffer = []
                    buffer_size = 0
                merged.append(chunk)
            else:
                buffer.append(chunk)
                buffer_size += chunk.char_count
                if buffer_size >= self.min_chunk_size:
                    merged.append(self._merge_chunk_list(buffer))
                    buffer = []
                    buffer_size = 0

        if buffer:
            if merged:
                last = merged.pop()
                buffer.insert(0, last)
            merged.append(self._merge_chunk_list(buffer))

        return merged

    def _split_large_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        result = []
        for chunk in chunks:
            if chunk.char_count <= self.max_chunk_size or chunk.metadata.has_formulas:
                result.append(chunk)
            else:
                result.extend(self._split_chunk(chunk))
        return result

    def _split_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        content = chunk.content
        sentences = re.split(r'[.!?]\s+', content)
        sub_chunks = []
        current_text = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.target_chunk_size and current_text:
                sub_content = ". ".join(current_text) + "."
                sub_chunks.append(self._create_sub_chunk(chunk, sub_content, len(sub_chunks)))
                current_text = [sentence]
                current_size = sentence_size
            else:
                current_text.append(sentence)
                current_size += sentence_size

        if current_text:
            sub_content = ". ".join(current_text) + "."
            sub_chunks.append(self._create_sub_chunk(chunk, sub_content, len(sub_chunks)))

        return sub_chunks if len(sub_chunks) > 1 else [chunk]

    def _merge_chunk_list(self, chunks: List[DocumentChunk]) -> DocumentChunk:
        if len(chunks) == 1:
            return chunks[0]

        merged_content = "\n\n".join([c.content for c in chunks])

        all_pages = []
        for chunk in chunks:
            all_pages.extend(chunk.page_numbers)
        unique_pages = sorted(list(set(all_pages)))

        image_ids, image_paths, images, formulas = [], [], [], []
        has_images, has_formulas = False, False

        for c in chunks:
            image_ids.extend(c.metadata.image_ids)
            image_paths.extend(c.metadata.image_paths)
            images.extend(c.metadata.images)
            formulas.extend(c.metadata.formulas)
            if c.metadata.has_images:
                has_images = True
            if c.metadata.has_formulas:
                has_formulas = True

        merged_metadata = replace(
            chunks[0].metadata,
            image_ids=list(dict.fromkeys(image_ids)),
            image_paths=list(dict.fromkeys(image_paths)),
            images=images,
            formulas=formulas,
            has_images=has_images,
            has_formulas=has_formulas,
            merged_from=[c.chunk_id for c in chunks],
        )

        return DocumentChunk(
            chunk_id=chunks[0].chunk_id,
            content=merged_content,
            document_id=chunks[0].document_id,
            document_name=chunks[0].document_name,
            page_numbers=unique_pages,
            chunk_index=chunks[0].chunk_index,
            total_chunks=chunks[0].total_chunks,
            metadata=merged_metadata,
        )

    def _create_sub_chunk(self, parent: DocumentChunk, content: str, sub_index: int) -> DocumentChunk:
        sub_metadata = replace(parent.metadata, split_from=parent.chunk_id, sub_index=sub_index)
        return DocumentChunk(
            chunk_id=f"{parent.chunk_id}_sub_{sub_index}",
            content=content,
            document_id=parent.document_id,
            document_name=parent.document_name,
            page_numbers=parent.page_numbers.copy(),
            chunk_index=parent.chunk_index,
            total_chunks=parent.total_chunks,
            metadata=sub_metadata,
        )

    def _reindex_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total
            chunk.chunk_id = f"{chunk.document_id}_chunk_{i}"
        return chunks

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())

    def _text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _compute_size_stats(self, chunks: List[DocumentChunk]) -> Dict:
        if not chunks:
            return {}
        sizes = [c.char_count for c in chunks]
        return {
            'min': min(sizes),
            'max': max(sizes),
            'mean': sum(sizes) / len(sizes),
            'median': sorted(sizes)[len(sizes) // 2],
            'total_chars': sum(sizes)
        }

    def analyze_chunks(self, chunks: List[DocumentChunk]) -> Dict:
        if not chunks:
            return {'total': 0}

        sizes = [c.char_count for c in chunks]
        word_counts = [c.word_count for c in chunks]
        page_distribution = Counter()
        for chunk in chunks:
            for page in chunk.page_numbers:
                page_distribution[page] += 1

        return {
            'total_chunks': len(chunks),
            'size_stats': {
                'chars': {'min': min(sizes), 'max': max(sizes), 'mean': sum(sizes) / len(sizes), 'median': sorted(sizes)[len(sizes) // 2]},
                'words': {'min': min(word_counts), 'max': max(word_counts), 'mean': sum(word_counts) / len(word_counts), 'median': sorted(word_counts)[len(word_counts) // 2]}
            },
            'page_distribution': dict(page_distribution.most_common(10)),
            'metadata_stats': {
                'with_section_title': sum(1 for c in chunks if c.metadata.section_title),
                'with_images': sum(1 for c in chunks if c.metadata.has_images),
                'with_tables': sum(1 for c in chunks if c.metadata.has_tables),
            },
            'quality_checks': {
                'too_small': sum(1 for s in sizes if s < self.min_chunk_size),
                'too_large': sum(1 for s in sizes if s > self.max_chunk_size),
                'optimal': sum(1 for s in sizes if self.min_chunk_size <= s <= self.max_chunk_size)
            }
        }

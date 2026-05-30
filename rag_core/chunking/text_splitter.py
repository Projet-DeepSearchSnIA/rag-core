from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional, Literal
import json
from dataclasses import replace
from pathlib import Path

from rag_core.extraction.document_schemas import ExtractedDocument, ContentBlock
from rag_core.chunking.chunk_schemas import ChunkMetadata, DocumentChunk
from rag_core.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["SmartTextSplitter", "DocumentChunk", "ChunkMetadata"]


class SmartTextSplitter:
    """
    splitter intelligent qui préserve la structure du document.
    utilise LangChain RecursiveCharacterTextSplitter.
    supporte trois stratégies : recursive, semantic, mixed.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        length_function: callable = len,
        strategy: Literal["recursive", "semantic", "mixed"] = "recursive"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        if separators is None:
            separators = [
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=length_function
        )

    def split_document(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        logger.info("chunking de: %s, pages: %d, stratégie: %s", doc.filename, len(doc.pages), self.strategy)

        if self.strategy == "recursive":
            chunks = self._split_recursive(doc)
        elif self.strategy == "semantic":
            chunks = self._split_semantic(doc)
        elif self.strategy == "mixed":
            chunks = self._split_mixed(doc)
        else:
            raise ValueError(f"stratégie inconnue: {self.strategy}")

        logger.info("%d chunks créés", len(chunks))
        return chunks

    def _split_recursive(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        chunks = []
        chunk_counter = 0

        for page in doc.pages:
            page_blocks = [
                block for block in page.content_blocks
                if block.type in ["text", "title", "list", "formula", "image", "table"]
            ]

            if not page_blocks:
                continue

            page_text, placeholder_maps = self._build_text_with_placeholders(page_blocks)
            text_chunks = self.splitter.split_text(page_text)

            for chunk_text in text_chunks:
                chunk_id = f"{doc.document_id}_chunk_{chunk_counter}"
                expanded_text, formulas_used, images_used = self._expand_placeholders_and_collect_metadata(
                    chunk_text, placeholder_maps
                )

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=expanded_text,
                    document_id=doc.document_id,
                    document_name=doc.filename,
                    page_numbers=[page.page_number],
                    chunk_index=chunk_counter,
                    total_chunks=0,
                    metadata=ChunkMetadata(
                        extraction_method=page.extraction_method,
                        has_images=len(images_used) > 0,
                        page_has_images=page.has_images,
                        has_tables=page.has_tables,
                        has_formulas=len(formulas_used) > 0,
                        document_title=doc.metadata.title,
                        document_author=", ".join(doc.metadata.author) if doc.metadata.author else None,
                        publication_id=doc.metadata.publication_id,
                        attachment_id=doc.metadata.attachment_id,
                        user_id=doc.metadata.user_id,
                        is_public=doc.metadata.is_public,
                        image_ids=[i.get('image_id') for i in images_used if i.get('image_id')],
                        image_paths=[i.get('image_path') for i in images_used if i.get('image_path')],
                        images=images_used,
                        formulas=formulas_used,
                    ),
                )

                chunks.append(chunk)
                chunk_counter += 1

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_semantic(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        chunks = []
        chunk_counter = 0

        current_blocks: List[ContentBlock] = []
        current_title: Optional[str] = None
        current_pages: set = set()

        def flush_section():
            nonlocal chunk_counter
            if current_title is None and not current_blocks:
                return

            # Traite tous les blocs d'un coup — les indices formula/image sont
            # séquentiels sur toute la section, pas remis à zéro bloc par bloc.
            if current_blocks:
                section_text, section_maps = self._build_text_with_placeholders(current_blocks)
            else:
                section_text, section_maps = "", {'formulas': {}, 'images': {}}

            full_text = "\n\n".join(part for part in [current_title, section_text] if part)
            if not full_text:
                return

            sub_chunks = (
                self.splitter.split_text(full_text)
                if len(full_text) > self.chunk_size * 1.5
                else [full_text]
            )

            for sub_chunk in sub_chunks:
                expanded_text, formulas_used, images_used = self._expand_placeholders_and_collect_metadata(
                    sub_chunk, section_maps
                )
                chunk = self._create_chunk(
                    expanded_text, doc, list(current_pages), chunk_counter,
                    section_title=current_title,
                    images_used=images_used,
                    formulas_used=formulas_used,
                )
                chunks.append(chunk)
                chunk_counter += 1

        for page in doc.pages:
            for block in page.content_blocks:
                if block.type == "title":
                    flush_section()
                    current_blocks = []
                    current_title = block.content
                    current_pages = {page.page_number}
                elif block.type in ["text", "list", "formula", "image", "table"]:
                    current_blocks.append(block)
                    current_pages.add(page.page_number)

        flush_section()

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_mixed(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        semantic_chunks = self._split_semantic(doc)
        final_chunks = []
        chunk_counter = 0

        for chunk in semantic_chunks:
            if chunk.char_count > self.chunk_size * 1.5 and not chunk.metadata.has_formulas:
                sub_texts = self.splitter.split_text(chunk.content)

                for sub_text in sub_texts:
                    new_chunk = DocumentChunk(
                        chunk_id=f"{doc.document_id}_chunk_{chunk_counter}",
                        content=sub_text,
                        document_id=doc.document_id,
                        document_name=doc.filename,
                        page_numbers=chunk.page_numbers,
                        chunk_index=chunk_counter,
                        total_chunks=0,
                        metadata=replace(chunk.metadata),
                    )
                    final_chunks.append(new_chunk)
                    chunk_counter += 1
            else:
                chunk.chunk_index = chunk_counter
                final_chunks.append(chunk)
                chunk_counter += 1

        for chunk in final_chunks:
            chunk.total_chunks = len(final_chunks)

        return final_chunks

    def _build_text_with_placeholders(self, blocks: List[ContentBlock]) -> tuple:
        text_parts = []
        formulas = {}
        images = {}
        formula_idx = 0
        image_idx = 0

        for block in blocks:
            if block.type == "title":
                prefix = "#" * (block.level or 1)
                text_parts.append(f"{prefix} {block.content}")

            elif block.type == "list":
                text_parts.append(block.content)

            elif block.type == "formula":
                token = f"[[FORMULA:{formula_idx}]]"
                formulas[token] = {
                    'latex': block.content,
                    'page_number': block.page_number,
                    'bbox': self._bbox_to_dict(block.bbox)
                }
                text_parts.append(token)
                formula_idx += 1

            elif block.type == "image":
                token = f"[[IMAGE:{image_idx}]]"
                image_id = block.image_id
                if not image_id and block.image_path:
                    image_id = Path(block.image_path).stem

                images[token] = {
                    'image_id': image_id,
                    'image_path': block.image_path,
                    'description': block.image_description or block.content or "",
                    'page_number': block.page_number,
                    'bbox': self._bbox_to_dict(block.bbox)
                }
                text_parts.append(token)
                image_idx += 1

            else:
                text_parts.append(block.content)

        return "\n\n".join(text_parts), {'formulas': formulas, 'images': images}

    def _expand_placeholders_and_collect_metadata(self, text: str, maps: Dict) -> tuple:
        formulas_used = []
        images_used = []

        for token, data in maps.get('formulas', {}).items():
            if token in text:
                text = text.replace(token, f"$$ {data.get('latex', '')} $$")
                formulas_used.append(data)

        for token, data in maps.get('images', {}).items():
            if token in text:
                image_path = data.get('image_path')
                desc = data.get('description', '').strip()
                replacement = f"[IMAGE {image_path}] {desc}" if desc else f"[IMAGE {image_path}]"
                text = text.replace(token, replacement)
                images_used.append(data)

        return text, formulas_used, images_used

    def _bbox_to_dict(self, bbox) -> Optional[Dict]:
        if bbox is None:
            return None
        return {'x0': bbox.x0, 'y0': bbox.y0, 'x1': bbox.x1, 'y1': bbox.y1, 'page': bbox.page}

    def _create_chunk(
        self,
        content: str,
        doc: ExtractedDocument,
        page_numbers: List[int],
        index: int,
        section_title: Optional[str] = None,
        images_used: Optional[List[Dict]] = None,
        formulas_used: Optional[List[Dict]] = None,
    ) -> DocumentChunk:
        images_used = images_used or []
        formulas_used = formulas_used or []
        metadata = ChunkMetadata(
            document_title=doc.metadata.title,
            document_author=", ".join(doc.metadata.author) if doc.metadata.author else None,
            publication_id=doc.metadata.publication_id,
            attachment_id=doc.metadata.attachment_id,
            user_id=doc.metadata.user_id,
            is_public=doc.metadata.is_public,
            section_title=section_title,
            has_images=len(images_used) > 0,
            has_formulas=len(formulas_used) > 0,
            image_ids=[i.get('image_id') for i in images_used if i.get('image_id')],
            image_paths=[i.get('image_path') for i in images_used if i.get('image_path')],
            images=images_used,
            formulas=formulas_used,
        )

        return DocumentChunk(
            chunk_id=f"{doc.document_id}_chunk_{index}",
            content=content,
            document_id=doc.document_id,
            document_name=doc.filename,
            page_numbers=page_numbers,
            chunk_index=index,
            total_chunks=0,
            metadata=metadata,
        )

    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        output = {
            'total_chunks': len(chunks),
            'chunks': [chunk.to_dict() for chunk in chunks]
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info("chunks sauvegardés: %s", output_path)

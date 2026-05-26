import os
import fitz  # PyMuPDF
from PIL import Image
import io
import time
import json
from pathlib import Path
from typing import List, Optional, Callable

from .document_schemas import (
    ExtractedDocument,
    PageContent,
    ContentBlock,
    BoundingBox,
    DocumentMetadata,
    TOCEntry,
    ExtractionStats
)
from .ocr_handler import DocTROCRHandler
from .preprocessor import TextPreprocessor
from .math_ocr_handler import MathOCRHandler
from rag_core.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """extracteur principal pour documents PDF"""

    def __init__(self, config: dict = None, upload_callback: Optional[Callable] = None):
        """
        upload_callback reçoit (file=bytes, public_id=str, folder=str)
        et retourne l'url de l'image hébergée.
        si pas de callback, les images sont sauvegardées localement dans temp_dir.
        """
        self.config = config or {}
        self.upload_callback = upload_callback

        self.ocr_handler = None
        self.math_ocr_handler = None
        self.preprocessor = TextPreprocessor(
            self.config.get('preprocessing', {})
        )

        self.extract_images = self.config.get('pymupdf', {}).get('extract_images', True)
        self.use_ocr_for_images = self.config.get('doctr', {}).get('use_for_images', True)
        self.use_ocr_for_scanned = self.config.get('doctr', {}).get('use_for_scanned', True)
        self.use_math_ocr = self.config.get('math_ocr', {}).get('enabled', True)
        self.math_ocr_device = self.config.get('math_ocr', {}).get('device', 'cuda')
        self.output_dir = Path(self.config.get('output_dir', 'data/extracted'))
        self.temp_dir = Path(self.config.get('temp_dir', 'data/temp'))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _init_ocr_handler(self):
        if self.ocr_handler is None:
            doctr_config = self.config.get('doctr', {})
            self.ocr_handler = DocTROCRHandler(
                det_arch=doctr_config.get('det_arch', 'db_resnet50'),
                reco_arch=doctr_config.get('reco_arch', 'crnn_vgg16_bn'),
                device=doctr_config.get('device', 'cuda'),
                pretrained=doctr_config.get('pretrained', True)
            )

    def _init_math_ocr_handler(self):
        if self.math_ocr_handler is None:
            try:
                self.math_ocr_handler = MathOCRHandler(device=self.math_ocr_device)
            except Exception as e:
                logger.warning("ocr math indisponible: %s", e)
                self.math_ocr_handler = None

    def extract_pdf(
        self,
        pdf_path: str,
        uploaded_url: str = "",
        default_metadata: dict = None,
        document_name_without_ext: str = ""
    ) -> ExtractedDocument:
        start_time = time.time()
        self._math_ocr_failures = 0

        logger.info("extraction de: %s", pdf_path)

        doc = ExtractedDocument.create_new(source_file=pdf_path, uploaded_url=uploaded_url)
        if document_name_without_ext:
            doc.filename = document_name_without_ext

        try:
            taille_mo = os.path.getsize(pdf_path) / (1024 * 1024)
            logger.debug("taille du pdf: %.2f mo", taille_mo)
            pdf_doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error("erreur ouverture pdf: %s", e)
            doc.stats.errors.append(f"erreur ouverture: {str(e)}")
            return doc

        doc.metadata = self._extract_metadata(pdf_doc, default_metadata=default_metadata, taille_mo=taille_mo)
        doc.table_of_contents = self._extract_toc(pdf_doc)

        is_scanned = self._is_scanned_pdf(pdf_doc)
        if is_scanned:
            logger.info("pdf scanné détecté, utilisation de doctr")
            if self.use_ocr_for_scanned:
                self._init_ocr_handler()

        logger.info("traitement de %d pages...", len(pdf_doc))

        for page_num in range(len(pdf_doc)):
            try:
                logger.debug("page %d/%d...", page_num + 1, len(pdf_doc))
                page_content = self._extract_page(
                    pdf_doc,
                    page_num,
                    is_scanned,
                    document_name_without_ext=document_name_without_ext
                )
                doc.pages.append(page_content)

                doc.stats.total_text_blocks += len([
                    b for b in page_content.content_blocks
                    if b.type in ["text", "title"]
                ])
                doc.stats.total_images += len([
                    b for b in page_content.content_blocks
                    if b.type == "image"
                ])
                doc.stats.total_tables += len([
                    b for b in page_content.content_blocks
                    if b.type == "table"
                ])

                if page_content.extraction_method == "ocr":
                    doc.stats.pages_with_ocr += 1

                logger.debug("page %d ok", page_num + 1)

            except Exception as e:
                logger.error("erreur page %d: %s", page_num + 1, e)
                doc.stats.errors.append(f"page {page_num + 1}: {str(e)}")

        pdf_doc.close()

        doc.stats.total_pages = len(doc.pages)
        doc.stats.processing_time_seconds = time.time() - start_time
        doc.stats.math_ocr_failures = self._math_ocr_failures

        if self._math_ocr_failures > 0:
            logger.warning(
                "%d bloc(s) mathématiques non convertis en LaTeX (math OCR échoué ou indisponible)",
                self._math_ocr_failures
            )

        logger.info(
            "extraction terminée en %.2fs — pages: %d, texte: %d, images: %d, tableaux: %d, ocr: %d, math_echecs: %d",
            doc.stats.processing_time_seconds, doc.stats.total_pages,
            doc.stats.total_text_blocks, doc.stats.total_images,
            doc.stats.total_tables, doc.stats.pages_with_ocr,
            doc.stats.math_ocr_failures
        )

        return doc

    def _extract_metadata(self, pdf_doc, default_metadata=None, taille_mo=None) -> DocumentMetadata:
        metadata = dict(pdf_doc.metadata) if pdf_doc.metadata else {}

        authors = []
        if default_metadata and default_metadata.get('author'):
            authors = default_metadata['author']
        elif metadata.get('author'):
            author_val = metadata['author']
            if isinstance(author_val, list):
                authors = author_val
            elif isinstance(author_val, str) and author_val.strip():
                authors = [author_val.strip()]

        if default_metadata:
            if (not metadata.get('title') or metadata.get('title') == '') and default_metadata.get('title'):
                metadata['title'] = default_metadata['title']
            if (not metadata.get('subject') or metadata.get('subject') == '') and default_metadata.get('subject'):
                metadata['subject'] = default_metadata['subject']
            if (not metadata.get('keywords') or metadata.get('keywords') == '') and default_metadata.get('keywords'):
                metadata['keywords'] = ','.join(default_metadata['keywords'])

        return DocumentMetadata(
            title=metadata.get('title'),
            author=authors,
            subject=metadata.get('subject'),
            keywords=metadata.get('keywords', '').split(',') if metadata.get('keywords') else [],
            creation_date=metadata.get('creationDate'),
            modification_date=metadata.get('modDate'),
            num_pages=len(pdf_doc),
            producer=metadata.get('producer'),
            language=None,
            file_size=taille_mo,
            publication_id=default_metadata.get('publication_id') if default_metadata else None,
            attachment_id=default_metadata.get('attachment_id') if default_metadata else None,
            user_id=default_metadata.get('user_id') if default_metadata else None,
            is_public=default_metadata.get('is_public', False) if default_metadata else False
        )

    def _extract_toc(self, pdf_doc) -> List[TOCEntry]:
        toc = []
        try:
            for entry in pdf_doc.get_toc():
                level, title, page = entry
                toc.append(TOCEntry(title=title, level=level, page=page))
        except Exception:
            pass
        return toc

    def _is_scanned_pdf(self, pdf_doc, sample_pages: int = 3) -> bool:
        pages_to_check = min(sample_pages, len(pdf_doc))
        text_found = 0

        for page_num in range(pages_to_check):
            page = pdf_doc[page_num]
            if len(page.get_text().strip()) > 100:
                text_found += 1

        return text_found < (pages_to_check / 2)

    def _extract_page(
        self,
        pdf_doc,
        page_num: int,
        is_scanned: bool,
        document_name_without_ext: str,
    ) -> PageContent:
        page = pdf_doc[page_num]
        blocks = []
        extraction_method = "pymupdf"

        if is_scanned and self.use_ocr_for_scanned and self.ocr_handler:
            logger.debug("page %d traitée avec doctr", page_num + 1)
            blocks = self._extract_with_ocr(page, page_num)
            extraction_method = "ocr"
        else:
            blocks = self._extract_with_pymupdf(page, page_num, document_name_without_ext=document_name_without_ext)

        blocks = self.preprocessor.preprocess_blocks(blocks)
        page_text = page.get_text()

        return PageContent(
            page_number=page_num + 1,
            content_blocks=blocks,
            page_text=page_text,
            has_images=any(b.type == "image" for b in blocks),
            has_tables=any(b.type == "table" for b in blocks),
            extraction_method=extraction_method
        )

    def _extract_with_pymupdf(self, page, page_num: int, document_name_without_ext: str) -> List[ContentBlock]:
        blocks = []
        text_blocks = page.get_text("dict")["blocks"]

        for block_idx, block in enumerate(text_blocks):
            if block["type"] == 0:
                bbox = BoundingBox(
                    x0=block["bbox"][0],
                    y0=block["bbox"][1],
                    x1=block["bbox"][2],
                    y1=block["bbox"][3],
                    page=page_num + 1
                )

                text_content = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_content += span.get("text", "") + " "

                text_content = text_content.strip()

                if text_content:
                    if self.use_math_ocr and self._is_likely_math_block(block, text_content):
                        if not self.math_ocr_handler:
                            self._init_math_ocr_handler()

                        if self.math_ocr_handler:
                            formula_block = self._extract_formula_from_block(page, block, page_num)
                            if formula_block:
                                blocks.append(formula_block)
                                continue
                            else:
                                self._math_ocr_failures += 1
                        else:
                            self._math_ocr_failures += 1

                    is_title = self._is_likely_title(block)

                    blocks.append(ContentBlock(
                        type="title" if is_title else "text",
                        content=text_content,
                        page_number=page_num + 1,
                        bbox=bbox,
                        level=1 if is_title else None
                    ))

            elif block["type"] == 1:
                if self.extract_images:
                    logger.debug("image détectée page %d", page_num + 1)
                    image_block = self._extract_image(
                        page,
                        block,
                        page_num,
                        block_idx,
                        document_name_without_ext=document_name_without_ext,
                    )
                    if image_block:
                        blocks.append(image_block)

        tables = page.find_tables()
        for table in tables.tables:
            table_text = self._extract_table_text(table)
            blocks.append(ContentBlock(
                type="table",
                content=table_text,
                page_number=page_num + 1
            ))

        return blocks

    def _extract_with_ocr(self, page, page_num: int) -> List[ContentBlock]:
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        return self.ocr_handler.process_page_image(image, page_num + 1)

    def _extract_image(
        self,
        page,
        image_block: dict,
        page_num: int,
        img_idx: int,
        document_name_without_ext: str,
    ) -> Optional[ContentBlock]:
        try:
            bbox = BoundingBox(
                x0=image_block["bbox"][0],
                y0=image_block["bbox"][1],
                x1=image_block["bbox"][2],
                y1=image_block["bbox"][3],
                page=page_num + 1
            )

            xref = image_block.get("xref", None)
            if xref is None:
                xref = image_block.get("image")

            image = None
            if isinstance(xref, int):
                base_image = page.parent.extract_image(xref)
                image_data = base_image.get("image")
                if image_data:
                    image = Image.open(io.BytesIO(image_data))
            else:
                # xref invalide, on rasterise la zone directement
                clip_rect = fitz.Rect(
                    image_block["bbox"][0],
                    image_block["bbox"][1],
                    image_block["bbox"][2],
                    image_block["bbox"][3]
                )
                pix = page.get_pixmap(clip=clip_rect, dpi=300)
                image = Image.open(io.BytesIO(pix.tobytes("png")))

            if image is None:
                raise ValueError("impossible d'extraire l'image")

            image_id = f"img_{page_num}_{img_idx}"
            image_url = None

            if self.upload_callback:
                try:
                    logger.debug("upload de l'image %s...", image_id)
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    image_url = self.upload_callback(
                        file=img_bytes,
                        public_id=image_id,
                        folder=f"rag-images/{document_name_without_ext}"
                    )
                    logger.debug("image uploadée: %s", image_url)
                except Exception as e:
                    logger.warning("échec upload image %s, sauvegarde locale", e)
                    image_path = self.temp_dir / f"{image_id}.png"
                    image.save(image_path)
                    image_url = str(image_path)
            else:
                # pas de callback configuré, on sauvegarde en local
                image_path = self.temp_dir / f"{image_id}.png"
                image.save(image_path)
                image_url = str(image_path)

            description = ""
            if not self.ocr_handler:
                self._init_ocr_handler()
            if self.use_ocr_for_images and self.ocr_handler:
                desc_block = self.ocr_handler.process_image_for_description(image, page_num + 1, image_id)
                description = desc_block.image_description or ""

            return ContentBlock(
                type="image",
                content=description,
                page_number=page_num + 1,
                bbox=bbox,
                image_id=image_id,
                image_description=description,
                image_path=image_url
            )

        except Exception as e:
            logger.error("erreur extraction image: %s", e)
            return None

    def _is_likely_title(self, block: dict) -> bool:
        if not block.get("lines"):
            return False
        first_line = block["lines"][0]
        if first_line.get("spans"):
            font_size = first_line["spans"][0].get("size", 0)
            return font_size > 14
        return False

    def _extract_table_text(self, table) -> str:
        try:
            rows = table.extract()
            table_text = []
            for row in rows:
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                table_text.append(row_text)
            return "\n".join(table_text)
        except Exception:
            return ""

    def _is_likely_math_block(self, block: dict, text: str) -> bool:
        if not text or len(text) > 200:
            return False

        math_chars = set("=<>+-*/×÷∑∫√≈≠≤≥∞πσμθλΔΩαβγδεζηικνξοπρστυφχψω∂∇^_")
        math_char_count = sum(1 for ch in text if ch in math_chars)
        ratio = math_char_count / max(1, len(text))

        font_hit = False
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_name = (span.get("font") or "").lower()
                if "math" in font_name or "symbol" in font_name:
                    font_hit = True
                    break
            if font_hit:
                break

        has_basic_equation = ("=" in text) or ("≤" in text) or ("≥" in text)

        return ratio >= 0.05 or font_hit or has_basic_equation

    def _extract_formula_from_block(self, page, block: dict, page_num: int) -> Optional[ContentBlock]:
        try:
            bbox = block["bbox"]
            clip_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            pix = page.get_pixmap(clip=clip_rect, dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png")))

            latex = self.math_ocr_handler.image_to_latex(image)
            if not latex:
                return None

            raw_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    raw_text += span.get("text", "") + " "
            raw_text = raw_text.strip()

            return ContentBlock(
                type="formula",
                content=latex,
                page_number=page_num + 1,
                bbox=BoundingBox(
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    y1=bbox[3],
                    page=page_num + 1
                ),
                metadata={"raw_text": raw_text}
            )
        except Exception:
            return None

    def save_document(self, doc: ExtractedDocument) -> str:
        output_file = self.output_dir / f"{doc.document_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("document sauvegardé: %s", output_file)
        return str(output_file)

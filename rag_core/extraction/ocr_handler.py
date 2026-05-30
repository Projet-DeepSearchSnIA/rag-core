from PIL import Image
import numpy as np
from typing import List
import re

from .document_schemas import ContentBlock, BoundingBox
from rag_core.utils.logger import get_logger

logger = get_logger(__name__)


class DocTROCRHandler:
    """handler docTR pour les PDFs scannés et les images"""

    def __init__(
        self,
        det_arch: str,
        reco_arch: str,
        device: str,
        pretrained: bool,
    ):
        import torch
        from doctr.models import ocr_predictor

        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            logger.warning("device demandé '%s' indisponible, fallback sur '%s'", device, self.device)

        try:
            self.model = ocr_predictor(
                det_arch=det_arch,
                reco_arch=reco_arch,
                pretrained=pretrained,
                assume_straight_pages=True
            )

            if self.device == "cuda":
                self.model.det_predictor.model.to(self.device)
                self.model.reco_predictor.model.to(self.device)

        except Exception as e:
            raise RuntimeError(f"erreur chargement docTR: {e}") from e

    def process_image(
        self,
        image: Image.Image,
        page_number: int,
    ) -> List[ContentBlock]:
        try:
            img_array = np.array(image)
            result = self.model([img_array])
            if not result.pages:
                return []
            return self._parse_doctr_page(result.pages[0], page_number)
        except Exception as e:
            logger.error("erreur traitement image: %s", e)
            return []

    def process_page_image(
        self,
        page_image: Image.Image,
        page_number: int
    ) -> List[ContentBlock]:
        return self.process_image(page_image, page_number)

    def process_image_for_description(
        self,
        image: Image.Image,
        page_number: int,
        image_id: str
    ) -> ContentBlock:
        blocks = self.process_image(image, page_number)
        description = " ".join([b.content for b in blocks if b.type == "text"])

        return ContentBlock(
            type="image",
            content=description,
            page_number=page_number,
            image_id=image_id,
            image_description=description
        )

    def _parse_doctr_page(
        self,
        page,
        page_number: int,
    ) -> List[ContentBlock]:
        if page is None:
            return []

        blocks = []

        for block in page.blocks:
            block_text_lines = []
            block_bbox = None

            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                block_text_lines.append(line_text)

                if block_bbox is None and line.words:
                    block_bbox = line.words[0].geometry

            block_text = " ".join(block_text_lines).strip()

            if not block_text:
                continue

            bbox = None
            if block_bbox is not None:
                bbox = BoundingBox(
                    x0=float(block_bbox[0][0]),
                    y0=float(block_bbox[0][1]),
                    x1=float(block_bbox[1][0]),
                    y1=float(block_bbox[1][1]),
                    page=page_number
                )

            content_type, level = self._detect_content_type(block_text)

            blocks.append(ContentBlock(
                type=content_type,
                content=block_text,
                page_number=page_number,
                bbox=bbox,
                level=level if content_type == "title" else None
            ))

        return blocks

    def _detect_content_type(self, text: str) -> tuple:
        text_stripped = text.strip()

        if len(text_stripped) < 100:
            if text_stripped.isupper() and len(text_stripped.split()) <= 10:
                return ("title", 1)
            if re.match(r'^(\d+\.|\d+\.\d+|[IVX]+\.)\s+', text_stripped):
                return ("title", 2)

        if re.match(r'^[\*\-\+•]\s+', text_stripped) or re.match(r'^\d+\.\s+', text_stripped):
            return ("list", None)

        if text_stripped.count('|') >= 2 or text_stripped.count('\t') >= 2:
            return ("table", None)

        return ("text", None)
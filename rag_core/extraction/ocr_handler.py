import torch
from doctr.models import ocr_predictor
from PIL import Image
import numpy as np
from typing import List, Optional
import re

from .document_schemas import ContentBlock, BoundingBox


class DocTROCRHandler:
    """handler docTR pour les PDFs scannés et les images"""

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        device: str = "cuda",
        pretrained: bool = True
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"chargement de docTR sur {self.device}...")

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

            print("docTR chargé")
        except Exception as e:
            print(f"erreur chargement docTR: {e}")
            raise

    def process_image(
        self,
        image: Image.Image,
        page_number: int,
        prompt_type: str = "ocr_layout",
        image_id: Optional[str] = None
    ) -> List[ContentBlock]:
        try:
            img_array = np.array(image)
            result = self.model([img_array])
            return self._parse_doctr_result(result, page_number, image_id)
        except Exception as e:
            print(f"erreur traitement image: {e}")
            return []

    def process_page_image(
        self,
        page_image: Image.Image,
        page_number: int
    ) -> List[ContentBlock]:
        return self.process_image(page_image, page_number, prompt_type="ocr_layout")

    def process_image_for_description(
        self,
        image: Image.Image,
        page_number: int,
        image_id: str
    ) -> ContentBlock:
        blocks = self.process_image(image, page_number, prompt_type="caption", image_id=image_id)
        description = " ".join([b.content for b in blocks if b.type == "text"])

        return ContentBlock(
            type="image",
            content=description,
            page_number=page_number,
            image_id=image_id,
            image_description=description
        )

    def _parse_doctr_result(
        self,
        result,
        page_number: int,
        image_id: Optional[str] = None
    ) -> List[ContentBlock]:
        blocks = []

        if len(result.pages) == 0:
            return blocks

        page = result.pages[0]

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

    def batch_process_images(
        self,
        images: List[tuple],
        batch_size: int = 4
    ) -> List[List[ContentBlock]]:
        all_blocks = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_images = []
            batch_metadata = []

            for image, page_num, img_id in batch:
                batch_images.append(np.array(image))
                batch_metadata.append((page_num, img_id))

            try:
                results = self.model(batch_images)

                for result, (page_num, img_id) in zip(results.pages, batch_metadata):
                    blocks = self._parse_doctr_result(
                        type('Result', (), {'pages': [result]})(),
                        page_num,
                        img_id
                    )
                    all_blocks.append(blocks)

            except Exception as e:
                print(f"erreur batch: {e}, traitement individuel")
                for image, page_num, img_id in batch:
                    blocks = self.process_image(image, page_num, image_id=img_id)
                    all_blocks.append(blocks)

        return all_blocks

    def export_to_text(self, blocks: List[ContentBlock]) -> str:
        text_lines = []

        for block in blocks:
            if block.type == "title":
                prefix = "#" * (block.level or 1)
                text_lines.append(f"{prefix} {block.content}\n")
            elif block.type == "list":
                text_lines.append(f"• {block.content}")
            else:
                text_lines.append(block.content)

            text_lines.append("")

        return "\n".join(text_lines)

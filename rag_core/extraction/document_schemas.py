from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Literal
from datetime import datetime
import uuid


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


@dataclass
class ContentBlock:
    type: Literal["text", "title", "image", "table", "list", "code", "formula"]
    content: str
    page_number: int
    bbox: Optional[BoundingBox] = None
    metadata: Optional[Dict] = None

    level: Optional[int] = None

    image_id: Optional[str] = None
    image_description: Optional[str] = None
    image_caption: Optional[str] = None
    image_path: Optional[str] = None

    table_structure: Optional[Dict] = None


@dataclass
class PageContent:
    page_number: int
    content_blocks: List[ContentBlock]
    page_text: str
    has_images: bool = False
    has_tables: bool = False
    extraction_method: str = "pymupdf"
    confidence_score: Optional[float] = None


@dataclass
class TOCEntry:
    title: str
    level: int
    page: int


@dataclass
class DocumentMetadata:
    title: Optional[str] = None
    author: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    num_pages: int = 0
    language: Optional[str] = None
    producer: Optional[str] = None
    file_size: Optional[int] = None
    publication_id: Optional[int] = None
    attachment_id: Optional[int] = None
    user_id: Optional[int] = None
    is_public: bool = False


@dataclass
class ExtractionStats:
    total_pages: int
    total_text_blocks: int
    total_images: int
    total_tables: int
    pages_with_ocr: int
    processing_time_seconds: float
    extraction_method: str
    errors: List[str] = field(default_factory=list)
    math_ocr_failures: int = 0


@dataclass
class ExtractedDocument:
    document_id: str
    source_file: str
    filename: str
    extraction_date: str
    metadata: DocumentMetadata
    pages: List[PageContent]
    table_of_contents: List[TOCEntry]
    stats: ExtractionStats

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def create_new(source_file: str, uploaded_url: str):
        import os
        return ExtractedDocument(
            document_id=str(uuid.uuid4()),
            source_file=uploaded_url,
            filename=os.path.basename(source_file.replace('\\', '/')),
            extraction_date=datetime.now().isoformat(),
            metadata=DocumentMetadata(),
            pages=[],
            table_of_contents=[],
            stats=ExtractionStats(
                total_pages=0,
                total_text_blocks=0,
                total_images=0,
                total_tables=0,
                pages_with_ocr=0,
                processing_time_seconds=0.0,
                extraction_method="hybrid"
            )
        )

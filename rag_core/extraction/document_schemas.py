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

    @classmethod
    def from_dict(cls, data: Dict) -> "ExtractedDocument":
        """Reconstruit un ExtractedDocument depuis un dict (inverse de to_dict)."""
        meta = data.get("metadata") or {}
        stats = data.get("stats") or {}
        pages = []
        for p in data.get("pages") or []:
            blocks = []
            for b in p.get("content_blocks") or []:
                bbox_data = b.get("bbox")
                blocks.append(ContentBlock(
                    type=b["type"], content=b["content"], page_number=b["page_number"],
                    bbox=BoundingBox(**bbox_data) if bbox_data else None,
                    metadata=b.get("metadata"), level=b.get("level"),
                    image_id=b.get("image_id"), image_description=b.get("image_description"),
                    image_caption=b.get("image_caption"), image_path=b.get("image_path"),
                    table_structure=b.get("table_structure"),
                ))
            pages.append(PageContent(
                page_number=p["page_number"], content_blocks=blocks,
                page_text=p.get("page_text", ""),
                has_images=p.get("has_images", False),
                has_tables=p.get("has_tables", False),
                extraction_method=p.get("extraction_method", "pymupdf"),
                confidence_score=p.get("confidence_score"),
            ))
        return cls(
            document_id=data["document_id"], source_file=data["source_file"],
            filename=data["filename"], extraction_date=data["extraction_date"],
            metadata=DocumentMetadata(**meta),
            pages=pages,
            table_of_contents=[TOCEntry(**t) for t in (data.get("table_of_contents") or [])],
            stats=ExtractionStats(**stats),
        )

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

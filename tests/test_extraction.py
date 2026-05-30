from tests.conftest import load_baseline

from rag_core.extraction.pdf_extractor import PDFExtractor
from rag_core.extraction.document_schemas import (
    ExtractedDocument, ContentBlock, BoundingBox, DocumentMetadata
)


def _load_extraction_cfg() -> dict:
    return load_baseline()["extraction"]


def test_pdf_extractor_init_sans_callback():
    extractor = PDFExtractor(config=_load_extraction_cfg())
    assert extractor.upload_callback is None


def test_pdf_extractor_init_avec_callback():
    def fake_upload(**kwargs):
        return "http://fake-url/image.png"

    extractor = PDFExtractor(config=_load_extraction_cfg(), upload_callback=fake_upload)
    assert extractor.upload_callback is fake_upload


def test_extracted_document_create_new():
    doc = ExtractedDocument.create_new(source_file="test.pdf", uploaded_url="http://example.com/test.pdf")
    assert doc.filename == "test.pdf"
    assert doc.source_file == "http://example.com/test.pdf"
    assert doc.pages == []
    assert doc.stats.total_pages == 0


def test_extracted_document_id_unique():
    doc1 = ExtractedDocument.create_new(source_file="a.pdf", uploaded_url="http://x.com/a.pdf")
    doc2 = ExtractedDocument.create_new(source_file="b.pdf", uploaded_url="http://x.com/b.pdf")
    assert doc1.document_id != doc2.document_id


def test_content_block_optionnel():
    block = ContentBlock(type="text", content="bonjour", page_number=1)
    assert block.bbox is None
    assert block.metadata is None
    assert block.image_id is None


def test_bounding_box():
    bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=50.0, page=1)
    assert bbox.x1 == 100.0
    assert bbox.page == 1


def test_document_metadata_defaut():
    meta = DocumentMetadata()
    assert meta.title is None
    assert meta.author == []
    assert meta.num_pages == 0
    assert meta.is_public is False


def test_extractor_callback_non_appele_sans_image():
    appels = []

    def fake_upload(**kwargs):
        appels.append(kwargs)
        return "http://fake/img.png"

    PDFExtractor(config=_load_extraction_cfg(), upload_callback=fake_upload)
    assert appels == [], "le callback ne doit pas être appelé à l'initialisation"

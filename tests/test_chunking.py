from rag_core.chunking.text_splitter import SmartTextSplitter, DocumentChunk
from rag_core.extraction.document_schemas import (
    ExtractedDocument, PageContent, ContentBlock, DocumentMetadata, ExtractionStats
)
import uuid


def _make_doc(pages_text: list[str]) -> ExtractedDocument:
    pages = []
    for i, text in enumerate(pages_text, start=1):
        block = ContentBlock(type="text", content=text, page_number=i)
        pages.append(PageContent(
            page_number=i,
            content_blocks=[block],
            page_text=text
        ))
    return ExtractedDocument(
        document_id=str(uuid.uuid4()),
        source_file="http://exemple.com/doc.pdf",
        filename="doc.pdf",
        extraction_date="2026-01-01T00:00:00",
        metadata=DocumentMetadata(),
        pages=pages,
        table_of_contents=[],
        stats=ExtractionStats(
            total_pages=len(pages_text),
            total_text_blocks=len(pages_text),
            total_images=0,
            total_tables=0,
            pages_with_ocr=0,
            processing_time_seconds=0.0,
            extraction_method="hybrid"
        )
    )


def test_chunk_document_simple():
    doc = _make_doc(["Ceci est un texte simple pour tester le chunking."])
    splitter = SmartTextSplitter(chunk_size=100, chunk_overlap=0, strategy="recursive")
    chunks = splitter.split_document(doc)
    assert len(chunks) >= 1
    assert all(isinstance(c, DocumentChunk) for c in chunks)


def test_chunk_preserve_document_id():
    doc = _make_doc(["Un texte quelconque."])
    splitter = SmartTextSplitter(chunk_size=500, strategy="recursive")
    chunks = splitter.split_document(doc)
    for chunk in chunks:
        assert chunk.document_id == doc.document_id


def test_chunk_total_chunks_coherent():
    doc = _make_doc(["a " * 300])
    splitter = SmartTextSplitter(chunk_size=100, chunk_overlap=0, strategy="recursive")
    chunks = splitter.split_document(doc)
    expected_total = len(chunks)
    for chunk in chunks:
        assert chunk.total_chunks == expected_total


def test_chunk_doc_vide():
    doc = _make_doc([])
    splitter = SmartTextSplitter(strategy="recursive")
    chunks = splitter.split_document(doc)
    assert chunks == []


def test_chunk_to_dict_complet():
    doc = _make_doc(["Du contenu."])
    splitter = SmartTextSplitter(strategy="recursive")
    chunks = splitter.split_document(doc)
    d = chunks[0].to_dict()
    assert "chunk_id" in d
    assert "content" in d
    assert "page_numbers" in d
    assert "char_count" in d


def test_strategy_mixed():
    doc = _make_doc(["Section 1", "Contenu de la section 1.", "Section 2", "Contenu de la section 2."])
    splitter = SmartTextSplitter(chunk_size=500, strategy="mixed")
    chunks = splitter.split_document(doc)
    assert len(chunks) >= 1


def test_splitter_defaut_recursive():
    splitter = SmartTextSplitter()
    assert splitter.strategy == "recursive"
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200

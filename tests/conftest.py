"""
Fixtures partagées entre tous les fichiers de test.

On centralise ici la construction d'ExtractedDocument synthétique pour ne
pas réécrire le même boilerplate dans chaque fichier.
"""
import uuid
import pytest

from rag_core.extraction.document_schemas import (
    ContentBlock,
    DocumentMetadata,
    ExtractedDocument,
    ExtractionStats,
    PageContent,
)
from rag_core.chunking.text_splitter import DocumentChunk


def make_doc(pages_text: list[str]) -> ExtractedDocument:
    """Construit un ExtractedDocument minimal à partir d'une liste de textes.

    Chaque élément de la liste devient une page avec un seul bloc texte.
    Pratique pour tester le chunking sans toucher au vrai PDF extractor.
    """
    pages = []
    for i, text in enumerate(pages_text, start=1):
        block = ContentBlock(type="text", content=text, page_number=i)
        pages.append(
            PageContent(
                page_number=i,
                content_blocks=[block],
                page_text=text,
            )
        )
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
            extraction_method="hybrid",
        ),
    )


def make_chunk(
    content: str,
    document_id: str = "doc-test",
    page_numbers: list[int] = None,
    metadata: dict = None,
) -> DocumentChunk:
    """Construit un DocumentChunk minimal pour tester l'optimiseur sans passer par le splitter."""
    return DocumentChunk(
        chunk_id=f"{document_id}_chunk_0",
        content=content,
        document_id=document_id,
        document_name="doc.pdf",
        page_numbers=page_numbers or [1],
        chunk_index=0,
        total_chunks=1,
        metadata=metadata or {},
    )


@pytest.fixture
def doc_simple():
    """Un document d'une page avec un texte court."""
    return make_doc(["Ceci est un texte simple pour tester."])


@pytest.fixture
def doc_multi_pages():
    """Un document de trois pages avec contenu varié."""
    return make_doc([
        "Introduction au machine learning et à ses applications.",
        "Les réseaux de neurones sont des modèles inspirés du cerveau humain.",
        "Conclusion et perspectives pour la recherche future.",
    ])


@pytest.fixture
def doc_long():
    """Un document avec beaucoup de texte pour forcer plusieurs chunks."""
    # ~600 chars, suffisant pour dépasser chunk_size=100
    return make_doc(["mot " * 150])
"""
Fixtures partagées entre tous les fichiers de test.

On centralise ici la construction d'ExtractedDocument synthétique pour ne
pas réécrire le même boilerplate dans chaque fichier.

Fixtures live (marqueur @pytest.mark.live) :
  pinecone_creds  — clés Pinecone lues depuis .env, skip automatique si absentes
  hf_token        — token HuggingFace lu depuis .env, skip automatique si absent
  live_retriever  — PineconeRetriever connecté à l'index réel
  live_llm        — LLMHandler connecté à HuggingFace
"""
import logging
import os
import uuid
import pytest
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

from rag_core.extraction.document_schemas import (
    ContentBlock,
    DocumentMetadata,
    ExtractedDocument,
    ExtractionStats,
    PageContent,
)
from rag_core.chunking.text_splitter import DocumentChunk
from rag_core.retrieval.retriever import PineconeRetriever


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


def _retriever_vide():
    """Instance PineconeRetriever sans appel réseau — bypasse __init__ via object.__new__."""
    return object.__new__(PineconeRetriever)


# ---------------------------------------------------------------------------
# Fixtures live — skip automatique si les clés .env sont absentes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pinecone_creds():
    """
    Retourne (api_key, index_name) depuis .env.
    Skip le test si l'une ou l'autre est absente.
    Usage : def test_foo(pinecone_creds): api_key, index_name = pinecone_creds
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key:
        pytest.skip("PINECONE_API_KEY absente du .env")
    if not index_name:
        pytest.skip("PINECONE_INDEX_NAME absente du .env")
    return api_key, index_name


@pytest.fixture(scope="session")
def hf_token():
    """
    Retourne le token HuggingFace depuis .env.
    Skip le test si absent.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN absent du .env")
    return token


@pytest.fixture(scope="session")
def live_retriever(pinecone_creds):
    """
    PineconeRetriever connecté à l'index réel.
    Scope session : la connexion est ouverte une seule fois pour toute la session de tests.
    """
    from rag_core.retrieval.retriever import PineconeRetriever
    api_key, index_name = pinecone_creds
    embed_model = os.getenv("PINECONE_EMBED_MODEL")
    rerank_model = os.getenv("PINECONE_RERANK_MODEL")
    namespace = os.getenv("PINECONE_NAMESPACE")
    if not embed_model:
        logger.error("PINECONE_EMBED_MODEL absente du .env")
        pytest.skip("PINECONE_EMBED_MODEL absente du .env")
    if not rerank_model:
        logger.error("PINECONE_RERANK_MODEL absente du .env")
        pytest.skip("PINECONE_RERANK_MODEL absente du .env")
    if not namespace:
        logger.error("PINECONE_NAMESPACE absente du .env")
        pytest.skip("PINECONE_NAMESPACE absente du .env")
    assert embed_model and rerank_model and namespace
    return PineconeRetriever(
        api_key=api_key,
        index_name=index_name,
        embed_model=embed_model,
        rerank_model=rerank_model,
        namespace=namespace,
    )


@pytest.fixture(scope="session")
def live_llm(hf_token):
    """
    LLMHandler connecté à HuggingFace.
    Scope session : le client est instancié une seule fois.
    """
    from rag_core.generation.llm_handler import LLMHandler
    model = os.getenv("LLM_MODEL")
    if not model:
        logger.error("LLM_MODEL absente du .env")
        pytest.skip("LLM_MODEL absente du .env")
    assert model
    return LLMHandler(model_name=model, api_key=hf_token)
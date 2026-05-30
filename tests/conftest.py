"""
Fixtures partagées entre tous les fichiers de test.

Centralise la construction d'ExtractedDocument synthétique et la lecture de la
config baseline.yaml pour les tests live.

Convention :
  - secrets (PINECONE_API_KEY, HF_TOKEN) -> .env
  - déploiement (PINECONE_INDEX_NAME, PINECONE_NAMESPACE) -> .env
  - tout le reste (modèles, cloud, region) -> configs/baseline.yaml
"""
import logging
import os
import uuid
from pathlib import Path

import pytest
import yaml
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


BASELINE_PATH = Path(__file__).parent.parent / "configs" / "baseline.yaml"


def load_baseline() -> dict:
    """Lit configs/baseline.yaml — la source unique de vérité pour les modèles et hyperparamètres."""
    with open(BASELINE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
# Fixtures live — skip automatique si les clés .env ou YAML manquent
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def baseline_cfg():
    """Config YAML chargée une fois pour la session."""
    return load_baseline()


@pytest.fixture(scope="session")
def pinecone_creds():
    """
    Retourne (api_key, index_name) depuis .env.
    Skip le test si l'une ou l'autre est absente.
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
    """Token HuggingFace depuis .env, skip si absent."""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN absent du .env")
    return token


@pytest.fixture(scope="session")
def live_retriever(pinecone_creds, baseline_cfg):
    """PineconeRetriever connecté à l'index réel."""
    from rag_core.retrieval.retriever import PineconeRetriever
    api_key, index_name = pinecone_creds
    embed_model = baseline_cfg.get("embedding", {}).get("model")
    rerank_model = baseline_cfg.get("retrieval", {}).get("rerank_model")
    namespace = os.getenv("PINECONE_NAMESPACE")
    if not embed_model:
        logger.error("embedding.model absent de configs/baseline.yaml")
        pytest.skip("embedding.model absent de baseline.yaml")
    if not rerank_model:
        logger.error("retrieval.rerank_model absent de configs/baseline.yaml")
        pytest.skip("retrieval.rerank_model absent de baseline.yaml")
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
def live_llm(hf_token, baseline_cfg):
    """LLMHandler connecté à HuggingFace."""
    from rag_core.generation.llm_handler import LLMHandler
    model = baseline_cfg.get("generation", {}).get("model")
    if not model:
        logger.error("generation.model absent de configs/baseline.yaml")
        pytest.skip("generation.model absent de baseline.yaml")
    assert model
    return LLMHandler(model_name=model, api_key=hf_token)

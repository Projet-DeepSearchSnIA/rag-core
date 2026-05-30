"""
CLI unifiée du pipeline RAG.

Six sous-commandes qui correspondent chacune à une étape isolable du pipeline,
plus un raccourci `index` pour exécuter tout le chemin d'ingestion.

Exemples :
    # Extraction seule
    python scripts/rag.py extract mon.pdf --config configs/baseline.yaml --out data/extracted/mon.json

    # Chunking d'un document déjà extrait
    python scripts/rag.py chunk data/extracted/mon.json --config configs/baseline.yaml --out chunks.json

    # Upload de chunks vers Pinecone
    python scripts/rag.py upload chunks.json --config configs/baseline.yaml --index mon-index --namespace default

    # Pipeline complet d'ingestion
    python scripts/rag.py index mon.pdf --config configs/baseline.yaml --index mon-index --namespace default

    # Retrieval seul
    python scripts/rag.py retrieve "ma question" --config configs/baseline.yaml --index mon-index --namespace default --out hits.json

    # Question/réponse RAG complète
    python scripts/rag.py ask "ma question" --config configs/baseline.yaml --index mon-index --namespace default

Aucune valeur n'a de défaut : tous les arguments sont explicites.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rag_core.utils.logger import get_logger

logger = get_logger("rag.cli")

load_dotenv()


# ---------------------------------------------------------------------------
# helpers de validation : tout est explicite, aucun défaut silencieux
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        logger.error("fichier de config introuvable : %s", p)
        sys.exit(2)
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_section(cfg: dict, section: str) -> dict:
    sub = cfg.get(section)
    if not sub:
        logger.error("section [%s] manquante dans le fichier de config", section)
        sys.exit(2)
    return sub


def require_key(cfg: dict, key: str, section: str):
    if key not in cfg or cfg[key] is None:
        logger.error("clé '%s' manquante dans la section [%s] du fichier de config", key, section)
        sys.exit(2)
    return cfg[key]


def require_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        logger.error("variable d'environnement %s absente (vérifie ton .env)", var)
        sys.exit(2)
    return val


# ---------------------------------------------------------------------------
# extract : PDF -> JSON extrait
# ---------------------------------------------------------------------------
def cmd_extract(args: argparse.Namespace) -> int:
    from rag_core.extraction.pdf_extractor import PDFExtractor

    cfg = load_config(args.config)
    extraction_cfg = require_section(cfg, "extraction")

    extractor = PDFExtractor(config=extraction_cfg)
    logger.info("extraction de %s", args.pdf_path)
    doc = extractor.extract_pdf(args.pdf_path, uploaded_url=args.pdf_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(
        "extraction terminée — %d pages, %d blocs texte, %d images, %d tables → %s",
        doc.stats.total_pages, doc.stats.total_text_blocks,
        doc.stats.total_images, doc.stats.total_tables, out,
    )
    return 0


# ---------------------------------------------------------------------------
# chunk : JSON extrait -> JSON chunks
# ---------------------------------------------------------------------------
def cmd_chunk(args: argparse.Namespace) -> int:
    from rag_core.chunking.chunk_optimizer import ChunkOptimizer
    from rag_core.chunking.text_splitter import SmartTextSplitter
    from rag_core.extraction.document_schemas import ExtractedDocument

    cfg = load_config(args.config)
    chunk_cfg = require_section(cfg, "chunking")

    with open(args.extracted_json, encoding="utf-8") as f:
        doc_dict = json.load(f)
    doc = ExtractedDocument.from_dict(doc_dict)

    splitter = SmartTextSplitter(
        chunk_size=require_key(chunk_cfg, "chunk_size", "chunking"),
        chunk_overlap=require_key(chunk_cfg, "chunk_overlap", "chunking"),
        strategy=require_key(chunk_cfg, "strategy", "chunking"),
    )
    chunks = splitter.split_document(doc)
    logger.info("%d chunks produits par le splitter", len(chunks))

    if require_key(chunk_cfg, "optimizer_enabled", "chunking"):
        optimizer = ChunkOptimizer()
        chunks, stats = optimizer.optimize_chunks(chunks)
        logger.info(
            "optimiseur : %d -> %d chunks (fusionnés=%d, splittés=%d)",
            stats.get("initial_count", 0), stats.get("final_count", len(chunks)),
            stats.get("merged", 0), stats.get("split", 0),
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splitter.save_chunks(chunks, str(out))
    logger.info("%d chunks sauvés dans %s", len(chunks), out)
    return 0


# ---------------------------------------------------------------------------
# upload : chunks JSON -> Pinecone
# ---------------------------------------------------------------------------
def cmd_upload(args: argparse.Namespace) -> int:
    from rag_core.vectorstore.pinecone_handler import PineconeInferenceUploader

    cfg = load_config(args.config)
    embed_cfg = require_section(cfg, "embedding")
    vs_cfg = require_section(cfg, "vectorstore")
    api_key = require_env("PINECONE_API_KEY")

    uploader = PineconeInferenceUploader(
        api_key=api_key,
        index_name=args.index,
        embed_model=require_key(embed_cfg, "model", "embedding"),
        cloud=require_key(vs_cfg, "cloud", "vectorstore"),
        region=require_key(vs_cfg, "region", "vectorstore"),
    )
    uploader.upload_chunks_from_json(args.chunks_json, namespace=args.namespace)
    logger.info("upload terminé vers %s/%s", args.index, args.namespace)
    return 0


# ---------------------------------------------------------------------------
# index : pipeline complet d'ingestion
# ---------------------------------------------------------------------------
def cmd_index(args: argparse.Namespace) -> int:
    from rag_core.chunking.chunk_optimizer import ChunkOptimizer
    from rag_core.chunking.text_splitter import SmartTextSplitter
    from rag_core.extraction.pdf_extractor import PDFExtractor
    from rag_core.vectorstore.pinecone_handler import PineconeInferenceUploader

    cfg = load_config(args.config)
    extraction_cfg = require_section(cfg, "extraction")
    chunk_cfg = require_section(cfg, "chunking")
    embed_cfg = require_section(cfg, "embedding")
    vs_cfg = require_section(cfg, "vectorstore")
    api_key = require_env("PINECONE_API_KEY")

    extractor = PDFExtractor(config=extraction_cfg)
    logger.info("extraction de %s", args.pdf_path)
    doc = extractor.extract_pdf(args.pdf_path, uploaded_url=args.pdf_path)

    splitter = SmartTextSplitter(
        chunk_size=require_key(chunk_cfg, "chunk_size", "chunking"),
        chunk_overlap=require_key(chunk_cfg, "chunk_overlap", "chunking"),
        strategy=require_key(chunk_cfg, "strategy", "chunking"),
    )
    chunks = splitter.split_document(doc)
    if require_key(chunk_cfg, "optimizer_enabled", "chunking"):
        chunks, _ = ChunkOptimizer().optimize_chunks(chunks)

    uploader = PineconeInferenceUploader(
        api_key=api_key,
        index_name=args.index,
        embed_model=require_key(embed_cfg, "model", "embedding"),
        cloud=require_key(vs_cfg, "cloud", "vectorstore"),
        region=require_key(vs_cfg, "region", "vectorstore"),
    )

    with tempfile.NamedTemporaryFile(suffix="_chunks.json", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = tmp.name
    splitter.save_chunks(chunks, tmp_path)
    try:
        uploader.upload_chunks_from_json(tmp_path, namespace=args.namespace)
    finally:
        os.remove(tmp_path)

    logger.info("indexation terminée — %d chunks dans %s/%s", len(chunks), args.index, args.namespace)
    return 0


# ---------------------------------------------------------------------------
# retrieve : query -> JSON chunks
# ---------------------------------------------------------------------------
def cmd_retrieve(args: argparse.Namespace) -> int:
    from rag_core.retrieval.retriever import PineconeRetriever

    cfg = load_config(args.config)
    embed_cfg = require_section(cfg, "embedding")
    ret_cfg = require_section(cfg, "retrieval")
    api_key = require_env("PINECONE_API_KEY")

    retriever = PineconeRetriever(
        api_key=api_key,
        index_name=args.index,
        embed_model=require_key(embed_cfg, "model", "embedding"),
        rerank_model=require_key(ret_cfg, "rerank_model", "retrieval"),
        namespace=args.namespace,
        truncation_max_tokens=require_key(ret_cfg, "truncation_max_tokens", "retrieval"),
        truncation_chars_per_token=require_key(ret_cfg, "truncation_chars_per_token", "retrieval"),
    )
    chunks = retriever.retrieve(
        query=args.query,
        retrieve_k=require_key(ret_cfg, "retrieve_k", "retrieval"),
        top_k=require_key(ret_cfg, "top_k", "retrieval"),
        rerank=require_key(ret_cfg, "rerank", "retrieval"),
        rerank_threshold=require_key(ret_cfg, "rerank_threshold", "retrieval"),
    )
    logger.info("%d chunks retournés", len(chunks))

    payload = [c.to_dict() for c in chunks]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("résultats sauvés dans %s", out)
    return 0


# ---------------------------------------------------------------------------
# ask : query -> réponse RAG complète
# ---------------------------------------------------------------------------
def cmd_ask(args: argparse.Namespace) -> int:
    from rag_core.generation.llm_handler import LLMHandler
    from rag_core.retrieval.retriever import PineconeRetriever

    cfg = load_config(args.config)
    embed_cfg = require_section(cfg, "embedding")
    ret_cfg = require_section(cfg, "retrieval")
    gen_cfg = require_section(cfg, "generation")
    api_key = require_env("PINECONE_API_KEY")
    hf_token = require_env("HF_TOKEN")

    retriever = PineconeRetriever(
        api_key=api_key,
        index_name=args.index,
        embed_model=require_key(embed_cfg, "model", "embedding"),
        rerank_model=require_key(ret_cfg, "rerank_model", "retrieval"),
        namespace=args.namespace,
        truncation_max_tokens=require_key(ret_cfg, "truncation_max_tokens", "retrieval"),
        truncation_chars_per_token=require_key(ret_cfg, "truncation_chars_per_token", "retrieval"),
    )
    chunks = retriever.retrieve(
        query=args.query,
        retrieve_k=require_key(ret_cfg, "retrieve_k", "retrieval"),
        top_k=require_key(ret_cfg, "top_k", "retrieval"),
        rerank=require_key(ret_cfg, "rerank", "retrieval"),
        rerank_threshold=require_key(ret_cfg, "rerank_threshold", "retrieval"),
    )

    llm = LLMHandler(
        model_name=require_key(gen_cfg, "model", "generation"),
        api_key=hf_token,
        temperature=require_key(gen_cfg, "temperature", "generation"),
        max_tokens=require_key(gen_cfg, "max_tokens", "generation"),
        max_retries=require_key(gen_cfg, "max_retries", "generation"),
        retry_delay_seconds=require_key(gen_cfg, "retry_delay_seconds", "generation"),
    )
    retrieved_chunks = [c.to_dict() for c in chunks]
    result = llm.generate_response(question=args.query, retrieved_chunks=retrieved_chunks)

    logger.info("réponse :\n%s", result["response"])
    logger.info("sources citées : %s", ", ".join(result.get("cited_sources", [])) or "(aucune)")
    logger.info("nombre de sources utilisées : %d", result.get("num_sources_used", 0))
    return 0


# ---------------------------------------------------------------------------
# parseur principal
# ---------------------------------------------------------------------------
def _add_config_arg(parser: argparse.ArgumentParser):
    parser.add_argument("--config", required=True, help="fichier YAML de config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="CLI unifiée pour le pipeline RAG (extract/chunk/upload/retrieve/ask/index)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<commande>")

    # extract
    p = sub.add_parser("extract", help="extraire un PDF en JSON")
    p.add_argument("pdf_path", help="chemin vers le PDF")
    p.add_argument("--out", required=True, help="chemin du JSON de sortie")
    _add_config_arg(p)
    p.set_defaults(func=cmd_extract)

    # chunk
    p = sub.add_parser("chunk", help="découper un document extrait en chunks")
    p.add_argument("extracted_json", help="JSON produit par `extract`")
    p.add_argument("--out", required=True, help="chemin du JSON de sortie")
    _add_config_arg(p)
    p.set_defaults(func=cmd_chunk)

    # upload
    p = sub.add_parser("upload", help="uploader des chunks JSON dans Pinecone")
    p.add_argument("chunks_json", help="JSON produit par `chunk`")
    p.add_argument("--index", required=True, help="nom de l'index Pinecone")
    p.add_argument("--namespace", required=True, help="namespace Pinecone")
    _add_config_arg(p)
    p.set_defaults(func=cmd_upload)

    # index
    p = sub.add_parser("index", help="pipeline complet : extract + chunk + upload")
    p.add_argument("pdf_path", help="chemin vers le PDF")
    p.add_argument("--index", required=True, help="nom de l'index Pinecone")
    p.add_argument("--namespace", required=True, help="namespace Pinecone")
    _add_config_arg(p)
    p.set_defaults(func=cmd_index)

    # retrieve
    p = sub.add_parser("retrieve", help="récupérer les chunks pertinents pour une question")
    p.add_argument("query", help="question en langage naturel")
    p.add_argument("--index", required=True, help="nom de l'index Pinecone")
    p.add_argument("--namespace", required=True, help="namespace Pinecone")
    p.add_argument("--out", required=True, help="dump JSON des résultats")
    _add_config_arg(p)
    p.set_defaults(func=cmd_retrieve)

    # ask
    p = sub.add_parser("ask", help="question/réponse RAG complète (retrieve + LLM)")
    p.add_argument("query", help="question en langage naturel")
    p.add_argument("--index", required=True, help="nom de l'index Pinecone")
    p.add_argument("--namespace", required=True, help="namespace Pinecone")
    _add_config_arg(p)
    p.set_defaults(func=cmd_ask)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

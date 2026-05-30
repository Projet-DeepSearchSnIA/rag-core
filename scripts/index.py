"""
indexe un PDF dans Pinecone : extraction -> chunking -> upload vectorstore.

usage:
    python scripts/index.py chemin/vers/fichier.pdf \
        --index mon-index \
        --namespace default \
        --config configs/baseline.yaml
"""

import argparse
import os
import sys
import tempfile
import yaml
from dotenv import load_dotenv

from rag_core.extraction.pdf_extractor import PDFExtractor
from rag_core.chunking.text_splitter import SmartTextSplitter
from rag_core.chunking.chunk_optimizer import ChunkOptimizer
from rag_core.vectorstore.pinecone_handler import PineconeInferenceUploader
from rag_core.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _require(cfg: dict, key: str, section: str):
    """Interrompt avec un message clair si la clé est absente du fichier de config."""
    if key not in cfg or cfg[key] is None:
        logger.error("Clé '%s' manquante dans la section [%s] du fichier de config", key, section)
        sys.exit(1)
    return cfg[key]


def main():
    parser = argparse.ArgumentParser(description="indexe un PDF dans Pinecone")
    parser.add_argument("pdf_path", help="chemin vers le PDF à indexer")
    parser.add_argument("--index", required=True, help="nom de l'index Pinecone")
    parser.add_argument("--namespace", required=True, help="namespace Pinecone")
    parser.add_argument("--config", required=True, help="fichier de config YAML")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY absente de l'environnement")
        sys.exit(1)

    extractor = PDFExtractor()
    logger.info("extraction de %s", args.pdf_path)
    doc = extractor.extract_pdf(args.pdf_path, uploaded_url=args.pdf_path)

    chunk_cfg = config.get("chunking") or {}
    splitter = SmartTextSplitter(
        chunk_size=_require(chunk_cfg, "chunk_size", "chunking"),
        chunk_overlap=_require(chunk_cfg, "chunk_overlap", "chunking"),
        strategy=_require(chunk_cfg, "strategy", "chunking"),
    )
    chunks = splitter.split_document(doc)

    if _require(chunk_cfg, "optimizer_enabled", "chunking"):
        optimizer = ChunkOptimizer()
        chunks, _ = optimizer.optimize_chunks(chunks)

    embed_cfg = config.get("embedding") or {}
    vs_cfg = config.get("vectorstore") or {}
    uploader = PineconeInferenceUploader(
        api_key=api_key,
        index_name=args.index,
        embed_model=_require(embed_cfg, "model", "embedding"),
        cloud=_require(vs_cfg, "cloud", "vectorstore"),
        region=_require(vs_cfg, "region", "vectorstore"),
    )

    with tempfile.NamedTemporaryFile(suffix="_chunks.json", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp_path = tmp.name

    splitter.save_chunks(chunks, tmp_path)
    try:
        uploader.upload_chunks_from_json(tmp_path, namespace=args.namespace)
    finally:
        os.remove(tmp_path)

    logger.info("indexation terminée ==> %d chunks", len(chunks))


if __name__ == "__main__":
    main()

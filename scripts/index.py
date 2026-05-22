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
import yaml
from dotenv import load_dotenv

from rag_core.extraction.pdf_extractor import PDFExtractor
from rag_core.chunking.text_splitter import SmartTextSplitter
from rag_core.chunking.chunk_optimizer import ChunkOptimizer
from rag_core.vectorstore.pinecone_handler import PineconeInferenceUploader
from rag_core.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="indexe un PDF dans Pinecone")
    parser.add_argument("pdf_path", help="chemin vers le PDF à indexer")
    parser.add_argument("--index", required=True, help="nom de l'index Pinecone")
    parser.add_argument("--namespace", default="__default__", help="namespace Pinecone")
    parser.add_argument("--config", default="configs/baseline.yaml", help="fichier de config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ["PINECONE_API_KEY"]

    extractor = PDFExtractor()
    logger.info("extraction de %s", args.pdf_path)
    doc = extractor.extract(args.pdf_path, uploaded_url=args.pdf_path)

    chunk_cfg = config.get("chunking", {})
    splitter = SmartTextSplitter(
        chunk_size=chunk_cfg.get("chunk_size", 1000),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 200),
        strategy=chunk_cfg.get("strategy", "mixed")
    )
    chunks = splitter.split_document(doc)

    if chunk_cfg.get("optimizer_enabled", True):
        optimizer = ChunkOptimizer()
        chunks = optimizer.optimize(chunks)

    embed_cfg = config.get("embedding", {})
    uploader = PineconeInferenceUploader(
        api_key=api_key,
        index_name=args.index,
        embed_model=embed_cfg.get("model", "multilingual-e5-large"),
        namespace=args.namespace
    )
    uploader.upsert_chunks(chunks)
    logger.info("indexation terminée — %d chunks", len(chunks))


if __name__ == "__main__":
    main()

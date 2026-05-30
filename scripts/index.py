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
    doc = extractor.extract_pdf(args.pdf_path, uploaded_url=args.pdf_path)

    chunk_cfg = config.get("chunking", {})
    splitter = SmartTextSplitter(
        chunk_size=chunk_cfg.get("chunk_size", 1000),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 200),
        strategy=chunk_cfg.get("strategy", "mixed")
    )
    chunks = splitter.split_document(doc)

    if chunk_cfg.get("optimizer_enabled", True):
        optimizer = ChunkOptimizer()
        chunks, _ = optimizer.optimize_chunks(chunks)

    embed_cfg = config.get("embedding", {})
    vs_cfg = config.get("vectorstore", {})
    uploader = PineconeInferenceUploader(
        api_key=api_key,
        index_name=args.index,
        embed_model=embed_cfg.get("model", "multilingual-e5-large"),
        cloud=vs_cfg.get("cloud", "aws"),
        region=vs_cfg.get("region", "us-east-1")
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

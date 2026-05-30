"""
interroge l'index Pinecone avec une question en langage naturel et affiche la réponse RAG.

usage:
    python scripts/query.py "quelle est la définition de X ?" \
        --index mon-index \
        --namespace default \
        --config configs/baseline.yaml
"""

import argparse
import os
import sys
import yaml
from dotenv import load_dotenv

from rag_core.retrieval.retriever import PineconeRetriever
from rag_core.generation.llm_handler import LLMHandler
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
    parser = argparse.ArgumentParser(description="requête RAG sur Pinecone")
    parser.add_argument("question", help="question en langage naturel")
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

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN absent de l'environnement")
        sys.exit(1)

    ret_cfg = config.get("retrieval") or {}
    embed_cfg = config.get("embedding") or {}
    retriever = PineconeRetriever(
        api_key=api_key,
        index_name=args.index,
        embed_model=_require(embed_cfg, "model", "embedding"),
        rerank_model=_require(ret_cfg, "rerank_model", "retrieval"),
        namespace=args.namespace,
        truncation_max_tokens=_require(ret_cfg, "truncation_max_tokens", "retrieval"),
        truncation_chars_per_token=_require(ret_cfg, "truncation_chars_per_token", "retrieval"),
    )

    chunks = retriever.retrieve(
        query=args.question,
        retrieve_k=_require(ret_cfg, "retrieve_k", "retrieval"),
        top_k=_require(ret_cfg, "top_k", "retrieval"),
        rerank=_require(ret_cfg, "rerank", "retrieval"),
        rerank_threshold=_require(ret_cfg, "rerank_threshold", "retrieval"),
    )

    gen_cfg = config.get("generation") or {}
    llm = LLMHandler(
        model_name=_require(gen_cfg, "model", "generation"),
        api_key=hf_token,
        temperature=_require(gen_cfg, "temperature", "generation"),
        max_tokens=_require(gen_cfg, "max_tokens", "generation"),
        max_retries=_require(gen_cfg, "max_retries", "generation"),
        retry_delay_seconds=_require(gen_cfg, "retry_delay_seconds", "generation"),
    )

    retrieved_chunks = [c.to_dict() for c in chunks]
    result = llm.generate_response(question=args.question, retrieved_chunks=retrieved_chunks)
    logger.info("réponse : %s", result["response"])


if __name__ == "__main__":
    main()

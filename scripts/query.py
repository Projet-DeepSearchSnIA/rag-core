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
import yaml
from dotenv import load_dotenv

from rag_core.retrieval.retriever import PineconeRetriever
from rag_core.generation.llm_handler import LLMHandler
from rag_core.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="requête RAG sur Pinecone")
    parser.add_argument("question", help="question en langage naturel")
    parser.add_argument("--index", required=True, help="nom de l'index Pinecone")
    parser.add_argument("--namespace", default="__default__", help="namespace Pinecone")
    parser.add_argument("--config", default="configs/baseline.yaml", help="fichier de config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ["PINECONE_API_KEY"]
    hf_token = os.environ.get("HF_TOKEN", "")

    ret_cfg = config.get("retrieval", {})
    embed_cfg = config.get("embedding", {})
    retriever = PineconeRetriever(
        api_key=api_key,
        index_name=args.index,
        embed_model=embed_cfg.get("model", "multilingual-e5-large"),
        rerank_model=ret_cfg.get("rerank_model", "bge-reranker-v2-m3"),
        namespace=args.namespace,
        truncation_max_tokens=ret_cfg.get("truncation_max_tokens", 200),
        truncation_chars_per_token=ret_cfg.get("truncation_chars_per_token", 4)
    )

    chunks = retriever.retrieve(
        query=args.question,
        retrieve_k=ret_cfg.get("retrieve_k", 20),
        top_k=ret_cfg.get("top_k", 5),
        rerank=ret_cfg.get("rerank", True),
        rerank_threshold=ret_cfg.get("rerank_threshold", 0.35)
    )

    gen_cfg = config.get("generation", {})
    llm = LLMHandler(
        model_name=gen_cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
        api_key=hf_token,
        temperature=gen_cfg.get("temperature", 0.7),
        max_tokens=gen_cfg.get("max_tokens", 1000),
        max_retries=gen_cfg.get("max_retries", 3),
        retry_delay_seconds=gen_cfg.get("retry_delay_seconds", 2)
    )

    retrieved_chunks = [c.to_dict() for c in chunks]
    result = llm.generate_response(question=args.question, retrieved_chunks=retrieved_chunks)
    print("\n" + result["response"])


if __name__ == "__main__":
    main()

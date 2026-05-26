from typing import List, Dict, Optional, Any
import json
from dataclasses import dataclass

from pinecone import Pinecone

from rag_core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichedChunk:
    """chunk enrichi avec toutes les métadonnées pour le LLM"""
    chunk_id: str
    text: str
    score: float
    rerank_score: Optional[float]

    document_id: str
    document_name: str
    document_title: str
    page_numbers: str

    has_formulas: bool
    formulas_latex: List[str]
    num_formulas: int

    has_images: bool
    image_ids: List[str]
    image_paths: List[str]
    num_images: int

    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'rerank_score': self.rerank_score,
            'document_id': self.document_id,
            'document_name': self.document_name,
            'document_title': self.document_title,
            'page_numbers': self.page_numbers,
            'has_formulas': self.has_formulas,
            'formulas_latex': self.formulas_latex,
            'num_formulas': self.num_formulas,
            'has_images': self.has_images,
            'image_ids': self.image_ids,
            'image_paths': self.image_paths,
            'num_images': self.num_images
        }


class PineconeRetriever:

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embed_model: str,
        rerank_model: str,
        namespace: str = "__default__",
        truncation_max_tokens: int = 200,
        truncation_chars_per_token: int = 4
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.namespace = namespace
        self.input_type = "query"
        self.truncation_max_tokens = truncation_max_tokens
        self.truncation_chars_per_token = truncation_chars_per_token

        logger.info("retriever initialisé — index: %s, embed: %s, rerank: %s", index_name, embed_model, rerank_model)

    def _parse_json_field(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                return json.loads(text)
            except Exception:
                return value
        return value

    def _parse_list_field(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = self._parse_json_field(value)
            if isinstance(parsed, list):
                return parsed
            if ',' in value:
                return [v.strip() for v in value.split(',') if v.strip()]
        return []

    def _truncate_for_rerank(self, text: str, max_tokens: Optional[int] = None) -> str:
        # ratio chars/token configurable, corrigé par rapport à noxa (qui utilisait 2)
        if not text:
            return ""
        tokens = max_tokens if max_tokens is not None else getattr(self, 'truncation_max_tokens', 200)
        chars_per_token = getattr(self, 'truncation_chars_per_token', 4)
        max_chars = tokens * chars_per_token
        return text[:max_chars] if len(text) > max_chars else text

    def _normalize_metadata(self, meta: Dict) -> Dict:
        if not meta:
            return {}
        normalized = {}
        for k, v in meta.items():
            if k in {"images", "formulas", "image_ids", "image_paths", "formulas_latex"}:
                normalized[k] = self._parse_list_field(v)
            else:
                normalized[k] = v
        return normalized

    def _create_enriched_chunk(self, doc: Dict) -> EnrichedChunk:
        meta = doc.get("metadata", {})
        formulas_latex = self._parse_list_field(meta.get("formulas_latex", []))
        image_ids = self._parse_list_field(meta.get("image_ids", []))
        image_paths = self._parse_list_field(meta.get("image_paths", []))

        return EnrichedChunk(
            chunk_id=doc.get("id", ""),
            text=doc.get("text", ""),
            score=doc.get("score", 0.0),
            rerank_score=doc.get("rerank_score"),
            document_id=meta.get("document_id", ""),
            document_name=meta.get("document_name", ""),
            document_title=meta.get("document_title", ""),
            page_numbers=meta.get("page_numbers", ""),
            has_formulas=meta.get("has_formulas", False),
            formulas_latex=formulas_latex,
            num_formulas=meta.get("num_formulas", 0),
            has_images=meta.get("has_images", False),
            image_ids=image_ids,
            image_paths=image_paths,
            num_images=meta.get("num_images", 0),
            metadata=meta
        )

    def _query_pinecone(self, query: str, top_k: int, filter: Optional[Dict] = None) -> List[Dict]:
        logger.debug("query pinecone top_k=%d...", top_k)
        try:
            emb_response = self.pc.inference.embed(
                model=self.embed_model,
                inputs=[query],
                parameters={"input_type": self.input_type, "truncate": "END"}
            )
            emb_item = emb_response.data[0] if hasattr(emb_response, 'data') else emb_response[0]
            vec = emb_item['values'] if isinstance(emb_item, dict) else emb_item.values
        except Exception as e:
            logger.error("erreur embedding: %s", e)
            raise

        try:
            results = self.index.query(
                vector=vec,
                top_k=top_k,
                filter=filter,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = results.matches if hasattr(results, 'matches') else results.get("matches", [])
        except Exception as e:
            logger.error("erreur query: %s", e)
            raise

        docs = []
        for m in matches:
            meta = self._normalize_metadata(m.get("metadata", {}))
            rerank_text = meta.get("chunk_text") or meta.get("text") or meta.get("content") or ""
            text = meta.get("content") or meta.get("text") or rerank_text

            docs.append({
                "id": m.get("id"),
                "score": m.get("score", 0.0),
                "text": text,
                "rerank_text": rerank_text,
                "metadata": meta
            })

        return docs

    def _rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        logger.debug("reranking top_k=%d...", top_k)

        if not hasattr(self.pc, "inference") or not hasattr(self.pc.inference, "rerank"):
            logger.debug("rerank non disponible, skip")
            return docs[:top_k]

        texts = [self._truncate_for_rerank(d.get("rerank_text", "") or d.get("text", "")) for d in docs]

        try:
            resp = self.pc.inference.rerank(
                model=self.rerank_model,
                query=query,
                documents=texts,
                top_n=top_k,
                return_documents=False
            )

            items = list(resp.data) if hasattr(resp, "data") else (resp.get("data", []) if isinstance(resp, dict) else resp if isinstance(resp, list) else [])

            ranked = []
            for item in items:
                idx = item.get("index")
                if idx is None or idx >= len(docs):
                    continue
                doc_copy = dict(docs[int(idx)])
                doc_copy["rerank_score"] = item.get("score", 0.0)
                ranked.append(doc_copy)

            ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return ranked[:top_k]

        except Exception as e:
            logger.warning("erreur rerank: %s, fallback sans rerank", e)
            return docs[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_k: int = 12,
        rerank_threshold: float = 0.35,
        retrieve_k: int = 20,
        rerank: bool = True,
        filter: Optional[Dict] = None
    ) -> List[EnrichedChunk]:
        logger.info("retrieval — query: %s...", query[:80])

        candidates = self._query_pinecone(query, retrieve_k, filter=filter)

        if not candidates:
            logger.info("aucun résultat")
            return []

        if rerank:
            ranked_docs = self._rerank(query, candidates, min(retrieve_k, max_k))
        else:
            ranked_docs = candidates[:min(top_k, max_k)]

        if rerank:
            filtered = [d for d in ranked_docs if d.get("rerank_score") is not None and d.get("rerank_score") >= rerank_threshold]
        else:
            filtered = ranked_docs

        final_limit = min(top_k, max_k)
        final_docs = filtered[:final_limit] if len(filtered) >= final_limit else filtered
        if len(final_docs) < final_limit:
            needed = final_limit - len(final_docs)
            final_ids = {d["id"] for d in final_docs}
            extras = [d for d in ranked_docs if d["id"] not in final_ids]
            final_docs.extend(extras[:needed])

        enriched_chunks = [self._create_enriched_chunk(doc) for doc in final_docs]

        logger.info(
            "%d chunks enrichis, formules: %d, images: %d",
            len(enriched_chunks),
            sum(1 for c in enriched_chunks if c.has_formulas),
            sum(1 for c in enriched_chunks if c.has_images)
        )
        return enriched_chunks

    def format_for_llm(self, chunks: List[EnrichedChunk]) -> str:
        if not chunks:
            return "aucun contexte pertinent trouvé."

        context_parts = ["# CONTEXTE RÉCUPÉRÉ\n"]

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"\n## SOURCE {i}")
            context_parts.append(f"Document: {chunk.document_name}")
            context_parts.append(f"Pages: {chunk.page_numbers}")
            context_parts.append(f"\nContenu:\n{chunk.text}")

            if chunk.has_formulas and chunk.formulas_latex:
                context_parts.append("\nFormules mathématiques:")
                for j, formula in enumerate(chunk.formulas_latex, 1):
                    context_parts.append(f"{j}. {formula}")

            if chunk.has_images:
                context_parts.append("\nImages/Figures:")
                for j, path in enumerate(chunk.image_paths or chunk.image_ids, 1):
                    context_parts.append(f"{j}. {path}")

            context_parts.append("\n" + "─" * 60)

        return "\n".join(context_parts)

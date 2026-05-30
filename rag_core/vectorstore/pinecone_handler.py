import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time

from rag_core.chunking.chunk_schemas import DocumentChunk
from rag_core.utils.logger import get_logger

logger = get_logger(__name__)


class PineconeInferenceUploader:
    """uploader pinecone avec embedding géré côté serveur par l'inference API"""

    def __init__(
        self,
        api_key: str,
        index_name: str,
        cloud: str,
        region: str,
        embed_model: str,
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.embed_model = embed_model

        self._init_index()

    def _init_index(self):
        existing = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name in existing:
            logger.info("index existant: %s", self.index_name)
            self.index = self.pc.Index(self.index_name)
            stats = self.index.describe_index_stats()
            logger.info("vecteurs actuels: %d", stats.total_vector_count)
        else:
            logger.info("création de l'index: %s", self.index_name)
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud=self.cloud,
                region=self.region,
                embed={
                    "model": self.embed_model,
                    "field_map": {"text": "chunk_text"}
                }
            )

            deadline = time.time() + 120
            while not self.pc.describe_index(self.index_name).status['ready']:
                if time.time() > deadline:
                    raise TimeoutError(f"index '{self.index_name}' not ready after 120s")
                time.sleep(1)

            self.index = self.pc.Index(self.index_name)
            logger.info("index créé")

    def _prepare_metadata(self, chunk: Dict) -> Dict:
        """Convertit un chunk JSON en métadonnées plates pour Pinecone.

        Valide d'abord la structure via DocumentChunk.from_dict — toute clé
        requise manquante lève une ValueError plutôt que d'indexer avec un
        défaut silencieux (string vide, 0).
        """
        doc_chunk = DocumentChunk.from_dict(chunk)
        m = doc_chunk.metadata

        pages = doc_chunk.page_numbers
        metadata = {
            "document_id": doc_chunk.document_id,
            "document_name": doc_chunk.document_name,
            "chunk_index": doc_chunk.chunk_index,
            "char_count": doc_chunk.char_count,
            "word_count": doc_chunk.word_count,
            "page_numbers": ",".join(map(str, pages)),
            "first_page": pages[0] if pages else 0,
            "publication_id": m.publication_id,
            "attachment_id": m.attachment_id,
            "user_id": m.user_id,
            "is_public": m.is_public,
            "document_title": (m.document_title or "")[:200],
            "document_author": (m.document_author or "")[:200],
            "has_images": m.has_images,
            "has_formulas": m.has_formulas,
        }
        if m.section_title:
            metadata["section_title"] = m.section_title[:200]

        if m.formulas:
            formulas_latex = [f.get("latex", "") for f in m.formulas if f.get("latex")]
            metadata["formulas_latex"] = formulas_latex[:5]
            metadata["formulas_latex_str"] = " || ".join(formulas_latex[:5])[:500]
            metadata["num_formulas"] = len(formulas_latex)

        if m.images or m.image_ids or m.image_paths:
            if m.image_ids:
                metadata["image_ids"] = m.image_ids[:10]
                metadata["image_ids_str"] = ",".join(m.image_ids[:10])
            if m.image_paths:
                metadata["image_paths"] = m.image_paths[:10]
                metadata["image_paths_str"] = ",".join(m.image_paths[:10])[:500]
            metadata["num_images"] = len(m.images) if m.images else len(m.image_ids or m.image_paths)

        footer_parts = []
        if m.formulas:
            footer_parts.append("[FORMULES MATHÉMATIQUES PRÉSENTES]")
            if metadata.get("formulas_latex"):
                footer_parts.append("FORMULES_LATEX: " + " || ".join(metadata["formulas_latex"]))
        if m.images:
            footer_parts.append("[IMAGES PRÉSENTES]")
            if metadata.get("image_paths"):
                footer_parts.append("IMAGE_PATHS: " + " || ".join(metadata["image_paths"]))
            elif metadata.get("image_ids"):
                footer_parts.append("IMAGE_IDS: " + " || ".join(metadata["image_ids"]))

        metadata["text"] = doc_chunk.content + "\n\n" + "\n".join(footer_parts) if footer_parts else doc_chunk.content
        metadata["content"] = doc_chunk.content

        return metadata

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                sanitized[key] = value
            else:
                try:
                    sanitized[key] = json.dumps(value, ensure_ascii=False)
                except Exception:
                    sanitized[key] = str(value)
        return sanitized

    def _flatten_metadata_for_records(self, metadata: Dict) -> Dict:
        flat = metadata.copy()
        flat.pop('text', None)
        return flat

    def upload_chunks_from_json(self, json_path: str, batch_size: int = 100, namespace: str = "__default__"):
        logger.info("traitement: %s", Path(json_path).name)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = data.get('chunks', [])
        total = len(chunks)
        logger.info("%d chunks à uploader", total)

        uploaded = 0

        for i in tqdm(range(0, total, batch_size), desc="upload"):
            batch = chunks[i:i + batch_size]
            records = []

            for chunk in batch:
                metadata = self._sanitize_metadata(self._prepare_metadata(chunk))
                records.append({'id': chunk.get('chunk_id', f"chunk_{i}"), 'metadata': metadata})

            try:
                if hasattr(self.index, "upsert_records"):
                    records_with_text = []
                    for r in records:
                        meta = r.get('metadata', {})
                        text = meta.get('text', '')
                        flat_meta = self._flatten_metadata_for_records(meta)
                        record_item = {'id': r['id'], 'text': text, 'chunk_text': text}
                        record_item.update(flat_meta)
                        records_with_text.append(record_item)

                    self.index.upsert_records(namespace=namespace, records=records_with_text)
                    uploaded += len(records)
                else:
                    texts = [r['metadata'].get('text', '') for r in records]
                    embeds = self.pc.inference.embed(
                        model=self.embed_model,
                        inputs=texts,
                        parameters={"input_type": "passage", "truncate": "END"}
                    )
                    embed_items = embeds.data if hasattr(embeds, 'data') else list(embeds)
                    vectors = []
                    for r, e in zip(records, embed_items):
                        values = e['values'] if isinstance(e, dict) else e.values
                        vectors.append({
                            'id': r['id'],
                            'values': values,
                            'metadata': r['metadata']
                        })
                    self.index.upsert(vectors=vectors, namespace=namespace)
                    uploaded += len(records)

            except Exception as e:
                logger.warning("erreur batch %d: %s, tentative record par record", i // batch_size, e)
                for record in records:
                    try:
                        if hasattr(self.index, "upsert_records"):
                            meta = record.get('metadata', {})
                            text = meta.get('text', '')
                            flat_meta = self._flatten_metadata_for_records(meta)
                            record_item = {'id': record['id'], 'text': text, 'chunk_text': text}
                            record_item.update(flat_meta)
                            self.index.upsert_records(namespace=namespace, records=[record_item])
                        else:
                            embed = self.pc.inference.embed(
                                model=self.embed_model,
                                inputs=[record['metadata'].get('text', '')],
                                parameters={"input_type": "passage", "truncate": "END"}
                            )
                            e = embed.data[0] if hasattr(embed, 'data') else embed[0]
                            values = e['values'] if isinstance(e, dict) else e.values
                            self.index.upsert(
                                vectors=[{'id': record['id'], 'values': values, 'metadata': record['metadata']}],
                                namespace=namespace
                            )
                        uploaded += 1
                    except Exception as e2:
                        logger.error("échec %s: %s", record["id"], e2)

        logger.info("%d/%d chunks uploadés", uploaded, total)
        return {
            'total': total,
            'uploaded': uploaded,
            'with_formulas': sum(1 for c in chunks if c.get('metadata', {}).get('has_formulas')),
            'with_images': sum(1 for c in chunks if c.get('metadata', {}).get('has_images'))
        }

    def upload_directory(self, chunks_dir: str, pattern: str = "*_chunks.json", namespace: str = "__default__"):
        chunks_path = Path(chunks_dir)
        files = list(chunks_path.glob(pattern))

        if not files:
            logger.warning("aucun fichier dans %s", chunks_dir)
            return

        logger.info("%d fichier(s) à uploader", len(files))

        global_stats = {'files': len(files), 'total_chunks': 0, 'uploaded': 0, 'with_formulas': 0, 'with_images': 0, 'errors': []}

        for file in files:
            try:
                stats = self.upload_chunks_from_json(str(file), namespace=namespace)
                global_stats['total_chunks'] += stats['total']
                global_stats['uploaded'] += stats['uploaded']
                global_stats['with_formulas'] += stats['with_formulas']
                global_stats['with_images'] += stats['with_images']
            except Exception as e:
                logger.error("erreur %s: %s", file.name, e)
                global_stats['errors'].append({'file': file.name, 'error': str(e)})

        logger.info("upload terminé — %d/%d chunks", global_stats["uploaded"], global_stats["total_chunks"])
        return global_stats

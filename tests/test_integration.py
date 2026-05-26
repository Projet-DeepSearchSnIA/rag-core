"""
Tests d'intégration end-to-end du pipeline RAG.

On mocke uniquement les APIs externes (Pinecone, HuggingFace). Tout le reste —
chunking, optimisation, sérialisation, préparation des métadonnées — est exercé
réellement. L'objectif est de détecter les régressions silencieuses, notamment
la divergence du format de métadonnées entre l'upload et le retrieval.
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_doc
from rag_core.chunking.text_splitter import SmartTextSplitter
from rag_core.chunking.chunk_optimizer import ChunkOptimizer
from rag_core.vectorstore.pinecone_handler import PineconeInferenceUploader
from rag_core.retrieval.retriever import EnrichedChunk, PineconeRetriever
from rag_core.generation.llm_handler import LLMHandler, RAGPipeline


# --- helpers ----------------------------------------------------------------

def _make_full_doc():
    # Document 5 pages, contenu varié, assez long pour forcer plusieurs chunks.
    return make_doc([
        "Introduction au machine learning. Le machine learning est une branche de l'intelligence artificielle.",
        "Les réseaux de neurones sont des modèles mathématiques inspirés du fonctionnement du cerveau humain.",
        "La régression logistique est utilisée pour les problèmes de classification binaire.",
        "Le gradient descent est un algorithme d'optimisation utilisé pour entraîner les modèles.",
        "Conclusion : le deep learning a révolutionné le traitement du langage naturel et la vision.",
    ])


def _uploader_sans_reseau():
    # On bypasse __init__ pour ne pas ouvrir de connexion Pinecone.
    uploader = object.__new__(PineconeInferenceUploader)
    uploader.cloud = "aws"
    uploader.region = "us-east-1"
    uploader.embed_model = "multilingual-e5-large"
    uploader.index_name = "test-index"
    return uploader


def _retriever_sans_reseau():
    # Même principe côté retrieval.
    retriever = object.__new__(PineconeRetriever)
    retriever.embed_model = "multilingual-e5-large"
    retriever.rerank_model = "bge-reranker-v2-m3"
    retriever.namespace = "__default__"
    retriever.input_type = "query"
    return retriever


def _make_enriched_chunk(text="contenu pertinent", score=0.9):
    return EnrichedChunk(
        chunk_id="c1", text=text, score=score, rerank_score=0.95,
        document_id="doc1", document_name="doc.pdf", document_title="Mon document",
        page_numbers="1,2", has_formulas=False, formulas_latex=[], num_formulas=0,
        has_images=False, image_ids=[], image_paths=[], num_images=0, metadata={}
    )


def _chunks_de_test():
    # Structure minimale qu'un LLMHandler attend en entrée.
    return [{
        "chunk_id": "c1", "text": "Le ML est une branche de l'IA.", "score": 0.9,
        "document_name": "ml.pdf", "document_title": "ML Intro", "page_numbers": "1",
        "has_formulas": False, "has_images": False, "metadata": {}
    }]


# --- chunking -> optimisation -----------------------------------------------

class TestChunkingPipeline:
    """La chaîne splitter -> optimizer doit produire des chunks cohérents et complets."""

    def test_split_et_optimize_preservent_le_contenu(self):
        # Vérifie que le texte source est bien présent dans le résultat final.
        doc = _make_full_doc()
        splitter = SmartTextSplitter(chunk_size=200, chunk_overlap=20, strategy="recursive")
        chunks = splitter.split_document(doc)
        optimizer = ChunkOptimizer(min_chunk_size=50, max_chunk_size=500)
        optimized, stats = optimizer.optimize_chunks(chunks)

        assert len(optimized) > 0
        assert stats['final_count'] == len(optimized)
        texte_complet = " ".join(c.content for c in optimized)
        assert "machine learning" in texte_complet.lower()

    def test_chunk_ids_uniques_apres_optimisation(self):
        # L'optimiseur fait des fusions : les IDs ne doivent jamais se dupliquer.
        doc = _make_full_doc()
        splitter = SmartTextSplitter(chunk_size=100, chunk_overlap=10, strategy="recursive")
        chunks = splitter.split_document(doc)
        optimizer = ChunkOptimizer(min_chunk_size=20)
        optimized, _ = optimizer.optimize_chunks(chunks)

        ids = [c.chunk_id for c in optimized]
        assert len(ids) == len(set(ids)), "IDs de chunks dupliqués après optimisation"

    def test_chunk_index_contigu_apres_optimisation(self):
        # chunk_index doit rester une séquence 0..N-1 après réindexation.
        doc = _make_full_doc()
        splitter = SmartTextSplitter(chunk_size=150, chunk_overlap=15, strategy="mixed")
        chunks = splitter.split_document(doc)
        optimizer = ChunkOptimizer()
        optimized, _ = optimizer.optimize_chunks(chunks)

        for i, chunk in enumerate(optimized):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(optimized)

    def test_metadonnees_document_survivent_au_chunking(self):
        # Les infos de haut niveau (titre, publication_id) doivent descendre dans chaque chunk.
        doc = _make_full_doc()
        doc.metadata.title = "Titre de test"
        doc.metadata.publication_id = 42

        splitter = SmartTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        optimizer = ChunkOptimizer()
        optimized, _ = optimizer.optimize_chunks(chunks)

        for chunk in optimized:
            assert chunk.metadata.get('document_title') == "Titre de test"
            assert chunk.metadata.get('publication_id') == 42


# --- serialisation JSON ------------------------------------------------------

class TestJsonSerialisation:
    """save_chunks doit produire un JSON rechargeable dont la structure est stable."""

    def test_save_et_reload_chunks(self):
        # Un aller-retour save/load doit donner le même nombre de chunks.
        doc = _make_full_doc()
        splitter = SmartTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        optimizer = ChunkOptimizer()
        optimized, _ = optimizer.optimize_chunks(chunks)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            splitter.save_chunks(optimized, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
        finally:
            os.unlink(tmp_path)

        assert "chunks" in data
        assert len(data["chunks"]) == len(optimized)

    def test_schema_json_contient_les_cles_attendues_par_prepare_metadata(self):
        # _prepare_metadata accède à ces clés côté upload, elles doivent toutes exister.
        doc = _make_full_doc()
        doc.metadata.title = "Doc de test"
        doc.metadata.publication_id = 7

        splitter = SmartTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_document(doc)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            splitter.save_chunks(chunks, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
        finally:
            os.unlink(tmp_path)

        for chunk in data["chunks"]:
            assert "document_id" in chunk
            assert "document_name" in chunk
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert isinstance(chunk.get("page_numbers"), list)
            assert isinstance(chunk.get("metadata"), dict)


# --- contrat metadonnees upload <-> retrieval --------------------------------

class TestMetadataContract:
    """
    Test de non-régression le plus important du projet.

    On simule le chemin complet des métadonnées : chunk JSON -> _prepare_metadata
    -> _sanitize_metadata (format Pinecone) -> _normalize_metadata -> _create_enriched_chunk.
    Si ce test casse, c'est que les deux côtés du pipeline ont divergé silencieusement.
    """

    def _roundtrip(self, chunk):
        # Côté upload : prépare et sérialise les métadonnées.
        uploader = _uploader_sans_reseau()
        metadata = uploader._prepare_metadata(chunk)
        sanitized = uploader._sanitize_metadata(metadata)

        # Côté retrieval : désérialise et construit l'EnrichedChunk.
        retriever = _retriever_sans_reseau()
        normalized = retriever._normalize_metadata(sanitized)
        doc_pinecone = {
            "id": chunk["chunk_id"], "score": 0.92,
            "text": normalized.get("content", ""),
            "rerank_score": None, "metadata": normalized,
        }
        return retriever._create_enriched_chunk(doc_pinecone)

    def test_roundtrip_chunk_simple(self):
        # Cas de base : chunk sans formules ni images.
        chunk = {
            "chunk_id": "doc123_chunk_0",
            "content": "Le machine learning est une branche de l'IA.",
            "document_id": "doc123", "document_name": "ml_intro.pdf",
            "page_numbers": [1, 2], "chunk_index": 0, "total_chunks": 5,
            "char_count": 46, "word_count": 9,
            "metadata": {
                "document_title": "Introduction au ML", "document_author": "Auteur Test",
                "publication_id": 42, "user_id": 7, "is_public": True,
                "has_images": False, "has_formulas": False,
                "formulas": [], "images": [], "image_ids": [], "image_paths": [],
            }
        }
        enriched = self._roundtrip(chunk)

        assert isinstance(enriched, EnrichedChunk)
        assert enriched.chunk_id == "doc123_chunk_0"
        assert enriched.document_id == "doc123"
        assert enriched.document_name == "ml_intro.pdf"
        assert enriched.document_title == "Introduction au ML"
        assert enriched.score == 0.92
        assert enriched.has_formulas is False
        assert enriched.has_images is False
        assert isinstance(enriched.formulas_latex, list)
        assert isinstance(enriched.image_ids, list)
        assert isinstance(enriched.image_paths, list)

    def test_roundtrip_chunk_avec_formules(self):
        # Les formules LaTeX doivent survivre à la sérialisation/désérialisation.
        chunk = {
            "chunk_id": "doc_math_chunk_0",
            "content": "La formule d'Euler est importante.",
            "document_id": "doc_math", "document_name": "math.pdf",
            "page_numbers": [3], "chunk_index": 0, "total_chunks": 2,
            "char_count": 34, "word_count": 6,
            "metadata": {
                "document_title": "Mathematiques", "document_author": "",
                "publication_id": None, "user_id": None, "is_public": False,
                "has_images": False, "has_formulas": True,
                "formulas": [
                    {"latex": r"e^{i\pi} + 1 = 0", "page": 3, "bbox": None},
                    {"latex": "F = ma", "page": 3, "bbox": None},
                ],
                "images": [], "image_ids": [], "image_paths": [],
            }
        }
        enriched = self._roundtrip(chunk)

        assert enriched.has_formulas is True
        assert isinstance(enriched.formulas_latex, list)
        assert len(enriched.formulas_latex) == 2
        assert r"e^{i\pi} + 1 = 0" in enriched.formulas_latex

    def test_roundtrip_chunk_avec_images(self):
        # Les image_ids et image_paths doivent rester des listes après le voyage.
        chunk = {
            "chunk_id": "doc_img_chunk_0",
            "content": "Voir la figure ci-dessous.",
            "document_id": "doc_img", "document_name": "rapport.pdf",
            "page_numbers": [5], "chunk_index": 0, "total_chunks": 3,
            "char_count": 26, "word_count": 5,
            "metadata": {
                "document_title": "Rapport avec images", "document_author": "",
                "publication_id": None, "user_id": None, "is_public": True,
                "has_images": True, "has_formulas": False,
                "formulas": [], "images": [{"id": "img_5_0"}],
                "image_ids": ["img_5_0", "img_5_1"],
                "image_paths": ["/data/temp/img_5_0.png", "/data/temp/img_5_1.png"],
            }
        }
        enriched = self._roundtrip(chunk)

        assert enriched.has_images is True
        assert isinstance(enriched.image_ids, list)
        assert "img_5_0" in enriched.image_ids
        assert isinstance(enriched.image_paths, list)

    def test_prepare_metadata_ne_plante_pas_sur_page_numbers_string(self):
        # Données legacy : page_numbers peut arriver en string au lieu d'une liste.
        chunk = {
            "chunk_id": "c_legacy", "content": "contenu",
            "document_id": "doc", "document_name": "doc.pdf",
            "page_numbers": "1,2,3",
            "chunk_index": 0, "total_chunks": 1, "char_count": 7, "word_count": 1,
            "metadata": {
                "has_images": False, "has_formulas": False,
                "formulas": [], "images": [], "image_ids": [], "image_paths": []
            }
        }
        uploader = _uploader_sans_reseau()
        metadata = uploader._prepare_metadata(chunk)
        # Peu importe la forme, la clé doit exister et ne pas crasher.
        assert "page_numbers" in metadata
        assert isinstance(metadata["page_numbers"], str)


# --- LLMHandler avec HuggingFace mocke ---------------------------------------

class TestLLMHandlerMocked:
    """Valide la logique de génération (retry, format de sortie) sans appel réseau."""

    def _make_llm(self):
        # On patche InferenceClient à la construction pour éviter une vraie connexion.
        with patch("rag_core.generation.llm_handler.InferenceClient"):
            llm = LLMHandler(api_key="hf-fake-token")
        llm.client = MagicMock()
        return llm

    def _mock_completion(self, llm, text):
        mock_choice = MagicMock()
        mock_choice.message.content = text
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        llm.client.chat.completions.create.return_value = mock_completion

    def test_generate_response_retourne_les_cles_requises(self):
        # Le dict de sortie doit toujours avoir ces 4 clés — c'est le contrat public.
        llm = self._make_llm()
        self._mock_completion(llm, "Le ML est une branche de l'IA [Source: ml.pdf, page 1].")

        result = llm.generate_response("Qu'est-ce que le ML ?", retrieved_chunks=_chunks_de_test())

        assert "response" in result
        assert "cited_sources" in result
        assert "all_sources" in result
        assert "num_sources_used" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_generate_simple_retourne_une_string(self):
        llm = self._make_llm()
        self._mock_completion(llm, "Reponse directe.")

        result = llm.generate_simple("Quelle est la capitale de la France ?")
        assert result == "Reponse directe."

    @patch("rag_core.generation.llm_handler.time.sleep")
    def test_retry_sur_erreur_serveur(self, mock_sleep):
        # Une erreur 502 doit déclencher un retry, pas un échec immédiat.
        llm = self._make_llm()

        mock_choice = MagicMock()
        mock_choice.message.content = "Reponse apres retry."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        llm.client.chat.completions.create.side_effect = [
            Exception("502 Bad Gateway"),
            mock_completion,
        ]

        result = llm.generate_response("question", retrieved_chunks=_chunks_de_test())

        assert llm.client.chat.completions.create.call_count == 2
        assert "Reponse apres retry." in result["response"]
        mock_sleep.assert_called_once()

    @patch("rag_core.generation.llm_handler.time.sleep")
    def test_echec_gracieux_sur_erreur_non_serveur(self, mock_sleep):
        # Une erreur 400 (pas serveur) ne doit pas retenter et doit échouer proprement.
        llm = self._make_llm()
        llm.client.chat.completions.create.side_effect = Exception("400 Bad Request")

        result = llm.generate_response("question", retrieved_chunks=_chunks_de_test())

        assert llm.client.chat.completions.create.call_count == 1
        assert "Desole" in result["response"] or "Désolé" in result["response"]
        mock_sleep.assert_not_called()


# --- RAGPipeline avec retriever + LLM mockes ---------------------------------

class TestRAGPipelineMocked:
    """Vérifie que RAGPipeline orchestre correctement retriever et LLM."""

    def test_ask_appelle_retriever_puis_llm(self):
        # Le pipeline doit appeler les deux composants et retourner la réponse du LLM.
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_make_enriched_chunk()]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = {
            "response": "Reponse generee.", "cited_sources": ["doc.pdf"],
            "all_sources": ["doc.pdf"], "num_sources_used": 1
        }

        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm)
        result = pipeline.ask("Qu'est-ce que ce document explique ?")

        mock_retriever.retrieve.assert_called_once()
        mock_llm.generate_response.assert_called_once()
        assert result["response"] == "Reponse generee."
        assert result["num_sources_used"] == 1

    def test_ask_retourne_fallback_si_aucun_chunk(self):
        # Si le retriever ne trouve rien, le LLM ne doit pas être appelé.
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_llm = MagicMock()

        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm)
        result = pipeline.ask("Question sans resultats")

        mock_llm.generate_response.assert_not_called()
        assert "pertinents" in result["response"]
        assert result["num_sources_used"] == 0

    def test_ask_transmet_top_k_au_retriever(self):
        # top_k passé à ask() doit être transmis au retriever sans être modifié.
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [_make_enriched_chunk()]
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = {
            "response": "ok", "cited_sources": [], "all_sources": [], "num_sources_used": 1
        }

        pipeline = RAGPipeline(retriever=mock_retriever, llm=mock_llm)
        pipeline.ask("question", top_k=3)

        call_kwargs = mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("top_k") == 3

    def test_rag_pipeline_requiert_llm(self):
        with pytest.raises(ValueError, match="llm"):
            RAGPipeline()

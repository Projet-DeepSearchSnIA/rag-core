"""
Tests live — nécessitent les clés API du .env.

Ces tests ne s'exécutent PAS dans la CI ni avec un simple `pytest`.
Ils sont exclus par défaut grâce à la config pyproject.toml (`addopts = "-m 'not live'"`).

Pour les lancer :
    pytest -m live -v                          # tous les tests live
    pytest -m live -v -k "retrieval"           # seulement les tests de retrieval
    pytest -m live -v -k "generation"          # seulement la génération
    pytest -m live -v -k "e2e"                 # le test complet bout en bout
    pytest -m live -v --tb=short               # avec tracebacks courts

Ce qu'ils testent (que les tests mockés ne peuvent pas vérifier) :
  - La connexion réelle à Pinecone et la qualité de l'index
  - Le comportement réel du modèle d'embedding et du reranker
  - La réponse réelle de HuggingFace (latence, format, contenu)
  - La cohérence de bout en bout sur de vraies données indexées
"""
import pytest

from tests.conftest import retrieve_with_baseline


# ---------------------------------------------------------------------------
# Retrieval — Pinecone
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveRetrieval:
    """
    Teste le retriever contre l'index Pinecone réel.
    Nécessite : PINECONE_API_KEY + PINECONE_INDEX_NAME dans .env
    et au moins un document déjà indexé.
    """

    def test_connexion_et_index_accessible(self, live_retriever):
        # Si l'index n'existe pas ou la clé est mauvaise, ça plante ici.
        # Un résultat vide est OK — ce qui compte, c'est que la requête passe.
        chunks = retrieve_with_baseline(live_retriever, "test de connexion", top_k=1, retrieve_k=3)
        assert isinstance(chunks, list)

    def test_retrieve_retourne_des_enriched_chunks(self, live_retriever):
        from rag_core.retrieval.retriever import EnrichedChunk
        chunks = retrieve_with_baseline(live_retriever, "introduction", top_k=3, retrieve_k=10)

        if not chunks:
            pytest.skip("L'index est vide — indexer des documents d'abord")

        for c in chunks:
            assert isinstance(c, EnrichedChunk)
            # Champs obligatoires toujours présents
            assert isinstance(c.chunk_id, str) and c.chunk_id
            assert isinstance(c.text, str)
            assert isinstance(c.score, float)
            assert 0.0 <= c.score <= 1.0
            assert isinstance(c.document_name, str)
            assert isinstance(c.page_numbers, str)
            assert isinstance(c.formulas_latex, list)
            assert isinstance(c.image_ids, list)
            assert isinstance(c.image_paths, list)

    def test_scores_coherents(self, live_retriever):
        # Les résultats doivent être triés par score décroissant.
        chunks = retrieve_with_baseline(live_retriever, "méthode algorithme", top_k=5, retrieve_k=15)

        if len(chunks) < 2:
            pytest.skip("Pas assez de résultats pour vérifier le tri")

        scores = [c.rerank_score if c.rerank_score is not None else c.score for c in chunks]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Scores non triés à la position {i} : {scores[i]:.3f} < {scores[i+1]:.3f}"
            )

    def test_retrieve_sans_rerank(self, live_retriever):
        # Le fallback sans reranking doit quand même retourner des résultats.
        chunks = retrieve_with_baseline(
            live_retriever, "résultats expériences",
            top_k=3, retrieve_k=10, rerank=False,
        )
        assert isinstance(chunks, list)
        # Sans rerank, rerank_score doit être None pour tous les chunks
        for c in chunks:
            assert c.rerank_score is None

    def test_top_k_respecte(self, live_retriever):
        # On ne doit jamais recevoir plus de chunks que demandé.
        chunks = retrieve_with_baseline(live_retriever, "définition", top_k=2, retrieve_k=10)
        assert len(chunks) <= 2

    def test_format_for_llm_produit_du_texte(self, live_retriever):
        chunks = retrieve_with_baseline(live_retriever, "résumé", top_k=3, retrieve_k=10)

        if not chunks:
            pytest.skip("L'index est vide")

        context = live_retriever.format_for_llm(chunks)
        assert isinstance(context, str)
        assert len(context) > 50
        # Le nom du document doit apparaître dans le contexte
        assert chunks[0].document_name in context


# ---------------------------------------------------------------------------
# Génération — HuggingFace
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveGeneration:
    """
    Teste le LLMHandler contre l'API HuggingFace réelle.
    Nécessite : HF_TOKEN dans .env
    """

    def _chunk_factice(self):
        # Chunk minimal pour alimenter le LLM sans passer par Pinecone.
        return {
            "chunk_id": "test_chunk_0",
            "text": (
                "Le machine learning est une branche de l'intelligence artificielle "
                "qui permet aux systèmes d'apprendre à partir de données. "
                "Les principaux algorithmes incluent la régression, les arbres de décision "
                "et les réseaux de neurones."
            ),
            "score": 0.9,
            "document_name": "ml_intro.pdf",
            "document_title": "Introduction au Machine Learning",
            "page_numbers": "1",
            "has_formulas": False,
            "has_images": False,
            "metadata": {}
        }

    def test_generate_response_retourne_une_chaine(self, live_llm):
        result = live_llm.generate_response(
            "Qu'est-ce que le machine learning ?",
            retrieved_chunks=[self._chunk_factice()]
        )
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 10, "Réponse trop courte, probablement vide"

    def test_generate_response_contient_les_cles_requises(self, live_llm):
        result = live_llm.generate_response(
            "Définir le machine learning.",
            retrieved_chunks=[self._chunk_factice()]
        )
        assert "response" in result
        assert "cited_sources" in result
        assert "all_sources" in result
        assert "num_sources_used" in result

    def test_generate_simple_retourne_une_chaine(self, live_llm):
        result = live_llm.generate_simple("Quelle est la capitale de la France ?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_response_ne_plante_pas_sur_plusieurs_chunks(self, live_llm):
        # Pousser plusieurs chunks pour vérifier que le prompt ne dépasse pas les limites.
        chunks = [
            {**self._chunk_factice(), "chunk_id": f"c{i}", "text": f"Contenu du chunk {i}. " * 20}
            for i in range(5)
        ]
        result = live_llm.generate_response(
            "Résume les informations disponibles.",
            retrieved_chunks=chunks
        )
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# E2E complet — Pinecone + HuggingFace
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestLiveE2E:
    """
    Test bout en bout : retrieve depuis l'index réel, puis générer une réponse.
    Nécessite : PINECONE_API_KEY + PINECONE_INDEX_NAME + HF_TOKEN dans .env
    et au moins un document indexé.

    C'est le test le plus proche de ce que fait l'application en production.
    """

    def test_retrieve_puis_generer(self, live_retriever, live_llm):
        question = "Quelle est la méthode principale décrite dans les documents ?"

        # Retrieval
        chunks = retrieve_with_baseline(live_retriever, question, top_k=3, retrieve_k=10)

        if not chunks:
            pytest.skip("L'index est vide — indexer des documents d'abord")

        # Conversion pour le LLM
        chunks_dicts = [c.to_dict() for c in chunks]

        # Génération
        result = live_llm.generate_response(question, retrieved_chunks=chunks_dicts)

        # Vérifications minimales — on ne teste pas le contenu exact (non déterministe)
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 20
        assert result["num_sources_used"] > 0

    def test_pipeline_avec_question_hors_domaine(self, live_retriever, live_llm):
        # Une question sans rapport avec les documents doit quand même retourner
        # quelque chose (soit une réponse "je ne sais pas", soit des chunks peu pertinents).
        question = "Quelle est la recette de la tarte aux pommes ?"

        chunks = retrieve_with_baseline(live_retriever, question, top_k=3, retrieve_k=10)
        chunks_dicts = [c.to_dict() for c in chunks] if chunks else []

        result = live_llm.generate_response(question, retrieved_chunks=chunks_dicts)

        # Le pipeline ne doit jamais crasher, même hors domaine.
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_streaming_produit_des_tokens(self, live_retriever, live_llm):
        chunks = retrieve_with_baseline(live_retriever, "définition", top_k=2, retrieve_k=5)

        if not chunks:
            pytest.skip("L'index est vide")

        tokens = list(live_llm.stream_response(
            "Donne une courte définition.",
            [c.to_dict() for c in chunks]
        ))
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0
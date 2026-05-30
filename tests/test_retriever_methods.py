"""
Tests pour EnrichedChunk et les méthodes privées de PineconeRetriever qui préparent
les chunks avant de les passer au LLM.

PineconeRetriever nécessite une vraie clé Pinecone pour s'instancier, mais ses méthodes
privées sont du pur Python. On crée une instance vide via object.__new__ pour contourner
__init__ (qui ferait un appel réseau) tout en gardant les méthodes liées entre elles.
"""
from tests.conftest import _retriever_vide
from rag_core.retrieval.retriever import EnrichedChunk


def _doc_brut(
    chunk_id: str = "doc1_chunk_0",
    text: str = "Texte de test.",
    score: float = 0.9,
    meta: dict = None,
) -> dict:
    """Simule la structure d'un document retourné par Pinecone avant enrichissement."""
    return {
        "id": chunk_id,
        "text": text,
        "score": score,
        "rerank_score": 0.75,
        "metadata": meta or {},
    }


def test_create_enriched_chunk_champs_de_base():
    """_create_enriched_chunk doit remplir chunk_id, text et score correctement."""
    doc = _doc_brut(
        chunk_id="mon_chunk",
        text="Contenu du chunk.",
        score=0.88,
        meta={
            "document_id": "doc-abc",
            "document_name": "papier.pdf",
            "document_title": "Titre du papier",
            "page_numbers": "[1, 2]",
        },
    )
    chunk = _retriever_vide()._create_enriched_chunk(doc)
    assert chunk.chunk_id == "mon_chunk"
    assert chunk.text == "Contenu du chunk."
    assert chunk.score == 0.88
    assert chunk.document_id == "doc-abc"
    assert chunk.document_name == "papier.pdf"


def test_create_enriched_chunk_formules_depuis_json():
    """formulas_latex stocké comme JSON string doit être parsé en liste Python."""
    meta = {
        "formulas_latex": '["E=mc^2", "\\\\frac{d}{dx}\\\\sin(x)=\\\\cos(x)"]',
        "has_formulas": True,
        "num_formulas": 2,
    }
    doc = _doc_brut(meta=meta)
    chunk = _retriever_vide()._create_enriched_chunk(doc)
    assert isinstance(chunk.formulas_latex, list)
    assert len(chunk.formulas_latex) == 2
    assert chunk.has_formulas is True


def test_create_enriched_chunk_images_depuis_csv():
    """image_ids stocké comme CSV doit être parsé en liste Python."""
    meta = {
        "image_ids": "img_001,img_002,img_003",
        "has_images": True,
        "num_images": 3,
    }
    doc = _doc_brut(meta=meta)
    chunk = _retriever_vide()._create_enriched_chunk(doc)
    assert isinstance(chunk.image_ids, list)
    assert len(chunk.image_ids) == 3
    assert "img_001" in chunk.image_ids


def test_create_enriched_chunk_metadata_vide():
    """Sans métadonnées, les champs doivent avoir des valeurs par défaut sans planter."""
    doc = _doc_brut(meta={})
    chunk = _retriever_vide()._create_enriched_chunk(doc)
    assert chunk.has_formulas is False
    assert chunk.has_images is False
    assert chunk.formulas_latex == []
    assert chunk.image_ids == []
    assert chunk.image_paths == []
    assert chunk.num_formulas == 0
    assert chunk.num_images == 0


def test_create_enriched_chunk_rerank_score_none():
    """Un doc sans rerank_score doit produire chunk.rerank_score = None."""
    doc = _doc_brut()
    doc.pop("rerank_score", None)  # on retire explicitement le champ
    chunk = _retriever_vide()._create_enriched_chunk(doc)
    assert chunk.rerank_score is None


def test_to_dict_contient_toutes_les_cles():
    """to_dict() doit exposer tous les champs nécessaires au LLM handler."""
    doc = _doc_brut(
        chunk_id="c1",
        text="Texte.",
        meta={"document_name": "doc.pdf"},
    )
    chunk = _retriever_vide()._create_enriched_chunk(doc)
    d = chunk.to_dict()
    cles_attendues = [
        "chunk_id", "text", "score", "rerank_score",
        "document_id", "document_name", "document_title", "page_numbers",
        "has_formulas", "formulas_latex", "num_formulas",
        "has_images", "image_ids", "image_paths", "num_images",
    ]
    for cle in cles_attendues:
        assert cle in d, f"Clé manquante dans to_dict(): {cle}"


def test_normalize_metadata_listes_json():
    """_normalize_metadata doit parser les champs de type liste depuis JSON."""
    meta_brute = {
        "image_ids": '["id1", "id2"]',
        "formulas_latex": '["x^2"]',
        "document_name": "doc.pdf",  # champ non-liste → inchangé
    }
    meta_propre = _retriever_vide()._normalize_metadata(meta_brute)
    assert isinstance(meta_propre["image_ids"], list)
    assert isinstance(meta_propre["formulas_latex"], list)
    assert meta_propre["document_name"] == "doc.pdf"  # inchangé


def test_normalize_metadata_dict_vide():
    """_normalize_metadata sur un dict vide doit retourner un dict vide."""
    assert _retriever_vide()._normalize_metadata({}) == {}


def test_normalize_metadata_none():
    """_normalize_metadata sur None doit retourner un dict vide."""
    assert _retriever_vide()._normalize_metadata(None) == {}


def test_format_for_llm_sans_chunks():
    """format_for_llm sur liste vide doit retourner le message 'aucun contexte'."""
    result = _retriever_vide().format_for_llm([])
    assert "aucun" in result.lower()


def test_format_for_llm_contient_nom_document():
    """format_for_llm doit inclure le nom du document dans la sortie."""
    chunk = EnrichedChunk(
        chunk_id="c1",
        text="L'attention calcule des projections.",
        score=0.9,
        rerank_score=0.8,
        document_id="doc-1",
        document_name="transformer.pdf",
        document_title="Attention is All You Need",
        page_numbers="[3]",
        has_formulas=False,
        formulas_latex=[],
        num_formulas=0,
        has_images=False,
        image_ids=[],
        image_paths=[],
        num_images=0,
        metadata={},
    )
    resultat = _retriever_vide().format_for_llm([chunk])
    assert "transformer.pdf" in resultat


def test_format_for_llm_contient_le_texte():
    """format_for_llm doit inclure le texte brut du chunk."""
    chunk = EnrichedChunk(
        chunk_id="c2",
        text="Le reranking améliore la précision.",
        score=0.85,
        rerank_score=None,
        document_id="doc-2",
        document_name="rag_survey.pdf",
        document_title="",
        page_numbers="[5]",
        has_formulas=False,
        formulas_latex=[],
        num_formulas=0,
        has_images=False,
        image_ids=[],
        image_paths=[],
        num_images=0,
        metadata={},
    )
    resultat = _retriever_vide().format_for_llm([chunk])
    assert "reranking améliore" in resultat


def test_format_for_llm_avec_formules():
    """format_for_llm doit mentionner les formules LaTeX quand present."""
    chunk = EnrichedChunk(
        chunk_id="c3",
        text="La formule de Bayes.",
        score=0.9,
        rerank_score=None,
        document_id="doc-3",
        document_name="proba.pdf",
        document_title="",
        page_numbers="[10]",
        has_formulas=True,
        formulas_latex=[r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}"],
        num_formulas=1,
        has_images=False,
        image_ids=[],
        image_paths=[],
        num_images=0,
        metadata={},
    )
    resultat = _retriever_vide().format_for_llm([chunk])
    assert "Formules" in resultat or "P(A|B)" in resultat
"""
Tests pour ChunkOptimizer — amélioration de la qualité des chunks avant l'indexation.

L'optimiseur fait quatre choses : supprimer les vides, dédupliquer, fusionner les petits
et découper les grands. On teste chacune de ces passes séparément, puis le pipeline complet.
"""
from rag_core.chunking.chunk_optimizer import ChunkOptimizer
from rag_core.chunking.text_splitter import DocumentChunk


def make_chunk(
    content: str,
    document_id: str = "doc-test",
    page_numbers=None,
    metadata=None,
) -> DocumentChunk:
    """Construit un DocumentChunk minimal pour tester l'optimiseur sans passer par le splitter."""
    return DocumentChunk(
        chunk_id=f"{document_id}_chunk_0",
        content=content,
        document_id=document_id,
        document_name="doc.pdf",
        page_numbers=page_numbers or [1],
        chunk_index=0,
        total_chunks=1,
        metadata=metadata or {},
    )


def test_liste_vide_ne_plante_pas():
    """optimize_chunks sur une liste vide doit retourner ([], stats) sans exception."""
    optimiseur = ChunkOptimizer()
    chunks, stats = optimiseur.optimize_chunks([])
    assert chunks == []
    assert "original_count" in stats
    assert stats["original_count"] == 0


def test_chunks_vides_supprimes():
    """Les chunks dont le contenu est vide ou trop court (≤ 10 chars) sont éliminés.

    En pratique ça arrive quand le splitter découpe sur un séparateur seul.
    """
    optimiseur = ChunkOptimizer(min_chunk_size=100)
    chunks = [
        make_chunk(""),
        make_chunk("   "),
        make_chunk("trop court"),  # 10 chars exactement — doit passer le filtre >10
        make_chunk("Ce chunk a suffisamment de contenu pour survivre à l'optimisation."),
    ]
    resultat, _ = optimiseur.optimize_chunks(chunks)
    # les deux premiers disparaissent, les deux autres restent
    contenus = [c.content for c in resultat]
    assert "" not in contenus
    assert "   " not in contenus
    assert any("suffisamment" in c for c in contenus)


def test_doublons_presque_identiques_supprimes():
    """Deux chunks très similaires (Jaccard ≥ 0.9) → le deuxième est supprimé."""
    optimiseur = ChunkOptimizer(similarity_threshold=0.9, merge_small_chunks=False)
    texte = "Le transformer utilise un mécanisme d'attention multi-tête pour traiter les séquences."
    chunks = [
        make_chunk(texte, document_id="doc1"),
        # légèrement différent mais pratiquement identique
        make_chunk(texte + " ", document_id="doc1"),
    ]
    resultat, stats = optimiseur.optimize_chunks(chunks)
    assert len(resultat) == 1


def test_doublons_textes_differents_conserves():
    """Deux chunks clairement différents doivent tous les deux être conservés."""
    optimiseur = ChunkOptimizer(similarity_threshold=0.9, merge_small_chunks=False)
    chunks = [
        make_chunk("Le transformer utilise l'attention multi-tête.", document_id="doc1"),
        make_chunk("Le BERT utilise un encodeur bidirectionnel pour les représentations.", document_id="doc1"),
    ]
    resultat, _ = optimiseur.optimize_chunks(chunks)
    assert len(resultat) == 2


def test_petits_chunks_fusionnes():
    """Des chunks inférieurs à min_chunk_size doivent être fusionnés ensemble."""
    optimiseur = ChunkOptimizer(
        min_chunk_size=50,
        merge_small_chunks=True,
        split_large_chunks=False,
        remove_duplicates=False,
    )
    # trois chunks de ~15 chars chacun → doivent être fusionnés
    chunks = [
        make_chunk("Intro courte.", document_id="doc1"),
        make_chunk("Suite courte.", document_id="doc1"),
        make_chunk("Fin courte.", document_id="doc1"),
    ]
    resultat, stats = optimiseur.optimize_chunks(chunks)
    # on doit avoir moins de chunks qu'au départ
    assert len(resultat) < 3


def test_grands_chunks_decoupes():
    """Un chunk qui dépasse max_chunk_size doit être découpé en sous-chunks."""
    optimiseur = ChunkOptimizer(
        max_chunk_size=100,
        target_chunk_size=50,
        split_large_chunks=True,
        merge_small_chunks=False,
        remove_duplicates=False,
    )
    # phrase longue de ~200 chars pour forcer la découpe
    contenu = (
        "L'attention est un mécanisme clé. "
        "Il permet de peser l'importance relative des tokens. "
        "BERT l'utilise de façon bidirectionnelle. "
        "GPT l'utilise de façon causale."
    )
    chunk = make_chunk(contenu, document_id="doc1")
    resultat, stats = optimiseur.optimize_chunks([chunk])
    assert len(resultat) >= 2


def test_chunk_avec_formule_pas_decoupe():
    """Un chunk qui contient des formules LaTeX ne doit PAS être découpé même s'il est grand.

    Les formules perdent leur sens si on les coupe en milieu. C'est une règle métier importante.
    """
    optimiseur = ChunkOptimizer(
        max_chunk_size=50,  # seuil très bas pour forcer la découpe normalement
        split_large_chunks=True,
        merge_small_chunks=False,
        remove_duplicates=False,
    )
    contenu = r"La formule de Bayes est $$ P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} $$. Très utile en probabilités."
    chunk = make_chunk(contenu, document_id="doc1", metadata={"has_formulas": True})
    resultat, _ = optimiseur.optimize_chunks([chunk])
    # le chunk formule doit rester entier
    assert len(resultat) == 1
    assert r"P(A|B)" in resultat[0].content


def test_reindexation_apres_optimisation():
    """Après optimisation, chunk_index et total_chunks doivent être cohérents."""
    optimiseur = ChunkOptimizer(
        merge_small_chunks=False,
        split_large_chunks=False,
        remove_duplicates=False,
    )
    chunks = [
        make_chunk("Premier chunk de taille correcte.", document_id="doc1"),
        make_chunk("Deuxième chunk de taille correcte.", document_id="doc1"),
        make_chunk("Troisième chunk de taille correcte.", document_id="doc1"),
    ]
    # on set manuellement des index incohérents pour tester la reindexation
    for i, c in enumerate(chunks):
        c.chunk_index = 99
        c.total_chunks = 99

    resultat, _ = optimiseur.optimize_chunks(chunks)
    total = len(resultat)
    for i, chunk in enumerate(resultat):
        assert chunk.chunk_index == i
        assert chunk.total_chunks == total


def test_stats_retournees_correctes():
    """optimize_chunks doit retourner un dict stats avec les clés attendues."""
    optimiseur = ChunkOptimizer()
    chunks = [make_chunk("Du contenu suffisamment long pour ne pas être supprimé.", document_id="doc1")]
    _, stats = optimiseur.optimize_chunks(chunks)
    assert "original_count" in stats
    assert "final_count" in stats
    assert "size_stats" in stats


def test_analyze_chunks_cles_attendues():
    """analyze_chunks doit retourner toutes les clés de statistiques documentées."""
    optimiseur = ChunkOptimizer()
    chunks = [
        make_chunk("Un chunk avec du contenu.", document_id="doc1"),
        make_chunk("Un autre chunk.", document_id="doc1"),
    ]
    analyse = optimiseur.analyze_chunks(chunks)
    assert "total_chunks" in analyse
    assert "size_stats" in analyse
    assert "page_distribution" in analyse
    assert "quality_checks" in analyse
    assert analyse["total_chunks"] == 2


def test_analyze_chunks_vide():
    """analyze_chunks sur liste vide renvoie {'total': 0} sans planter."""
    optimiseur = ChunkOptimizer()
    analyse = optimiseur.analyze_chunks([])
    assert analyse == {"total": 0}


def test_similarite_texte_identique():
    """Deux textes identiques → similarité Jaccard = 1.0."""
    optimiseur = ChunkOptimizer()
    texte = "le transformer est un modèle de traitement du langage naturel"
    score = optimiseur._text_similarity(texte, texte)
    assert score == 1.0


def test_similarite_texte_sans_overlap():
    """Deux textes sans aucun mot commun → similarité = 0.0."""
    optimiseur = ChunkOptimizer()
    score = optimiseur._text_similarity("chat chien lapin", "table chaise bureau")
    assert score == 0.0


def test_similarite_texte_vide():
    """Quand un texte est vide, la similarité vaut 0 sans lever d'exception."""
    optimiseur = ChunkOptimizer()
    score = optimiseur._text_similarity("", "quelque chose")
    assert score == 0.0


def test_fusion_preserver_pages():
    """Après fusion de petits chunks, les numéros de page doivent tous être conservés."""
    optimiseur = ChunkOptimizer(
        min_chunk_size=200,
        merge_small_chunks=True,
        split_large_chunks=False,
        remove_duplicates=False,
    )
    # contenu > 10 chars pour passer _remove_empty_chunks, mais < min_chunk_size=200 pour déclencher la fusion
    chunk_p1 = make_chunk("Contenu de la première page.", document_id="doc1", page_numbers=[1])
    chunk_p2 = make_chunk("Contenu de la deuxième page.", document_id="doc1", page_numbers=[2])
    resultat, _ = optimiseur.optimize_chunks([chunk_p1, chunk_p2])
    # les deux pages doivent apparaître dans le chunk fusionné
    pages = resultat[0].page_numbers
    assert 1 in pages
    assert 2 in pages
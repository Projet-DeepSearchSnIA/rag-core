"""
Fonctions de métriques RAG et leurs tests unitaires.

Ces fonctions serviront à évaluer nos expériences dans les labs. On les définit
ici dans rag-core pour avoir une référence stable, et on les teste sur des cas
dont on connaît le résultat attendu à la main.

Métriques de retrieval : MRR@K, Recall@K, NDCG@K, Precision@K.
Métriques de génération : faithfulness, hallucination_rate.

Toutes ces fonctions sont pures (aucun effet de bord), ce qui les rend triviales à tester.
"""
import math


# ---------------------------------------------------------------------------
# Fonctions de métriques — on les définit ici pour les avoir sous la main
# sans dépendre d'un module externe qui n'existe pas encore.
# ---------------------------------------------------------------------------

def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Position du premier chunk pertinent dans les k premiers résultats.

    Retourne 1/rang si un pertinent est trouvé, 0 sinon.
    MRR=1 signifie que le premier résultat était toujours pertinent.
    """
    relevant = set(relevant_ids)
    for rang, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rang
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Proportion de chunks pertinents retrouvés dans les k premiers résultats.

    Recall=1 signifie qu'on a retrouvé tous les chunks attendus.
    Si relevant_ids est vide, on retourne 1.0 par convention (rien à retrouver).
    """
    if not relevant_ids:
        return 1.0
    retrouves = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(retrouves) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Proportion de résultats pertinents parmi les k premiers.

    Precision=1 signifie que tous les résultats retournés étaient pertinents.
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    pertinents = sum(1 for cid in top_k if cid in set(relevant_ids))
    return pertinents / k


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain — pénalise les pertinents mal classés.

    NDCG=1 quand tous les pertinents sont en tête de liste.
    NDCG=0 quand aucun pertinent n'est dans les k premiers résultats.
    """
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rang + 1)
        for rang, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in relevant
    )
    idcg = sum(
        1.0 / math.log2(rang + 1)
        for rang in range(1, min(len(relevant_ids), k) + 1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def faithfulness_score(response: str, context_chunks: list[str]) -> float:
    """Proportion de phrases de la réponse qui s'appuient sur le contexte.

    Approximation lexicale : une phrase est "supportée" si au moins un mot
    de plus de 4 lettres apparaît dans le contexte. C'est une borne basse,
    pas un vrai NLI, mais ça suffit pour les tests unitaires.
    """
    phrases = [s.strip() for s in response.split(".") if s.strip()]
    if not phrases:
        return 0.0
    contexte = " ".join(context_chunks).lower()
    supportees = sum(
        1 for phrase in phrases
        if any(mot in contexte for mot in phrase.lower().split() if len(mot) > 4)
    )
    return supportees / len(phrases)


def hallucination_rate(response: str, context_chunks: list[str]) -> float:
    """Complément du faithfulness — proportion de phrases sans support dans le contexte."""
    return 1.0 - faithfulness_score(response, context_chunks)


# ---------------------------------------------------------------------------
# Tests pour mrr_at_k
# ---------------------------------------------------------------------------

def test_mrr_premier_resultat_pertinent():
    """Premier résultat pertinent → MRR = 1.0."""
    score = mrr_at_k(["chunk_1", "chunk_2", "chunk_3"], ["chunk_1"], k=5)
    assert score == 1.0


def test_mrr_deuxieme_resultat_pertinent():
    """Deuxième résultat pertinent → MRR = 0.5."""
    score = mrr_at_k(["chunk_x", "chunk_1", "chunk_3"], ["chunk_1"], k=5)
    assert score == 0.5


def test_mrr_troisieme_resultat_pertinent():
    """Troisième résultat pertinent → MRR = 1/3."""
    score = mrr_at_k(["a", "b", "chunk_bon"], ["chunk_bon"], k=5)
    assert abs(score - 1 / 3) < 1e-9


def test_mrr_aucun_pertinent():
    """Aucun pertinent dans les k premiers → MRR = 0."""
    score = mrr_at_k(["a", "b", "c"], ["chunk_attendu"], k=3)
    assert score == 0.0


def test_mrr_pertinent_hors_k():
    """Pertinent présent mais au-delà de k → non compté → MRR = 0."""
    score = mrr_at_k(["a", "b", "c", "chunk_bon"], ["chunk_bon"], k=3)
    assert score == 0.0


def test_mrr_plusieurs_pertinents_premier_compte():
    """Quand plusieurs résultats sont pertinents, seul le premier rang compte."""
    score = mrr_at_k(["chunk_1", "chunk_2"], ["chunk_1", "chunk_2"], k=5)
    assert score == 1.0  # premier est pertinent → rang 1


# ---------------------------------------------------------------------------
# Tests pour recall_at_k
# ---------------------------------------------------------------------------

def test_recall_tous_retrouves():
    """Tous les pertinents retrouvés dans les k premiers → Recall = 1."""
    score = recall_at_k(["c1", "c2", "c3"], ["c1", "c2"], k=3)
    assert score == 1.0


def test_recall_moitie_retrouvee():
    """La moitié des pertinents retrouvés → Recall = 0.5."""
    score = recall_at_k(["c1", "c_autre"], ["c1", "c2"], k=2)
    assert score == 0.5


def test_recall_aucun_retrouve():
    """Aucun pertinent retrouvé → Recall = 0."""
    score = recall_at_k(["x", "y", "z"], ["c1", "c2"], k=3)
    assert score == 0.0


def test_recall_liste_vide_pertinents():
    """Aucun pertinent attendu → Recall = 1 par convention."""
    score = recall_at_k(["c1", "c2"], [], k=5)
    assert score == 1.0


# ---------------------------------------------------------------------------
# Tests pour precision_at_k
# ---------------------------------------------------------------------------

def test_precision_tous_pertinents():
    """Tous les résultats retournés sont pertinents → Precision@K = 1."""
    score = precision_at_k(["c1", "c2"], ["c1", "c2", "c3"], k=2)
    assert score == 1.0


def test_precision_aucun_pertinent():
    """Aucun résultat pertinent → Precision@K = 0."""
    score = precision_at_k(["x", "y"], ["c1", "c2"], k=2)
    assert score == 0.0


def test_precision_moitie():
    """Un résultat sur deux est pertinent → Precision@2 = 0.5."""
    score = precision_at_k(["c1", "x"], ["c1"], k=2)
    assert score == 0.5


def test_precision_k_zero():
    """k=0 → Precision = 0 sans division par zéro."""
    score = precision_at_k(["c1"], ["c1"], k=0)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Tests pour ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_classement_parfait():
    """Pertinents en tête de liste dans l'ordre optimal → NDCG = 1."""
    score = ndcg_at_k(["c1", "c2", "c3"], ["c1", "c2"], k=3)
    assert abs(score - 1.0) < 1e-9


def test_ndcg_aucun_pertinent():
    """Aucun pertinent dans les k premiers → NDCG = 0."""
    score = ndcg_at_k(["x", "y", "z"], ["c1", "c2"], k=3)
    assert score == 0.0


def test_ndcg_ordre_degrade_moins_bon_que_parfait():
    """Un pertinent en 2e position donne un NDCG < 1 (pénalité de rang)."""
    ndcg_parfait = ndcg_at_k(["c1", "irrelevant"], ["c1"], k=2)
    ndcg_degrade = ndcg_at_k(["irrelevant", "c1"], ["c1"], k=2)
    assert ndcg_parfait > ndcg_degrade


def test_ndcg_liste_vide_pertinents():
    """Aucun pertinent attendu → NDCG = 0 (pas de division par zéro)."""
    score = ndcg_at_k(["c1", "c2"], [], k=3)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Tests pour faithfulness_score
# ---------------------------------------------------------------------------

def test_faithfulness_reponse_dans_contexte():
    """Une réponse dont tous les mots viennent du contexte → score élevé."""
    contexte = ["Le transformer utilise l'attention multi-tête pour traiter les séquences."]
    reponse = "Le transformer utilise l'attention pour les séquences."
    score = faithfulness_score(reponse, contexte)
    assert score > 0.5


def test_faithfulness_reponse_hors_contexte():
    """Une réponse inventée qui ne figure pas dans le contexte → score bas."""
    contexte = ["Le transformer est un modèle de NLP."]
    reponse = "Les pingouins vivent en Antarctique et mangent du poisson."
    score = faithfulness_score(reponse, contexte)
    assert score < 0.5


def test_faithfulness_reponse_vide():
    """Réponse vide → faithfulness = 0 sans erreur."""
    score = faithfulness_score("", ["Contexte quelconque."])
    assert score == 0.0


def test_hallucination_complement_faithfulness():
    """hallucination_rate doit toujours valoir 1 - faithfulness_score."""
    contexte = ["BERT est un modèle bidirectionnel basé sur le Transformer."]
    reponse = "BERT utilise un encodeur bidirectionnel."
    faith = faithfulness_score(reponse, contexte)
    halluc = hallucination_rate(reponse, contexte)
    assert abs(faith + halluc - 1.0) < 1e-9


def test_hallucination_reponse_inventee():
    """Une réponse complètement inventée → hallucination élevée."""
    contexte = ["Contenu académique sur le machine learning."]
    reponse = "La capitale de la France est Paris."
    halluc = hallucination_rate(reponse, contexte)
    assert halluc > 0.5
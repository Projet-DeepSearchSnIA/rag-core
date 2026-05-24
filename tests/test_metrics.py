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


from rag_core.utils.metrics import (
    mrr_at_k,
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    faithfulness_score,
    hallucination_rate,
)



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
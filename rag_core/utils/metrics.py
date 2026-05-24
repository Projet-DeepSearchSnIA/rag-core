import math


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

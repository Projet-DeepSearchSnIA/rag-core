# Tests rag-core

Ce dossier contient tous les tests unitaires du projet. L'idée c'est d'avoir
une suite de tests rapides qu'on peut lancer avant chaque commit pour s'assurer
qu'on n'a rien cassé. Pas de réseau, pas de clé API — juste du Python pur.

---

## Lancer les tests

```bash
# Installer le projet en mode éditable (une seule fois)
pip install -e ".[dev]"

# Tous les tests unitaires
pytest tests/ -v

# Un fichier précis
pytest tests/test_preprocessor.py -v

# Un test précis
pytest tests/test_chunk_optimizer.py::test_grands_chunks_decoupes -v

# Avec couverture de code
pytest tests/ --cov=rag_core --cov-report=term-missing

# Tests d'intégration (réseau requis, exclus par défaut)
pytest tests/ -m integration -v

# Exclure les intégrations (comportement par défaut recommandé en CI)
pytest tests/ -m "not integration" -v
```

---

## Structure des tests

```
tests/
├── conftest.py              ← fixtures partagées (make_doc, make_chunk)
├── test_extraction.py       ← schemas de données, PDFExtractor
├── test_chunking.py         ← SmartTextSplitter, DocumentChunk
├── test_preprocessor.py     ← TextPreprocessor (nettoyage texte)
├── test_chunk_optimizer.py  ← ChunkOptimizer (qualité chunks)
├── test_prompt_template.py  ← PromptTemplates, routage par type de question
├── test_enriched_chunk.py   ← EnrichedChunk, méthodes privées PineconeRetriever
├── test_metrics.py          ← métriques d'évaluation RAG (MRR, Recall, NDCG...)
└── README.md                ← ce fichier
```

---

## Ce que chaque fichier teste

### conftest.py — Fixtures partagées

Deux fonctions utilitaires disponibles partout :

```python
make_doc(pages_text: list[str]) -> ExtractedDocument
```
Construit un document synthétique à partir d'une liste de textes. Chaque élément
devient une page avec un seul bloc texte. On s'en sert pour tester le chunking
sans avoir besoin d'un vrai PDF.

```python
make_chunk(content, document_id, page_numbers, metadata) -> DocumentChunk
```
Construit un chunk minimal pour tester l'optimiseur sans passer par le splitter.

Trois fixtures pytest prêtes à l'emploi :
- `doc_simple` — un document d'une page
- `doc_multi_pages` — trois pages avec du contenu varié
- `doc_long` — beaucoup de texte pour forcer plusieurs chunks

---

### test_extraction.py — 8 tests

Teste les dataclasses (`ContentBlock`, `ExtractedDocument`, `BoundingBox`, `DocumentMetadata`)
et l'initialisation de `PDFExtractor`. On ne teste pas `extract_pdf()` car ça
nécessite un vrai PDF et docTR installé.

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_pdf_extractor_init_sans_callback` | `upload_callback` vaut `None` par défaut |
| `test_pdf_extractor_init_avec_callback` | Le callback est bien conservé |
| `test_extracted_document_create_new` | `create_new()` initialise `pages=[]` et `total_pages=0` |
| `test_extracted_document_id_unique` | Deux docs ont des `document_id` différents (UUID) |
| `test_content_block_optionnel` | `bbox`, `metadata`, `image_id` valent `None` par défaut |
| `test_bounding_box` | Les coordonnées sont bien stockées |
| `test_document_metadata_defaut` | `title=None`, `author=[]`, `is_public=False` |
| `test_extractor_callback_non_appele_sans_image` | Le callback n'est pas déclenché à l'init |

---

### test_chunking.py — 7 tests

Teste `SmartTextSplitter` et `DocumentChunk`. La fonction helper `_make_doc` est
définie localement dans ce fichier (et aussi dans `conftest.py` pour les autres).

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_chunk_document_simple` | Un texte court produit ≥ 1 chunk |
| `test_chunk_preserve_document_id` | Tous les chunks héritent du `document_id` |
| `test_chunk_total_chunks_coherent` | `chunk.total_chunks == len(chunks)` pour tous |
| `test_chunk_doc_vide` | Document sans pages → liste vide |
| `test_chunk_to_dict_complet` | `to_dict()` contient les clés attendues |
| `test_strategy_mixed` | Stratégie `mixed` produit des chunks sans erreur |
| `test_splitter_defaut_recursive` | Paramètres par défaut corrects |

---

### test_preprocessor.py — 13 tests

Teste `TextPreprocessor` qui nettoie le texte extrait avant le chunking.
Toutes les règles sont testées indépendamment.

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_blocs_vides_supprimes` | Blocs vides ou espaces seuls sont filtrés |
| `test_normalisation_espaces_multiples` | Plusieurs espaces → un seul |
| `test_trait_union_fin_de_ligne_colle_les_mots` | `ap-\nprentissage` → `apprentissage` |
| `test_formule_latex_non_touchee` | Le contenu des blocs `formula` est intouché |
| `test_guillemets_typographiques_remplaces` | `'` `'` → `'` |
| `test_tirets_longs_remplaces` | `–` `—` → `-` |
| `test_urls_extraites_dans_metadata` | URLs dans `block.metadata['urls']` |
| `test_emails_extraits_dans_metadata` | Emails dans `block.metadata['emails']` |
| `test_patrons_repetes_supprimes` | Header/footer répété > 3 fois supprimé |
| `test_tableau_lignes_vides_supprimees` | Tables : lignes vides internes supprimées |
| `test_sans_config_les_defauts_sont_actifs` | Toutes les règles actives par défaut |
| `test_config_desactive_extraction_urls` | On peut désactiver via config |
| `test_plusieurs_blocs_independants` | Plusieurs blocs valides tous retournés |
| `test_bloc_trop_court_apres_nettoyage_supprime` | Caractères de contrôle → bloc supprimé |

---

### test_chunk_optimizer.py — 13 tests

Teste `ChunkOptimizer` qui améliore la qualité des chunks avant l'indexation.
Chaque passe (remove_empty, deduplicate, merge_small, split_large) est testée seule.

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_liste_vide_ne_plante_pas` | `optimize_chunks([])` → `([], stats)` sans erreur |
| `test_chunks_vides_supprimes` | Contenu vide ou ≤ 10 chars est éliminé |
| `test_doublons_presque_identiques_supprimes` | Jaccard ≥ 0.9 → dédupliqué |
| `test_doublons_textes_differents_conserves` | Textes différents → tous conservés |
| `test_petits_chunks_fusionnes` | Chunks < min_size sont fusionnés |
| `test_grands_chunks_decoupes` | Chunks > max_size sont découpés |
| `test_chunk_avec_formule_pas_decoupe` | Formules LaTeX → jamais découpées |
| `test_reindexation_apres_optimisation` | `chunk_index` et `total_chunks` cohérents |
| `test_stats_retournees_correctes` | Stats dict avec les bonnes clés |
| `test_analyze_chunks_cles_attendues` | `analyze_chunks()` retourne toutes les clés |
| `test_analyze_chunks_vide` | `analyze_chunks([])` → `{'total': 0}` |
| `test_similarite_texte_identique` | Textes identiques → similarité = 1.0 |
| `test_similarite_texte_sans_overlap` | Aucun mot commun → similarité = 0.0 |
| `test_similarite_texte_vide` | Texte vide → similarité = 0.0 |
| `test_fusion_preserver_pages` | Pages de tous les chunks fusionnés conservées |

---

### test_prompt_template.py — 19 tests

Teste `PromptTemplates` (formatage du contexte, construction des prompts, parsing
des métadonnées dans les réponses LLM) et `get_template_for_question_type`.

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_format_context_contient_le_texte_du_chunk` | Texte présent dans le contexte |
| `test_format_context_contient_le_nom_du_document` | Nom du doc dans le contexte |
| `test_format_context_limite_max_chunks` | `max_chunks=2` → exactement 2 blocs |
| `test_format_context_pages_liste` | Pages de type liste formatées |
| `test_format_context_avec_score` | `include_scores=True` → "pertinence" visible |
| `test_build_rag_prompt_contient_la_question` | Question de l'utilisateur dans le prompt |
| `test_build_rag_prompt_contient_le_contexte` | Contexte inclus dans le prompt |
| `test_build_rag_prompt_template_par_defaut` | Template RAG_WITH_SOURCES utilisé par défaut |
| `test_build_chat_messages_roles_corrects` | Messages avec rôles `system` et `user` |
| `test_build_chat_messages_question_dans_user` | Question dans le message `user` |
| `test_build_chat_messages_historique_inclus` | Historique inséré entre system et user |
| `test_extraction_sources_pattern_standard` | `[Source: doc.pdf, page 3]` extrait |
| `test_extraction_sources_plusieurs` | Plusieurs citations extraites |
| `test_extraction_sources_aucune` | Pas de citation → liste vide |
| `test_extraction_metadata_blocks_sources_used` | `SOURCES_USED: [...]` parsé |
| `test_extraction_metadata_blocks_follow_up` | `FOLLOW_UP_QUESTIONS: [...]` parsé |
| `test_extraction_metadata_blocks_clean_response` | Blocs meta supprimés de la réponse |
| `test_template_question_comparaison` | "comparer" → template comparaison |
| `test_template_question_explication` | "comment" → template explication |
| `test_template_question_resume` | "résumé" → template synthèse |
| `test_template_question_factuelle` | "qui" → template factuel |
| `test_template_question_par_defaut` | Pas de mot-clé → RAG_WITH_SOURCES |
| `test_format_response_with_sources_structure` | Dict avec toutes les clés attendues |

---

### test_enriched_chunk.py — 13 tests

Teste `EnrichedChunk` et les méthodes privées de `PineconeRetriever` qui préparent
les chunks. On appelle les méthodes statiques-like avec `None` comme `self` car
elles ne lisent pas `self` (pattern déjà établi dans `test_retrieval.py`).

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_create_enriched_chunk_champs_de_base` | `chunk_id`, `text`, `score` corrects |
| `test_create_enriched_chunk_formules_depuis_json` | `formulas_latex` parsé depuis JSON |
| `test_create_enriched_chunk_images_depuis_csv` | `image_ids` parsé depuis CSV |
| `test_create_enriched_chunk_metadata_vide` | Valeurs par défaut sans crash |
| `test_create_enriched_chunk_rerank_score_none` | `rerank_score=None` quand absent |
| `test_to_dict_contient_toutes_les_cles` | Toutes les clés documentées présentes |
| `test_normalize_metadata_listes_json` | JSON strings → listes Python |
| `test_normalize_metadata_dict_vide` | `{}` → `{}` |
| `test_normalize_metadata_none` | `None` → `{}` |
| `test_format_for_llm_sans_chunks` | Liste vide → message "aucun contexte" |
| `test_format_for_llm_contient_nom_document` | Nom du doc dans la sortie LLM |
| `test_format_for_llm_contient_le_texte` | Texte du chunk dans la sortie LLM |
| `test_format_for_llm_avec_formules` | Formules listées dans la sortie LLM |

---

### test_metrics.py — 21 tests

Définit les fonctions de métriques RAG et les teste sur des cas à résultat connu.
Ces fonctions sont aussi la référence documentaire pour les labs d'évaluation.

**Fonctions définies dans ce fichier :**

```python
mrr_at_k(retrieved_ids, relevant_ids, k) -> float
recall_at_k(retrieved_ids, relevant_ids, k) -> float
precision_at_k(retrieved_ids, relevant_ids, k) -> float
ndcg_at_k(retrieved_ids, relevant_ids, k) -> float
faithfulness_score(response, context_chunks) -> float
hallucination_rate(response, context_chunks) -> float
```

**Seuils attendus dans les expériences labs :**

| Métrique | Seuil minimum | Objectif cible |
|----------|--------------|----------------|
| MRR@5 | > 0.50 | > 0.75 |
| Recall@5 | > 0.60 | > 0.85 |
| Recall@10 | > 0.75 | > 0.90 |
| NDCG@5 | > 0.55 | > 0.70 |
| Precision@1 | > 0.40 | > 0.65 |
| faithfulness | > 0.80 | > 0.95 |
| hallucination_rate | < 0.20 | < 0.05 |

| Test | Ce qu'on vérifie |
|------|-----------------|
| `test_mrr_premier_resultat_pertinent` | Premier pertinent → MRR = 1.0 |
| `test_mrr_deuxieme_resultat_pertinent` | Deuxième pertinent → MRR = 0.5 |
| `test_mrr_troisieme_resultat_pertinent` | Troisième pertinent → MRR = 1/3 |
| `test_mrr_aucun_pertinent` | Pas de pertinent → MRR = 0 |
| `test_mrr_pertinent_hors_k` | Pertinent au-delà de k → non compté |
| `test_mrr_plusieurs_pertinents_premier_compte` | Seul le rang du premier compte |
| `test_recall_tous_retrouves` | Tous retrouvés → Recall = 1 |
| `test_recall_moitie_retrouvee` | La moitié → Recall = 0.5 |
| `test_recall_aucun_retrouve` | Rien retrouvé → Recall = 0 |
| `test_recall_liste_vide_pertinents` | Rien attendu → Recall = 1 (convention) |
| `test_precision_tous_pertinents` | Tous retournés pertinents → Precision = 1 |
| `test_precision_aucun_pertinent` | Aucun pertinent → Precision = 0 |
| `test_precision_moitie` | Un sur deux pertinent → Precision@2 = 0.5 |
| `test_precision_k_zero` | k=0 → 0 sans division par zéro |
| `test_ndcg_classement_parfait` | Ordre optimal → NDCG = 1 |
| `test_ndcg_aucun_pertinent` | Aucun pertinent → NDCG = 0 |
| `test_ndcg_ordre_degrade_moins_bon_que_parfait` | Rang 2 < rang 1 |
| `test_ndcg_liste_vide_pertinents` | Pas de pertinent attendu → 0 |
| `test_faithfulness_reponse_dans_contexte` | Réponse ancrée → score élevé |
| `test_faithfulness_reponse_hors_contexte` | Réponse inventée → score bas |
| `test_faithfulness_reponse_vide` | Réponse vide → 0 sans erreur |
| `test_hallucination_complement_faithfulness` | `halluc + faith == 1.0` |
| `test_hallucination_reponse_inventee` | Hallucination élevée pour réponse inventée |

---

## Résumé couverture

| Fichier | Tests | Modules couverts |
|---------|-------|-----------------|
| test_extraction.py | 8 | PDFExtractor, tous les dataclasses |
| test_chunking.py | 7 | SmartTextSplitter, DocumentChunk |
| test_preprocessor.py | 14 | TextPreprocessor |
| test_chunk_optimizer.py | 15 | ChunkOptimizer |
| test_prompt_template.py | 23 | PromptTemplates, routage question |
| test_enriched_chunk.py | 13 | EnrichedChunk, méthodes privées Retriever |
| test_metrics.py | 23 | MRR, Recall, Precision, NDCG, faithfulness |
| **Total** | **103** | |

Tous les tests s'exécutent en < 10 secondes sans réseau.

---

## Patterns utilisés dans les tests

### Appel de méthodes privées Python

Certaines méthodes critiques sont privées (`_parse_json_field`, `_truncate_for_rerank`,
etc.). On les teste directement en passant `None` comme `self` quand elles ne lisent
pas d'attributs d'instance :

```python
result = PineconeRetriever._truncate_for_rerank(None, "texte long", max_tokens=200)
```

C'est acceptable ici parce que ces méthodes encapsulent de la logique critique
(ratio tokens/chars, parsing JSON) qu'on ne peut pas tester autrement sans un vrai
index Pinecone.

### Construire un document synthétique

```python
from conftest import make_doc

doc = make_doc(["Paragraphe 1.", "Paragraphe 2.", "Conclusion."])
```

Chaque string devient une page. Utile pour tester le chunking sans PDF.

### Construire un chunk synthétique

```python
from conftest import make_chunk

chunk = make_chunk(
    "Contenu du chunk.",
    document_id="mon-doc",
    page_numbers=[1, 2],
    metadata={"has_formulas": True}
)
```

Utile pour tester l'optimiseur en isolation.

### Simuler un chunk Pinecone (dict)

```python
chunk_dict = {
    "text": "L'attention multi-tête...",
    "score": 0.9,
    "metadata": {"document_name": "paper.pdf", "page_numbers": [3]},
}
```

`PromptTemplates.format_context()` accepte des dicts comme des objets — les deux
formats sont testés.

---

## Ajouter un nouveau test

Checklist rapide avant de committer :

- Le test passe sans réseau ni clé API
- Le nom du test décrit ce qui est vérifié (pas comment)
- Si réseau requis → `@pytest.mark.integration`
- Utilise `make_doc()` pour les fixtures de chunking
- Pas de `print()` → utiliser `assert` avec un message d'erreur clair

```python
def test_mon_nouveau_comportement():
    """Une phrase qui explique POURQUOI ce comportement est important."""
    # setup
    doc = make_doc(["Contenu de test."])
    splitter = SmartTextSplitter(chunk_size=500)

    # action
    chunks = splitter.split_document(doc)

    # assertion avec message d'erreur utile
    assert len(chunks) >= 1, "Un document non-vide doit produire au moins un chunk"
```

---

## Tests d'intégration (réseau requis)

Marqués `@pytest.mark.integration` et exclus par défaut. À lancer manuellement
avec les bonnes variables d'environnement :

```bash
export PINECONE_API_KEY="..."
export HF_TOKEN="..."
pytest tests/ -m integration -v
```

```python
import pytest
import os

@pytest.mark.integration
def test_upload_et_retrieve_reel():
    """Cycle complet upload → retrieve sur Pinecone réel."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY non définie")
    # ... test complet

@pytest.mark.integration
def test_extract_vrai_pdf():
    """Extraction sur un vrai PDF — nécessite docTR installé (~30s)."""
    pytest.skip("Nécessite un PDF et docTR")
```

---

## Lien avec les labs

```
rag-core/tests/          ← invariants : "le code fait ce qu'il dit"
lab-retrieval/eval/      ← performance : "le système est-il bon ?"
rag-eval/metrics/        ← fonctions de métriques partagées entre labs
```

Les métriques dans `test_metrics.py` sont la version de référence.
Les labs les importeront depuis `rag-eval` une fois ce module créé.

---

## Configuration pytest

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: tests nécessitant un accès réseau ou des clés API",
    "slow: tests longs (> 10s)",
]
addopts = "-v --tb=short"
```
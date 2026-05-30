# Tests rag-core

Suite de tests pytest couvrant le pipeline RAG complet. Deux niveaux :

- **Tests rapides** (152) sans réseau ni clé API — lancés par défaut, < 3 minutes
- **Tests live** (13) qui appellent réellement Pinecone et HuggingFace — `@pytest.mark.live`, lancés explicitement

Aucun test ne dépend de valeurs hardcodées : tous les modèles, dimensions et
hyperparamètres sont lus depuis `configs/baseline.yaml` ; les secrets et
identifiants de déploiement viennent de `.env`.

---

## Lancer les tests

```bash
# Tous les tests rapides (par défaut)
pytest tests/

# Un fichier précis
pytest tests/test_chunking.py -v

# Un test précis
pytest tests/test_chunk_optimizer.py::test_grands_chunks_decoupes -v

# Avec couverture
pytest tests/ --cov=rag_core --cov-report=term-missing

# Tests live uniquement (réseau requis)
pytest tests/ -m live

# Tout (rapides + live)
pytest tests/ -m "live or not live"
```

Via uv :

```bash
uv run pytest tests/
```

---

## Structure

```
tests/
├── conftest.py                  fixtures partagées (make_doc, make_chunk, load_baseline, live_*)
├── test_extraction.py           PDFExtractor, dataclasses d'extraction
├── test_chunking.py             SmartTextSplitter, DocumentChunk
├── test_chunk_optimizer.py      ChunkOptimizer (qualité chunks)
├── test_preprocessor.py         TextPreprocessor (nettoyage texte)
├── test_retrieval.py            PineconeRetriever (parsing JSON/CSV, truncate)
├── test_retriever_methods.py    construction EnrichedChunk, normalisation métadonnées, format LLM
├── test_prompt_template.py      PromptTemplates, routage par type de question
├── test_metrics.py              métriques d'évaluation RAG (MRR, Recall, NDCG, faithfulness)
├── test_integration.py          chaînage splitter -> optimizer -> upload -> retrieve, RAGPipeline mocké
├── test_scripts_smoke.py        sanité du CLI unifié scripts/rag.py
├── test_live.py                 tests live Pinecone + HuggingFace (@pytest.mark.live)
└── README.md                    ce fichier
```

---

## Helpers (conftest.py)

Quatre fonctions utilitaires disponibles dans tous les fichiers de test :

```python
make_doc(pages_text: list[str]) -> ExtractedDocument
```
Construit un document synthétique. Chaque string devient une page avec un seul
bloc texte. Utile pour tester le chunking sans PDF réel.

```python
make_chunk(content, document_id, page_numbers, metadata) -> DocumentChunk
```
Construit un `DocumentChunk` typé avec `ChunkMetadata` reconstruit depuis le
dict passé (via `ChunkMetadata.from_dict`). Utile pour l'optimizer en isolation.

```python
load_baseline() -> dict
```
Lit `configs/baseline.yaml` — source unique pour les modèles et hyperparamètres
dans les tests.

```python
_retriever_vide() -> PineconeRetriever
```
Instance `PineconeRetriever` sans `__init__` (pas de réseau). Sert à tester les
méthodes privées en isolation.

Fixtures de session pour les tests live :

- `pinecone_creds` — `(api_key, index_name)` depuis `.env`, skip si absent
- `hf_token` — token HuggingFace, skip si absent
- `baseline_cfg` — `configs/baseline.yaml` parsé, scope session
- `live_retriever` — `PineconeRetriever` réel branché sur l'index live
- `live_llm` — `LLMHandler` réel branché sur HuggingFace

---

## Ce que chaque fichier teste

### test_extraction.py — 8 tests

Dataclasses (`ContentBlock`, `ExtractedDocument`, `BoundingBox`,
`DocumentMetadata`) et instanciation de `PDFExtractor`. Les tests qui
construisent un extractor passent par `load_baseline()["extraction"]` plutôt
qu'un dict hardcodé.

| Test | Vérifie |
|------|---------|
| `test_pdf_extractor_init_sans_callback` | `upload_callback` vaut `None` quand non passé |
| `test_pdf_extractor_init_avec_callback` | Le callback est conservé |
| `test_extracted_document_create_new` | `create_new()` produit `pages=[]`, `total_pages=0` |
| `test_extracted_document_id_unique` | Deux docs ont des UUID différents |
| `test_content_block_optionnel` | `bbox`, `metadata`, `image_id` valent `None` par défaut |
| `test_bounding_box` | Coordonnées stockées correctement |
| `test_document_metadata_defaut` | `title=None`, `author=[]`, `is_public=False` |
| `test_extractor_callback_non_appele_sans_image` | Callback non déclenché à l'init |

---

### test_chunking.py — 7 tests

`SmartTextSplitter` et `DocumentChunk` (dataclass définie dans `chunk_schemas`).

| Test | Vérifie |
|------|---------|
| `test_chunk_document_simple` | Un texte court produit au moins un chunk |
| `test_chunk_preserve_document_id` | Tous les chunks héritent du `document_id` du doc source |
| `test_chunk_total_chunks_coherent` | `chunk.total_chunks == len(chunks)` partout |
| `test_chunk_doc_vide` | Document sans pages produit une liste vide |
| `test_chunk_to_dict_complet` | `to_dict()` contient les clés attendues |
| `test_strategy_mixed` | Stratégie `mixed` produit des chunks sans erreur |
| `test_splitter_defaut_recursive` | Paramètres par défaut corrects |

---

### test_chunk_optimizer.py — 15 tests

`ChunkOptimizer` : remove_empty, deduplicate, merge_small, split_large, reindex.
Chaque passe est testée seule. Avec le passage à `ChunkMetadata` typé, l'accès
aux flags est désormais attribut-style (`chunk.metadata.has_formulas`) plutôt
que dict-style.

| Test | Vérifie |
|------|---------|
| `test_liste_vide_ne_plante_pas` | `optimize_chunks([])` renvoie `([], stats)` |
| `test_chunks_vides_supprimes` | Contenu vide ou ≤ 10 chars éliminé |
| `test_doublons_presque_identiques_supprimes` | Jaccard ≥ 0.9 → dédupliqué |
| `test_doublons_textes_differents_conserves` | Textes différents → conservés |
| `test_petits_chunks_fusionnes` | Chunks < `min_size` fusionnés |
| `test_grands_chunks_decoupes` | Chunks > `max_size` découpés |
| `test_chunk_avec_formule_pas_decoupe` | Chunks avec `has_formulas=True` non découpés |
| `test_reindexation_apres_optimisation` | `chunk_index` et `total_chunks` cohérents |
| `test_stats_retournees_correctes` | Stats dict avec les clés attendues |
| `test_analyze_chunks_cles_attendues` | `analyze_chunks()` retourne toutes les clés |
| `test_analyze_chunks_vide` | `analyze_chunks([])` renvoie `{'total': 0}` |
| `test_similarite_texte_identique` | Textes identiques → similarité = 1.0 |
| `test_similarite_texte_sans_overlap` | Aucun mot commun → similarité = 0.0 |
| `test_similarite_texte_vide` | Texte vide → similarité = 0.0 |
| `test_fusion_preserver_pages` | Pages de tous les chunks fusionnés conservées |

---

### test_preprocessor.py — 14 tests

`TextPreprocessor` (nettoyage du texte extrait avant chunking). Chaque règle de
nettoyage est testée indépendamment.

| Test | Vérifie |
|------|---------|
| `test_blocs_vides_supprimes` | Blocs vides ou espaces seuls filtrés |
| `test_normalisation_espaces_multiples` | Plusieurs espaces deviennent un seul |
| `test_trait_union_fin_de_ligne_colle_les_mots` | `ap-\nprentissage` → `apprentissage` |
| `test_formule_latex_non_touchee` | Le contenu des blocs `formula` est intouché |
| `test_guillemets_typographiques_remplaces` | Guillemets courbes normalisés |
| `test_tirets_longs_remplaces` | `–` `—` deviennent `-` |
| `test_urls_extraites_dans_metadata` | URLs détectées et placées dans `metadata['urls']` |
| `test_emails_extraits_dans_metadata` | Emails détectés et placés dans `metadata['emails']` |
| `test_patrons_repetes_supprimes` | Header/footer répété plus de 3 fois supprimé |
| `test_tableau_lignes_vides_supprimees` | Tables : lignes vides internes supprimées |
| `test_sans_config_les_defauts_sont_actifs` | Toutes les règles actives par défaut |
| `test_config_desactive_extraction_urls` | Désactivation possible via config |
| `test_plusieurs_blocs_independants` | Plusieurs blocs valides tous retournés |
| `test_bloc_trop_court_apres_nettoyage_supprime` | Caractères de contrôle seuls → bloc supprimé |

---

### test_retrieval.py — 9 tests

Méthodes utilitaires de `PineconeRetriever` (parsing JSON, CSV, troncature). Ces
méthodes sont privées mais critiques car elles décident comment les
métadonnées Pinecone sont reconstruites côté client.

---

### test_retriever_methods.py — 13 tests

Méthodes de `PineconeRetriever` qui construisent et formatent les chunks pour
le LLM : `_create_enriched_chunk`, `_normalize_metadata`, `format_for_llm`.

| Test | Vérifie |
|------|---------|
| `test_create_enriched_chunk_champs_de_base` | `chunk_id`, `text`, `score` corrects |
| `test_create_enriched_chunk_formules_depuis_json` | `formulas_latex` parsé depuis JSON string |
| `test_create_enriched_chunk_images_depuis_csv` | `image_ids` parsé depuis CSV string |
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

### test_prompt_template.py — 33 tests

`PromptTemplates` (formatage du contexte, construction des messages chat,
parsing des sources dans les réponses LLM) et `get_template_for_question_type`
(routage par type de question : factuelle, comparaison, explication, synthèse).

Couvre le formatage du contexte (groupement par document, limite de chunks,
inclusion ou non des scores), la construction du prompt RAG, le parsing
post-réponse pour extraire sources et follow-ups, et le routage vers le bon
template selon les mots-clés de la question.

---

### test_metrics.py — 23 tests

Métriques d'évaluation RAG : `mrr_at_k`, `recall_at_k`, `precision_at_k`,
`ndcg_at_k`, `faithfulness_score`, `hallucination_rate`. Ces fonctions
servent de référence partagée entre les labs d'évaluation.

Seuils attendus dans les expériences labs :

| Métrique | Seuil minimum | Objectif cible |
|----------|---------------|----------------|
| MRR@5 | > 0.50 | > 0.75 |
| Recall@5 | > 0.60 | > 0.85 |
| Recall@10 | > 0.75 | > 0.90 |
| NDCG@5 | > 0.55 | > 0.70 |
| Precision@1 | > 0.40 | > 0.65 |
| faithfulness | > 0.80 | > 0.95 |
| hallucination_rate | < 0.20 | < 0.05 |

---

### test_integration.py — 18 tests

Tests d'intégration end-to-end avec uniquement Pinecone et HuggingFace mockés.
Chunking, optimisation, sérialisation et préparation des métadonnées sont
exercés réellement. L'objectif est de détecter les régressions silencieuses
entre upload et retrieval.

Quatre classes :

- `TestChunkingPipeline` (4 tests) — la chaîne splitter → optimizer produit
  des chunks cohérents : contenu préservé, IDs uniques, indices contigus,
  métadonnées document propagées.
- `TestJsonSerialisation` (2 tests) — l'aller-retour `save_chunks` / reload
  donne le même nombre de chunks, et le JSON contient toutes les clés que
  `_prepare_metadata` lit côté upload.
- `TestMetadataContract` (4 tests) — **le test le plus important** : on
  simule le voyage complet d'un chunk JSON via `_prepare_metadata` →
  `_sanitize_metadata` → `_normalize_metadata` → `_create_enriched_chunk`. Si
  ce test casse, c'est que les schémas amont et aval ont divergé.
- `TestLLMHandlerMocked` (4 tests) — logique de génération (retry sur 502,
  format de sortie, échec gracieux) sans appel réseau.
- `TestRAGPipelineMocked` (4 tests) — orchestration retriever + LLM,
  fallback si aucun chunk, validation que `(retriever=, llm=)` sont
  keyword-only obligatoires.

---

### test_scripts_smoke.py — 12 tests

Sanité du CLI unifié `scripts/rag.py`. Vérifie l'import, le `--help` global,
le `--help` de chaque sous-commande (`extract`, `chunk`, `upload`, `index`,
`retrieve`, `ask`), l'échec propre sans sous-commande, et le roundtrip
`ExtractedDocument.to_dict() / from_dict()`.

Le timeout subprocess est lu depuis `configs/baseline.yaml`
(`tests.smoke_subprocess_timeout_seconds`) — pas hardcodé.

---

### test_live.py — 13 tests (@pytest.mark.live)

Tests qui appellent réellement Pinecone et HuggingFace. Lancés explicitement
avec `pytest -m live`. Skip automatique si `.env` n'a pas les clés.

| Classe | Ce qui est testé |
|--------|------------------|
| `TestLiveRetrieval` | recherche vectorielle + rerank sur l'index réel |
| `TestLiveGeneration` | inférence LLM réelle (Llama 3.1 via HuggingFace) |
| `TestLiveE2E` | pipeline complet retrieve → generate, question hors-domaine, streaming |

---

## Résumé couverture

| Fichier | Tests | Modules couverts |
|---------|-------|------------------|
| test_extraction.py | 8 | PDFExtractor, dataclasses extraction |
| test_chunking.py | 7 | SmartTextSplitter, DocumentChunk |
| test_chunk_optimizer.py | 15 | ChunkOptimizer |
| test_preprocessor.py | 14 | TextPreprocessor |
| test_retrieval.py | 9 | PineconeRetriever (parsing : truncate, parse_json, parse_list) |
| test_retriever_methods.py | 13 | PineconeRetriever (enrichissement : create_chunk, normalize, format_for_llm) |
| test_prompt_template.py | 33 | PromptTemplates, groupement chunks, normalize_pages, routage question |
| test_metrics.py | 23 | MRR, Recall, Precision, NDCG, faithfulness |
| test_integration.py | 18 | chaînage splitter → optimizer → upload → retrieve, RAGPipeline mocké |
| test_scripts_smoke.py | 12 | scripts/rag.py (CLI unifiée) |
| **Total non-live** | **152** | tous modules sauf appels réseau |
| test_live.py | 13 | Pipeline complet Pinecone + HuggingFace (@pytest.mark.live) |

Les 152 tests rapides s'exécutent sans réseau ni clé API.

---

## Patterns utilisés

### Appel de méthodes privées sans réseau

Certaines méthodes critiques sont privées (`_parse_json_field`,
`_truncate_for_rerank`, etc.). On les teste directement en bypassant `__init__`
via `object.__new__(PineconeRetriever)`, parce que ces méthodes encapsulent
de la logique critique (ratio tokens/chars, parsing JSON) qu'on ne peut pas
tester autrement sans un vrai index Pinecone.

### Construire un document synthétique

```python
from tests.conftest import make_doc

doc = make_doc(["Paragraphe 1.", "Paragraphe 2.", "Conclusion."])
```

### Construire un chunk synthétique avec métadonnées typées

```python
from tests.conftest import make_chunk

chunk = make_chunk(
    "Contenu du chunk.",
    document_id="mon-doc",
    page_numbers=[1, 2],
    metadata={"has_formulas": True, "section_title": "Introduction"},
)
# chunk.metadata est un ChunkMetadata, pas un dict
```

### Charger la config baseline dans un test

```python
from tests.conftest import load_baseline

cfg = load_baseline()
extractor = PDFExtractor(config=cfg["extraction"])
```

---

## Ajouter un nouveau test

Checklist :

- Le test passe sans réseau ni clé API (sinon `@pytest.mark.live`)
- Le nom décrit ce qui est vérifié, pas comment
- Utilise `make_doc()` / `make_chunk()` pour les fixtures
- Aucun défaut hardcodé : lit depuis `load_baseline()` ou `os.getenv()`
- Pas de `print()` — `logger.warning()` ou `assert` avec message clair

```python
def test_mon_nouveau_comportement():
    """Une phrase qui explique POURQUOI ce comportement est important."""
    doc = make_doc(["Contenu de test."])
    splitter = SmartTextSplitter(chunk_size=500, chunk_overlap=0, strategy="recursive")

    chunks = splitter.split_document(doc)

    assert len(chunks) >= 1, "Un document non-vide doit produire au moins un chunk"
```

---

## Tests live (réseau requis)

Marqués `@pytest.mark.live` et exclus par défaut. Lancement manuel :

```bash
pytest -m live -v

# Seulement le retrieval
pytest -m live -v -k retrieval

# Pipeline complet
pytest -m live -v -k e2e
```

Les fixtures `live_retriever` et `live_llm` font un skip automatique si les
clés `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` ou `HF_TOKEN` sont absentes
du `.env`, donc pas besoin de les gérer dans chaque test.

---

## Lien avec les labs

```
rag-core/tests/          invariants : "le code fait ce qu'il dit"
lab-retrieval/eval/      performance : "le système est-il bon ?"
rag-eval/metrics/        fonctions de métriques partagées entre labs
```

Les métriques dans `test_metrics.py` sont la version de référence. Les labs
les importeront depuis `rag-eval` une fois ce module créé.

---

## Configuration pytest

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "-m 'not live'"
markers = [
    "live: nécessite les clés API du .env (Pinecone et/ou HuggingFace)",
]
```

Par défaut, `pytest` lance uniquement les 152 tests rapides sans réseau.
`pytest -m live` pour activer les 13 tests live.

# scripts/

CLI  pour piloter le pipeline rag-core en ligne de commande. Tout passe
par un seul point d'entrée — `scripts/rag.py` — qui expose six sous-commandes
correspondant chacune à une étape isolable du pipeline.

Le but de cette organisation est de permettre à l'utilisateur de tester ses
propres PDFs et de cibler précisément l'endroit où une régression apparaît,
sans forcément ré-exécuter tout le pipeline à chaque fois.

---

## Sous-commandes

| Commande | Étape | Entrée | Sortie | API externe |
|----------|-------|--------|--------|-------------|
| `extract` | extraction PDF | PDF | JSON `ExtractedDocument` | — |
| `chunk` | segmentation | JSON `ExtractedDocument` | JSON chunks | — |
| `upload` | indexation | JSON chunks | Pinecone | Pinecone |
| `index` | pipeline d'ingestion complet | PDF | Pinecone | Pinecone |
| `retrieve` | recherche | question texte | JSON `EnrichedChunk[]` | Pinecone |
| `ask` | Q/R complet | question texte | réponse + sources | Pinecone + HuggingFace |

L'idée : `index` est le raccourci `extract` + `chunk` + `upload`. `ask` est le
raccourci `retrieve` + génération LLM. Les sous-commandes atomiques permettent
de geler un état intermédiaire (`extract` → JSON, `chunk` → JSON) et de
réindexer ou requêter sans repasser par les étapes coûteuses.

---

## Convention d'arguments

Toutes les sous-commandes :

- `--config` est **obligatoire**
- Si une clé manque dans le YAML ou dans `.env`, la commande log une erreur
  explicite et termine avec `exit(2)`. Le pipeline ne s'exécute jamais avec des
  valeurs implicites.
- Les valeurs `--index` et `--namespace` sont passées en CLI uniquement (jamais
  lues d'une variable d'environnement par la CLI elle-même), pour éviter toute
  ambiguïté entre dev et prod.

---

## extract — PDF en JSON

Lit un PDF, en sort un `ExtractedDocument` sérialisé en JSON. Aucun appel
réseau : utile pour comparer plusieurs configs d'extraction sur le même document.

```bash
python scripts/rag.py extract mon.pdf \
    --config configs/baseline.yaml \
    --out data/extracted/mon.json
```

Lit dans baseline.yaml : section `[extraction]` complète (pymupdf, doctr, math_ocr, preprocessing, output_dir, temp_dir).

Le JSON produit contient : metadata du document, pages avec leurs `content_blocks` (texte, titres, formules, images), table of contents, statistiques d'extraction.

---

## chunk — JSON extrait en chunks

Reprend un JSON produit par `extract` et applique la stratégie de chunking
(recursive / semantic / mixed) + l'optimizer.

```bash
python scripts/rag.py chunk data/extracted/mon.json \
    --config configs/baseline.yaml \
    --out data/chunks/mon_chunks.json
```

Lit dans baseline.yaml : section `[chunking]` (`chunk_size`, `chunk_overlap`, `strategy`, `optimizer_enabled`).

Le JSON produit contient : `total_chunks` et la liste des chunks avec `chunk_id`, `content`, `document_id`, `page_numbers`, `metadata` typées (`ChunkMetadata`).

---

## upload — Chunks JSON vers Pinecone

Reprend un JSON produit par `chunk` et l'envoie dans un index Pinecone.
L'embedding est calculé côté Pinecone Inference (pas localement).

```bash
python scripts/rag.py upload data/chunks/mon_chunks.json \
    --config configs/baseline.yaml \
    --index mon-index --namespace default
```

Lit dans baseline.yaml : `[embedding]` (`model`), `[vectorstore]` (`cloud`, `region`).

Lit dans `.env` : `PINECONE_API_KEY`.

L'index est créé automatiquement s'il n'existe pas (avec le model spécifié dans `embedding.model`).

---

## index — Pipeline d'ingestion complet

Raccourci `extract` + `chunk` + `upload` en une seule commande, sans JSON
intermédiaire persistant. Utile pour ingérer un PDF en production.

```bash
python scripts/rag.py index mon.pdf \
    --config configs/baseline.yaml \
    --index mon-index --namespace default
```

Lit toutes les sections : `[extraction]`, `[chunking]`, `[embedding]`, `[vectorstore]`.

---

## retrieve — Question en chunks

Cherche les chunks pertinents pour une question (recherche vectorielle +
reranking), sans appeler le LLM. Utile pour évaluer la qualité du retrieval
indépendamment de la génération.

```bash
python scripts/rag.py retrieve "ma question" \
    --config configs/baseline.yaml \
    --index mon-index --namespace default \
    --out hits.json
```

Lit dans baseline.yaml : `[embedding].model`, `[retrieval]` (`rerank_model`, `retrieve_k`, `top_k`, `rerank`, `rerank_threshold`, `truncation_max_tokens`, `truncation_chars_per_token`).

Le JSON produit contient une liste d'`EnrichedChunk` sérialisés : `chunk_id`, `text`, `score`, `rerank_score`, `document_name`, `document_title`, `page_numbers`, `formulas_latex`, `image_ids`, etc.

---

## ask — Question/réponse RAG complète

Pipeline complet de question/réponse : retrieval + reranking + génération LLM.

```bash
python scripts/rag.py ask "Quelle est la définition de X ?" \
    --config configs/baseline.yaml \
    --index mon-index --namespace default
```

Lit toutes les sections : `[embedding]`, `[retrieval]`, `[generation]`.

Lit dans `.env` : `PINECONE_API_KEY`, `HF_TOKEN`.

Affiche : la réponse formatée, les sources citées extraites de la réponse, le nombre de chunks utilisés.

---

## Workflow type pour tester un PDF

```bash
# 1. Extraction seule (rapide, pas de réseau)
python scripts/rag.py extract papers/transformers.pdf \
    --config configs/baseline.yaml --out tmp/transformers.json

# 2. Inspecter le JSON pour vérifier que l'extraction est correcte
#    (texte propre, formules en LaTeX, images détectées, etc.)

# 3. Chunker à partir du JSON
python scripts/rag.py chunk tmp/transformers.json \
    --config configs/baseline.yaml --out tmp/transformers_chunks.json

# 4. Vérifier les chunks (taille, métadonnées préservées)

# 5. Uploader
python scripts/rag.py upload tmp/transformers_chunks.json \
    --config configs/baseline.yaml --index test-papers --namespace default

# 6. Tester la recherche sur la même question avec différents configs
python scripts/rag.py retrieve "qu'est-ce que l'attention multi-tête ?" \
    --config configs/baseline.yaml --index test-papers --namespace default \
    --out hits_baseline.json

python scripts/rag.py retrieve "qu'est-ce que l'attention multi-tête ?" \
    --config configs/variant_no_rerank.yaml --index test-papers --namespace default \
    --out hits_no_rerank.json

# 7. Comparer les deux JSON pour voir l'impact du rerank
```

---

## Codes de sortie

| Code | Signification |
|------|---------------|
| `0` | succès |
| `2` | config invalide : clé manquante dans YAML ou variable d'environnement absente |
| autre | erreur non gérée (laisse remonter la stacktrace Python) |

---

## Structure du module

```
scripts/
├── rag.py        Point d'entrée unique avec argparse + dispatch des sous-commandes
└── README.md     ce fichier
```

`rag.py` contient les six fonctions `cmd_extract`, `cmd_chunk`, `cmd_upload`,
`cmd_index`, `cmd_retrieve`, `cmd_ask`, plus quatre helpers de validation
(`load_config`, `require_section`, `require_key`, `require_env`). Tout est dans
un seul fichier pour rester self-contained et faciliter `python scripts/rag.py`
sans installation préalable.

---

## Tests CLI

Les tests fumée du CLI sont dans [tests/test_scripts_smoke.py](../tests/test_scripts_smoke.py) :
ils vérifient que chaque sous-commande accepte `--help`, que les imports
fonctionnent, et que `baseline.yaml` permet l'instanciation correcte de
`PDFExtractor`. Le timeout subprocess est configurable via
`tests.smoke_subprocess_timeout_seconds` dans baseline.yaml.

```bash
pytest tests/test_scripts_smoke.py -v
```
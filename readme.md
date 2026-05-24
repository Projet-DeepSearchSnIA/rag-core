# rag-core

`rag-core` est un package Python qui contient le cœur du pipeline RAG utilisé dans le projet NOXA.

Il centralise toutes les opérations liées au traitement de documents : extraction PDF, OCR, nettoyage, chunking, embeddings, retrieval, reranking et préparation du contexte pour les modèles de langage.

## Objectif

L’objectif de rag-core est de séparer complètement la logique RAG de l’application Django noxa afin de rendre le pipeline :

* indépendant du backend web
* réutilisable dans différents contextes (API, notebooks, scripts, recherche)
* plus simple à tester et à faire évoluer
* compatible avec des environnements de recherche et de benchmark

## Installation

Pour installer le package en mode éditable à partir des dépendances déclarées dans `pyproject.toml` :

```bash
pip install -e .
```

Pour reproduire un environnement  depuis `requirements.txt` :

```bash
pip install -r requirements.txt
```

Pour inclure les outils de développement et de test :

```bash
pip install -e ".[dev]"
```

## Configuration

Copier `.env.example` en `.env` et renseigner les variables selon l'environnement :


Les paramètres du pipeline tels que la taille des chunks, le nombre de résultats à récupérer ou le seuil de reranking sont dans `configs/baseline.yaml`.

## Utilisation

Le package s'utilise via les scripts fournis dans `scripts/` ou directement en important ses composants.

Indexer un document PDF dans Pinecone :

```bash
python scripts/index.py chemin/vers/fichier.pdf --index nom-index --namespace ns --config configs/baseline.yaml
```

Interroger le pipeline :

```bash
python scripts/query.py "Quelle est la définition de X ?" --index nom-index --namespace ns --config configs/baseline.yaml
```

Chaque étape du pipeline est accessible indépendamment :

```python
from rag_core import PDFExtractor, SmartTextSplitter, ChunkOptimizer

extractor = PDFExtractor()
doc = extractor.extract("chemin/vers/fichier.pdf")

splitter = SmartTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split(doc)

optimizer = ChunkOptimizer()
optimized = optimizer.optimize(chunks)
```

## Tests

Lancer l'ensemble des tests unitaires :

```bash
pytest tests/ -v
```

Avec rapport de couverture :

```bash
pytest tests/ --cov=rag_core --cov-report=term-missing
```

Les tests marqués `integration` s'appuient sur les services externes configurés dans `.env` :

```bash
pytest tests/ -m integration -v
```

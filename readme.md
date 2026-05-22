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

```bash
pip install -e .
```

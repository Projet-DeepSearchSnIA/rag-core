"""
Tests pour PromptTemplates et le routage des questions par type.

Ces fonctions vivent dans rag_core/generation/prompt_template.py et transforment
les chunks récupérés en prompts LLM. Elles sont pure Python, aucun appel réseau,
donc rapides et faciles à tester.

On utilise des dicts simples pour simuler les chunks — PromptTemplates accepte
à la fois des dicts et des objets avec to_dict(), on teste les deux formes.
"""
from rag_core.generation.prompt_template import PromptTemplates, get_template_for_question_type


def _chunk_dict(text: str, doc_name: str = "article.pdf", pages: list = None) -> dict:
    """Simule un chunk sous forme de dict, comme ce que retourne PineconeRetriever."""
    return {
        "text": text,
        "score": 0.85,
        "metadata": {
            "document_name": doc_name,
            "page_numbers": pages or [1],
        },
    }


def test_format_context_contient_le_texte_du_chunk():
    """Le contexte formaté doit inclure le texte du chunk fourni."""
    chunks = [_chunk_dict("L'attention multi-tête calcule des projections en parallèle.")]
    contexte = PromptTemplates.format_context(chunks)
    assert "attention multi-tête" in contexte


def test_format_context_contient_le_nom_du_document():
    """La source (nom du document) doit apparaître dans le contexte formaté."""
    chunks = [_chunk_dict("Contenu quelconque.", doc_name="transformer_paper.pdf")]
    contexte = PromptTemplates.format_context(chunks)
    assert "transformer_paper.pdf" in contexte


def test_format_context_limite_max_chunks():
    """Quand on passe max_chunks=2, les chunks en surplus sont ignorés."""
    chunks = [_chunk_dict(f"Chunk {i}.") for i in range(5)]
    contexte = PromptTemplates.format_context(chunks, max_chunks=2)
    # on vérifie qu'on n'a que 2 blocs "--- Document" dans le résultat
    assert contexte.count("--- Document") == 2


def test_format_context_pages_liste():
    """Quand page_numbers est une liste, les pages doivent apparaître dans le contexte."""
    chunks = [_chunk_dict("Texte.", pages=[3, 4])]
    contexte = PromptTemplates.format_context(chunks)
    assert "3" in contexte or "4" in contexte


def test_format_context_avec_score():
    """Avec include_scores=True, la pertinence doit apparaître dans le contexte."""
    chunks = [_chunk_dict("Texte avec score.")]
    contexte = PromptTemplates.format_context(chunks, include_scores=True)
    assert "pertinence" in contexte


def test_build_rag_prompt_contient_la_question():
    """Le prompt RAG construit doit contenir la question de l'utilisateur."""
    chunks = [_chunk_dict("Contexte pertinent.")]
    prompt = PromptTemplates.build_rag_prompt("Qu'est-ce que l'attention ?", chunks)
    assert "Qu'est-ce que l'attention ?" in prompt


def test_build_rag_prompt_contient_le_contexte():
    """Le prompt RAG construit doit contenir le texte des chunks."""
    chunks = [_chunk_dict("Le mécanisme d'attention pèse l'importance des tokens.")]
    prompt = PromptTemplates.build_rag_prompt("Question ?", chunks)
    assert "attention" in prompt


def test_build_rag_prompt_template_par_defaut():
    """Sans template fourni, le template RAG_WITH_SOURCES est utilisé."""
    chunks = [_chunk_dict("Texte.")]
    prompt = PromptTemplates.build_rag_prompt("Question ?", chunks)
    # RAG_WITH_SOURCES contient cette phrase caractéristique
    assert "UNIQUEMENT" in prompt


def test_build_chat_messages_roles_corrects():
    """build_chat_messages doit retourner une liste avec au moins 'system' et 'user'."""
    chunks = [_chunk_dict("Du contexte.")]
    messages = PromptTemplates.build_chat_messages("Quelle est la formule ?", chunks)
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles


def test_build_chat_messages_question_dans_user():
    """La question de l'utilisateur doit se retrouver dans le message de rôle 'user'."""
    chunks = [_chunk_dict("Contexte.")]
    question = "Comment fonctionne le reranking ?"
    messages = PromptTemplates.build_chat_messages(question, chunks)
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) == 1
    assert question in user_messages[0]["content"]


def test_build_chat_messages_historique_inclus():
    """L'historique de conversation doit être inséré entre system et user."""
    chunks = [_chunk_dict("Contexte.")]
    historique = [
        {"role": "user", "content": "Première question."},
        {"role": "assistant", "content": "Première réponse."},
    ]
    messages = PromptTemplates.build_chat_messages(
        "Deuxième question.", chunks, conversation_history=historique
    )
    # system → historique → nouvelle question
    assert messages[1]["content"] == "Première question."
    assert messages[2]["content"] == "Première réponse."
    assert messages[-1]["content"] == "Deuxième question."


def test_extraction_sources_pattern_standard():
    """Les sources au format [Source: doc.pdf, page 3] doivent être extraites."""
    reponse = "Selon [Source: article.pdf, page 3], l'attention est clé."
    sources = PromptTemplates.extract_sources_from_response(reponse)
    assert len(sources) == 1
    assert "article.pdf" in sources[0]


def test_extraction_sources_plusieurs():
    """Plusieurs citations dans une même réponse doivent toutes être extraites."""
    reponse = (
        "[Source: doc1.pdf, page 1] et [Source: doc2.pdf, page 7] confirment."
    )
    sources = PromptTemplates.extract_sources_from_response(reponse)
    assert len(sources) == 2


def test_extraction_sources_aucune():
    """Une réponse sans citation retourne une liste vide."""
    reponse = "Voici ma réponse sans aucune source citée."
    sources = PromptTemplates.extract_sources_from_response(reponse)
    assert sources == []


def test_extraction_metadata_blocks_sources_used():
    """extract_metadata_blocks doit parser le bloc SOURCES_USED."""
    reponse = "Réponse principale.\nSOURCES_USED: [doc1.pdf, doc2.pdf]"
    parsed = PromptTemplates.extract_metadata_blocks(reponse)
    assert "doc1.pdf" in parsed["metadata"]["sources_used"]
    assert "doc2.pdf" in parsed["metadata"]["sources_used"]


def test_extraction_metadata_blocks_follow_up():
    """extract_metadata_blocks doit parser le bloc FOLLOW_UP_QUESTIONS."""
    reponse = (
        "Réponse.\n"
        "FOLLOW_UP_QUESTIONS: [Comment calculer l'attention ? ; Qu'est-ce que BERT ?]"
    )
    parsed = PromptTemplates.extract_metadata_blocks(reponse)
    questions = parsed["metadata"]["follow_up_questions"]
    assert len(questions) >= 1


def test_extraction_metadata_blocks_clean_response():
    """La réponse nettoyée ne doit pas contenir les blocs de métadonnées."""
    reponse = "Réponse utile.\nSOURCES_USED: [doc.pdf]"
    parsed = PromptTemplates.extract_metadata_blocks(reponse)
    assert "SOURCES_USED" not in parsed["clean_response"]


def test_template_question_comparaison():
    """Une question avec 'comparer' ou 'différence' → template de comparaison."""
    template = get_template_for_question_type("Comparer BERT et GPT sur les benchmarks.")
    assert "Compare" in template or "Comparaison" in template or "comparison" in template.lower()


def test_template_question_explication():
    """Une question avec 'comment' ou 'expliquer' → template d'explication."""
    template = get_template_for_question_type("Comment fonctionne le mécanisme d'attention ?")
    assert "Expliqu" in template or "xplication" in template


def test_template_question_resume():
    """Une question avec 'résumé' ou 'synthèse' → template de synthèse."""
    template = get_template_for_question_type("Fais un résumé des principaux points.")
    assert "ynthèse" in template or "ésumé" in template


def test_template_question_factuelle():
    """Une question commençant par 'qui', 'quoi', 'combien' → template factuel."""
    template = get_template_for_question_type("Qui a inventé le Transformer ?")
    assert "factuel" in template.lower() or "courte" in template.lower() or "précise" in template.lower()


def test_template_question_par_defaut():
    """Une question sans mot-clé reconnu → retourne RAG_WITH_SOURCES."""
    template = get_template_for_question_type("Donne-moi des informations sur l'attention.")
    # RAG_WITH_SOURCES contient toujours ce marqueur
    assert "UNIQUEMENT" in template


def test_format_response_with_sources_structure():
    """format_response_with_sources doit retourner un dict avec les clés attendues."""
    reponse = "L'attention est clé [Source: doc.pdf, page 2].\nSOURCES_USED: [doc.pdf]"
    chunks = [_chunk_dict("Contexte.", doc_name="doc.pdf")]
    resultat = PromptTemplates.format_response_with_sources(reponse, chunks)
    assert "response" in resultat
    assert "cited_sources" in resultat
    assert "all_sources" in resultat
    assert "num_sources_used" in resultat
    assert resultat["num_sources_used"] == 1
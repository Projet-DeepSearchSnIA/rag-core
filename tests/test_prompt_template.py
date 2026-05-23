"""
Tests pour PromptTemplates et le routage des questions par type.

Ces fonctions vivent dans rag_core/generation/prompt_template.py et transforment
les chunks récupérés en prompts LLM. Elles sont pure Python, aucun appel réseau,
donc rapides et faciles à tester.

On utilise des dicts simples pour simuler les chunks — PromptTemplates accepte
à la fois des dicts et des objets avec to_dict(), on teste les deux formes.
"""
from rag_core.generation.prompt_template import (
    PromptTemplates,
    QUESTION_TYPE_TEMPLATES,
    get_template_for_question_type,
)


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
    """Quand on passe max_chunks=2, les chunks au-delà sont ignorés."""
    chunks = [_chunk_dict(f"Chunk {i}.") for i in range(5)]
    contexte = PromptTemplates.format_context(chunks, max_chunks=2)
    assert "Chunk 0." in contexte
    assert "Chunk 1." in contexte
    assert "Chunk 2." not in contexte


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


# ---------------------------------------------------------------------------
# Groupement par document (nouveau comportement de format_context)
# ---------------------------------------------------------------------------

def test_format_context_groupe_meme_document():
    """Plusieurs chunks du même fichier → un seul en-tête Source."""
    chunks = [
        _chunk_dict("Extrait 1.", doc_name="cours.pdf"),
        _chunk_dict("Extrait 2.", doc_name="cours.pdf"),
        _chunk_dict("Extrait 3.", doc_name="cours.pdf"),
    ]
    contexte = PromptTemplates.format_context(chunks)
    assert contexte.count("--- Source") == 1
    assert "cours.pdf" in contexte
    assert "Extrait 1." in contexte
    assert "Extrait 2." in contexte


def test_format_context_documents_distincts():
    """Chunks de fichiers différents → un en-tête par fichier."""
    chunks = [
        _chunk_dict("Extrait A.", doc_name="doc_a.pdf"),
        _chunk_dict("Extrait B.", doc_name="doc_b.pdf"),
    ]
    contexte = PromptTemplates.format_context(chunks)
    assert contexte.count("--- Source") == 2
    assert "doc_a.pdf" in contexte
    assert "doc_b.pdf" in contexte


def test_format_context_max_chunks_per_doc():
    """max_chunks_per_doc limite le nombre d'extraits affichés par document."""
    chunks = [_chunk_dict(f"Extrait {i}.", doc_name="gros_doc.pdf") for i in range(5)]
    contexte = PromptTemplates.format_context(chunks, max_chunks=5, max_chunks_per_doc=2)
    assert "Extrait 0." in contexte
    assert "Extrait 1." in contexte
    assert "Extrait 2." not in contexte


# ---------------------------------------------------------------------------
# _normalize_pages
# ---------------------------------------------------------------------------

def test_normalize_pages_liste():
    assert PromptTemplates._normalize_pages([1, 2, 3]) == "1, 2, 3"


def test_normalize_pages_csv_string():
    assert PromptTemplates._normalize_pages("1,2,3") == "1,2,3"


def test_normalize_pages_string_avec_crochets():
    assert PromptTemplates._normalize_pages("[1, 2]") == "1, 2"


def test_normalize_pages_entier():
    assert PromptTemplates._normalize_pages(5) == "5"


def test_normalize_pages_none():
    assert PromptTemplates._normalize_pages(None) == "inconnue"


# ---------------------------------------------------------------------------
# Template adaptatif dans build_chat_messages
# ---------------------------------------------------------------------------

def test_build_chat_messages_template_adaptatif_ajoute_style():
    """Un template non-default → la dernière ligne est ajoutée au message user."""
    chunks = [_chunk_dict("Contexte.")]
    template_factuel = QUESTION_TYPE_TEMPLATES["factual"]
    messages = PromptTemplates.build_chat_messages(
        "Qui a inventé le Transformer ?", chunks, template=template_factuel
    )
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "Qui a inventé le Transformer ?" in user_content
    style = [l.strip() for l in template_factuel.strip().split("\n") if l.strip()][-1]
    assert style in user_content


def test_build_chat_messages_rag_with_sources_question_inchangee():
    """Avec RAG_WITH_SOURCES (défaut), la question n'est pas modifiée."""
    chunks = [_chunk_dict("Contexte.")]
    question = "Quelle est la formule de l'attention ?"
    messages = PromptTemplates.build_chat_messages(
        question, chunks, template=PromptTemplates.RAG_WITH_SOURCES
    )
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert user_content == question
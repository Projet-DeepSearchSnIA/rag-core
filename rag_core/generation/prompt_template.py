from typing import Any, List, Dict, Optional
from collections import defaultdict
import re


class PromptTemplates:
    """templates de prompts pour différents cas d'usage RAG"""

    RAG_WITH_SOURCES = """Tu es un assistant expert qui répond aux questions en te basant UNIQUEMENT sur les documents fournis.

RÈGLES IMPORTANTES:
1. Réponds UNIQUEMENT avec les informations présentes dans les documents ci-dessous
2. Si l'information n'est PAS dans les documents, dis clairement "Je ne trouve pas cette information dans les documents fournis"
3. CITE TOUJOURS tes sources en utilisant le format [Source: nom_du_document, page X]
4. Sois précis et factuel
5. Si plusieurs documents disent des choses différentes, mentionne les deux points de vue

DOCUMENTS DE RÉFÉRENCE:
{context}

QUESTION DE L'UTILISATEUR:
{question}

RÉPONSE (avec citations):"""

    MULTI_DOC_SYNTHESIS = """Tu es un assistant expert en synthèse documentaire.

Ta tâche est de synthétiser les informations des documents suivants pour répondre à la question.

DOCUMENTS:
{context}

QUESTION:
{question}

Fournis une réponse complète qui:
1. Synthétise les informations de TOUS les documents pertinents
2. Cite chaque source utilisée avec [Source: document, page X]
3. Signale les contradictions éventuelles entre documents
4. Reste factuel et objective

RÉPONSE:"""

    NO_CONTEXT = """Tu es un assistant utile.

L'utilisateur pose la question suivante, mais aucun document pertinent n'a été trouvé dans la base de connaissances.

QUESTION:
{question}

Réponds de manière générale en précisant que tu n'as pas de documents spécifiques sur ce sujet dans la base de connaissances.

RÉPONSE:"""

    RELEVANCE_CHECK = """Les documents suivants sont-ils pertinents pour répondre à la question?

QUESTION: {question}

DOCUMENTS:
{context}

Réponds uniquement par OUI ou NON."""

    @staticmethod
    def _normalize_pages(pages) -> str:
        """Convertit page_numbers (list, CSV string, int) en string lisible."""
        if isinstance(pages, list):
            return ', '.join(map(str, pages))
        if isinstance(pages, str):
            cleaned = pages.strip().strip('[]')
            return cleaned if cleaned else 'inconnue'
        if pages is not None:
            return str(pages)
        return 'inconnue'

    @staticmethod
    def _extract_chunk_fields(chunk) -> tuple:
        """Retourne (metadata, text, score) depuis un chunk dict ou EnrichedChunk."""
        if hasattr(chunk, "to_dict"):
            chunk_dict = chunk.to_dict()
            metadata = getattr(chunk, "metadata", {}) or {}
            text = chunk_dict.get("text", "")
            score_val = chunk_dict.get("score")
        else:
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', chunk.get('content', ''))
            score_val = chunk.get('score')
        return metadata, text, score_val

    @staticmethod
    def format_context(
        retrieved_chunks: List[Dict],
        include_scores: bool = False,
        max_chunks: int = 5,
        max_chunks_per_doc: int = 3,
    ) -> str:
        """Formate les chunks en groupant par document source.

        Chaque document est présenté une seule fois avec ses extraits listés
        dessous, évitant que le LLM confonde plusieurs chunks du même fichier
        avec des documents distincts.
        """
        # Grouper les chunks par document (en respectant max_chunks global)
        docs: dict = defaultdict(list)
        for chunk in retrieved_chunks[:max_chunks]:
            metadata, text, score_val = PromptTemplates._extract_chunk_fields(chunk)
            doc_name = (
                metadata.get('document_path')
                or metadata.get('document_name')
                or 'Document inconnu'
            )
            docs[doc_name].append((metadata, text, score_val))

        context_parts = []
        for doc_num, (doc_name, chunks) in enumerate(docs.items(), 1):
            context_parts.append(f"--- Source {doc_num} : {doc_name} ---")

            for chunk_data in chunks[:max_chunks_per_doc]:
                metadata, text, score_val = chunk_data
                pages_str = PromptTemplates._normalize_pages(metadata.get('page_numbers'))
                score_str = f" (pertinence : {score_val:.2f})" if include_scores and score_val is not None else ""

                images = metadata.get('image_paths') or metadata.get('image_ids') or []
                formulas = metadata.get('formulas_latex') or []

                images_str = ""
                formulas_str = ""
                if images:
                    imgs = images if isinstance(images, list) else [images]
                    images_str = "\nImages : " + " | ".join(str(img) for img in imgs[:5])
                if formulas:
                    fmls = formulas if isinstance(formulas, list) else [formulas]
                    formulas_str = "\nFormules : " + " | ".join(str(f) for f in fmls[:3])

                context_parts.append(
                    f"[page(s) {pages_str}{score_str}]\n"
                    f"{text}"
                    f"{images_str}{formulas_str}\n"
                )

        return "\n".join(context_parts)

    @staticmethod
    def build_rag_prompt(question: str, retrieved_chunks: List[Dict], template: Optional[str] = None, **kwargs) -> str:
        if template is None:
            template = PromptTemplates.RAG_WITH_SOURCES

        context = PromptTemplates.format_context(
            retrieved_chunks,
            include_scores=kwargs.get('include_scores', False),
            max_chunks=kwargs.get('max_chunks', 5)
        )

        return template.format(context=context, question=question, **kwargs)

    @staticmethod
    def build_chat_messages(
        question: str,
        retrieved_chunks: List[Dict],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        topic: Optional[str] = None,
        template: Optional[str] = None
    ) -> List[Dict]:
        messages = []

        if system_prompt is None:
            context = PromptTemplates.format_context(retrieved_chunks)
            context_noxa = "Tu es Noxa AI, l'IA intégrée dans Noxa. Noxa est une application permettant d'aider les étudiants et chercheurs dans la rédaction de leurs documents académiques et scientifiques. Tu dois toujours répondre dans la langue de l'utilisateur."
            topic_line = f"Tu es spécialisé en {topic}." if topic else "Tu es un assistant expert."
            system_prompt = f"""{topic_line} {context_noxa} Tu réponds en te basant UNIQUEMENT sur les documents fournis.

FORMAT DE RÉPONSE OBLIGATOIRE:
1) Réponse structurée et concise en texte.
2) NE PAS écrire de phrase d'introduction sur tes sources. Les citations sont obligatoires dans le texte via [Source: ...].
3) Toute équation doit être entourée par $$ ... $$ (LaTeX).
4) À la fin de ta réponse, ajoute OBLIGATOIREMENT ces blocs pour le système :
   SOURCES_USED: [document1.pdf, document2.pdf]
   IMAGES_USED: [id_image1, id_image2]
   FOLLOW_UP_QUESTIONS: [Question 1; Question 2; Question 3]
5) Cite toujours tes sources dans le texte: [Source: document_path, page X]
6) Si l'info n'est pas dans les documents, dis-le clairement.
7) Propose 3 à 5 questions courtes et pertinentes en FOLLOW_UP_QUESTIONS.

DOCUMENTS DE RÉFÉRENCE:
{context}
"""

        messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            messages.extend(conversation_history)

        # Appliquer le style de réponse issu du template (ex. "Réponse courte et précise avec source:")
        user_content = question
        if template is not None and template != PromptTemplates.RAG_WITH_SOURCES:
            style_lines = [line.strip() for line in template.strip().split('\n') if line.strip()]
            if style_lines:
                user_content = f"{question}\n\n{style_lines[-1]}"

        messages.append({"role": "user", "content": user_content})

        return messages

    @staticmethod
    def extract_sources_from_response(response: str) -> List[str]:
        pattern = r'\[Source:\s*([^\]]+)\]'
        return list(set(re.findall(pattern, response)))

    @staticmethod
    def extract_metadata_blocks(response: str) -> Dict:
        metadata = {
            'sources_used': [],
            'images_used': [],
            'equations_used': [],
            'follow_up_questions': []
        }

        clean_response = response

        for key in metadata.keys():
            tag_name = key.upper()
            pattern = rf'{tag_name}\s*:\s*(?:\[(.*?)\]|(.*?))(?=\s*(?:SOURCES_USED|IMAGES_USED|EQUATIONS_USED|FOLLOW_UP_QUESTIONS)|$)'
            match = re.search(pattern, clean_response, re.DOTALL | re.IGNORECASE)
            if match:
                full_match = match.group(0)
                content = (match.group(1) or match.group(2) or "").strip()
                items = [item.strip() for item in re.split(r'[;,\n]', content) if item.strip()]
                items = [item.strip('[] ') for item in items]
                metadata[key] = [i for i in items if i]
                clean_response = clean_response.replace(full_match, '')

        clean_response = re.sub(r'\n{3,}', '\n\n', clean_response).strip()

        return {'clean_response': clean_response, 'metadata': metadata}

    @staticmethod
    def format_response_with_sources(response: str, retrieved_chunks: List[Any]) -> Dict:
        parsed_result = PromptTemplates.extract_metadata_blocks(response)
        clean_response = parsed_result['clean_response']
        metadata = parsed_result['metadata']

        cited_sources = PromptTemplates.extract_sources_from_response(clean_response)
        if metadata['sources_used']:
            cited_sources.extend(metadata['sources_used'])
        cited_sources = list(set(cited_sources))

        sources_info = []
        for chunk in retrieved_chunks:
            if hasattr(chunk, "to_dict"):
                chunk_dict = chunk.to_dict()
                metadata_chunk = getattr(chunk, "metadata", {}) or {}
                chunk_id = chunk_dict.get("chunk_id", "")
                score_val = chunk_dict.get("score", 0.0)
            else:
                chunk_dict = chunk
                metadata_chunk = chunk.get('metadata', {})
                chunk_id = chunk.get('id', '')
                score_val = chunk.get('score', 0.0)

            source_info = {
                'document': metadata_chunk.get('document_name', ''),
                'pages': metadata_chunk.get('page_numbers', ''),
                'chunk_id': chunk_id,
                'score': score_val
            }
            if source_info not in sources_info:
                sources_info.append(source_info)

        return {
            'response': clean_response,
            'cited_sources': cited_sources,
            'all_sources': sources_info,
            'num_sources_used': len(retrieved_chunks),
            'extracted_metadata': metadata
        }


QUESTION_TYPE_TEMPLATES = {
    'factual': """Réponds à cette question factuelle en te basant sur les documents:\n\nDOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nRéponse courte et précise avec source:""",
    'explanation': """Explique le concept suivant en te basant sur les documents:\n\nDOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nExplication détaillée avec exemples et sources:""",
    'comparison': """Compare les éléments mentionnés en te basant sur les documents:\n\nDOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nComparaison structurée avec sources pour chaque point:""",
    'summary': """Fais une synthèse en te basant sur les documents:\n\nDOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nSynthèse concise avec points clés et sources:"""
}


def get_template_for_question_type(question: str) -> str:
    question_lower = question.lower()

    if any(word in question_lower for word in ['comparer', 'différence', 'versus', 'vs']):
        return QUESTION_TYPE_TEMPLATES['comparison']
    elif any(word in question_lower for word in ['expliquer', 'comment', 'pourquoi', "c'est quoi"]):
        return QUESTION_TYPE_TEMPLATES['explanation']
    elif any(word in question_lower for word in ['résumer', 'synthèse', 'résumé', 'principaux points']):
        return QUESTION_TYPE_TEMPLATES['summary']
    elif any(word in question_lower for word in ['qui', 'quoi', 'où', 'quand', 'combien']):
        return QUESTION_TYPE_TEMPLATES['factual']
    else:
        return PromptTemplates.RAG_WITH_SOURCES

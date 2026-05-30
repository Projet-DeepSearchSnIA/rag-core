import time
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient

from rag_core.generation.prompt_template import PromptTemplates, get_template_for_question_type
from rag_core.utils.logger import get_logger

logger = get_logger(__name__)


class LLMHandler:
    """gestionnaire LLM pour génération de réponses RAG"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        retry_delay_seconds: int,
        provider: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("api_key requis (HF_TOKEN dans .env, à passer explicitement)")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        logger.info("initialisation LLMHandler — modèle: %s", model_name)
        self.client = InferenceClient(api_key=api_key)

    def generate_response(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        use_adaptive_template: bool = True,
        include_sources: bool = True,
        conversation_history: Optional[List[Dict]] = None,
        topic: Optional[str] = None
    ) -> Dict:
        logger.info("génération — question: %s..., %d chunks", question[:80], len(retrieved_chunks))

        template = get_template_for_question_type(question) if use_adaptive_template else PromptTemplates.RAG_WITH_SOURCES

        messages = PromptTemplates.build_chat_messages(
            question=question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            topic=topic,
            template=template
        )

        max_retries = self.max_retries
        retry_delay = self.retry_delay_seconds
        response_text = ""
        t0 = time.time()

        for attempt in range(max_retries):
            try:
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                if self.provider:
                    params["provider"] = self.provider

                completion = self.client.chat.completions.create(**params)
                response_text = completion.choices[0].message.content
                logger.info("réponse générée en %.2fs (%d chars)", time.time() - t0, len(response_text))
                break

            except Exception as e:
                error_str = str(e)
                logger.warning("erreur génération tentative %d: %s", attempt + 1, error_str)
                is_server_error = any(code in error_str for code in ["502", "503", "504", "Bad Gateway", "Gateway Time-out"])
                if is_server_error and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("génération échouée après %.2fs (%d tentative(s))", time.time() - t0, attempt + 1)
                    response_text = "Désolé, je n'ai pas pu générer de réponse. Erreur technique (HuggingFace)."
                    break

        if include_sources:
            return PromptTemplates.format_response_with_sources(response_text, retrieved_chunks)

        return {
            'response': response_text,
            'cited_sources': [],
            'all_sources': [],
            'num_sources_used': len(retrieved_chunks)
        }

    def generate_simple(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens
        }
        if self.provider:
            params["provider"] = self.provider

        try:
            completion = self.client.chat.completions.create(**params)
            return completion.choices[0].message.content or ""
        except Exception as e:
            logger.error("erreur generate_simple: %s", e)
            return "Désolé, je n'ai pas pu générer de réponse."

    def stream_response(self, question: str, retrieved_chunks: List[Dict], **kwargs):
        messages = PromptTemplates.build_chat_messages(question=question, retrieved_chunks=retrieved_chunks)

        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            result = self.generate_response(question, retrieved_chunks, **kwargs)
            yield result['response']


class RAGPipeline:
    """pipeline RAG complet : retriever + LLM.

    Une seule façon d'initialiser : RAGPipeline(retriever=..., llm=...).
    Le retriever doit exposer une méthode .retrieve(query, top_k) qui retourne
    des objets convertibles via .to_dict().
    """

    def __init__(self, *, retriever, llm: LLMHandler):
        if retriever is None:
            raise ValueError("retriever requis")
        if llm is None:
            raise ValueError("llm requis")
        self.retriever = retriever
        self.llm = llm
        logger.info("RAGPipeline initialisé")

    def ask(self, question: str, top_k: int = 5, **kwargs) -> Dict:
        logger.info("question RAG: %s", question)

        chunks = self.retriever.retrieve(query=question, top_k=top_k)
        retrieved_chunks = [c.to_dict() for c in chunks]

        if not retrieved_chunks:
            return {
                'response': "Je n'ai pas trouvé de documents pertinents pour répondre à cette question dans ma base de connaissances.",
                'cited_sources': [], 'all_sources': [], 'num_sources_used': 0,
            }

        return self.llm.generate_response(question=question, retrieved_chunks=retrieved_chunks, **kwargs)

    def batch_ask(self, questions: List[str], **kwargs) -> List[Dict]:
        return [self.ask(question, **kwargs) for question in questions]

from dataclasses import dataclass

from ai.config import settings


@dataclass(frozen=True)
class Context:
    max_retries: int = settings.max_retries
    kb_top_k: int = settings.kb_top_k
    web_max_results: int = settings.web_max_results
    min_relevant_docs: int = settings.min_relevant_docs
    use_reranker: bool = settings.use_reranker
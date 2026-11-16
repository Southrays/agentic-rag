from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# 1) Push .env into os.environ so the SDKs (OpenAI, Pinecone, Tavily, LangSmith)
#    can read their keys directly.
load_dotenv(ROOT / ".env")


# 2) Typed, validated settings. Fails fast if a required key is missing.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str
    pinecone_api_key: str
    tavily_api_key: str

    pinecone_index_name: str = "agentic-rag"
    pinecone_namespace: str = "kb-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    llm_model: str = "openai:gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    kb_description: str = "General company documents."

    max_retries: int = 2
    kb_top_k: int = 5
    web_max_results: int = 5
    min_relevant_docs: int = 1
    use_reranker: bool = False
    rerank_model: str = "bge-reranker-v2-m3"


settings = Settings()  # pyright: ignore[reportCallIssue]
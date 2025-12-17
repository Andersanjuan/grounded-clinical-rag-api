from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Vector DB
    chroma_dir: str = "chroma_db"
    chroma_collection: str = "medrag_docs"

    # Retrieval / grounding
    max_distance: float = 1.2
    require_citations: bool = True

    # LLM
    ollama_model: str = "llama3.1"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"

    # API security
    api_key: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()

from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DEEPSEEK_API_KEY: str
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/fraud_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str = "http://localhost:6333"

    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    MAX_ITERATIONS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.7
    TIMEOUT_SECONDS: int = 30

    PII_MASKING_ENABLED: bool = True
    AUDIT_LOGGING_ENABLED: bool = True
    JWT_SECRET: str = "change-this-in-production"

    RAGAS_ENABLED: bool = True
    METRICS_PORTION_SIZE: int = 100

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    ENABLE_STREAMING: bool = True
    ENABLE_CACHING: bool = True
    ENABLE_METRICS: bool = True

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    # Map environment variables here
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_names": {
            "OPENAI_API_KEY": "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
            "DATABASE_URL": "DATABASE_URL",
            "REDIS_URL": "REDIS_URL",
            "QDRANT_URL": "QDRANT_URL",
            "JWT_SECRET": "JWT_SECRET",
        },
    }


# global instance
settings = Settings()

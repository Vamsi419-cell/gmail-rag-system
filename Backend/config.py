"""
config.py — Centralised settings loaded from environment / .env
All tuneable knobs live here so nothing is hard-coded across the app.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Paths (all relative to repo root so they work from any CWD)         #
    # ------------------------------------------------------------------ #
    BASE_DIR: Path = Path(__file__).parent.parent
    DB_PATH: Path = Path(__file__).parent.parent / "notebook" / "rag_database.db"
    CREDENTIALS_PATH: Path = Path(__file__).parent.parent / "notebook" / "credentials.json"
    TOKEN_PATH: Path = Path(__file__).parent.parent / "notebook" / "token.json"
    FAISS_INDEX_PATH: Path = Path(__file__).parent.parent / "data" / "faiss.index"
    METADATA_PATH: Path = Path(__file__).parent.parent / "data" / "metadata.pkl"
    SYNC_FILE: Path = Path(__file__).parent.parent / "notebook" / "last_sync.json"

    # ------------------------------------------------------------------ #
    # Gmail OAuth                                                          #
    # ------------------------------------------------------------------ #
    GMAIL_SCOPES: List[str] = ["https://www.googleapis.com/auth/gmail.readonly"]
    OAUTH_REDIRECT_PORT: int = 8080          # local port for OAuth callback
    INITIAL_FETCH_LIMIT: int = 100

    # ------------------------------------------------------------------ #
    # Embedding + retrieval                                                #
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 2000

    # ------------------------------------------------------------------ #
    # Generator  (swap GENERATOR_MODEL to e.g. "google/flan-t5-large"    #
    # or set USE_OPENAI_COMPAT=true and point OPENAI_BASE_URL at any     #
    # OpenAI-compatible endpoint — LM Studio, vLLM, Groq, etc.)         #
    # ------------------------------------------------------------------ #
    GENERATOR_MODEL: str = "google/flan-t5-small"
    MAX_NEW_TOKENS: int = 300
    REPETITION_PENALTY: float = 1.2
    TEMPERATURE: float = 0.7

    # Optional: drop-in replacement via OpenAI-compatible API
    USE_OPENAI_COMPAT: bool = False
    OPENAI_BASE_URL: str = "http://localhost:1234/v1"   # e.g. LM Studio
    OPENAI_API_KEY: str = "lm-studio"
    OPENAI_MODEL: str = "local-model"

    # ------------------------------------------------------------------ #
    # Server                                                               #
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:8501"]   # Streamlit default


settings = Settings()

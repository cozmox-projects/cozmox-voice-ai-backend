"""
config.py — Central configuration loaded from .env
All services import settings from here.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Deepgram ──────────────────────────────────────────────────────────
    deepgram_api_key: str = Field(..., env="DEEPGRAM_API_KEY")

    # ── Azure OpenAI ──────────────────────────────────────────────────────
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(
        default="gpt-4o", env="AZURE_OPENAI_DEPLOYMENT"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01", env="AZURE_OPENAI_API_VERSION"
    )

    # ── ElevenLabs ────────────────────────────────────────────────────────
    elevenlabs_api_key: str = Field(..., env="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", env="ELEVENLABS_VOICE_ID"
    )

    # ── Twilio ────────────────────────────────────────────────────────────
    twilio_account_sid: str = Field(default="", env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(default="", env="TWILIO_PHONE_NUMBER")

    # ── LiveKit ───────────────────────────────────────────────────────────
    livekit_url: str = Field(default="ws://localhost:7880", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="devkey", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="devsecret", env="LIVEKIT_API_SECRET")

    # ── App ───────────────────────────────────────────────────────────────
    max_concurrent_calls: int = Field(default=100, env="MAX_CONCURRENT_CALLS")
    worker_pool_size: int = Field(default=100, env="WORKER_POOL_SIZE")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    webhook_port: int = Field(default=8000, env="WEBHOOK_PORT")
    # Public HTTPS/WSS URL of this webhook service (needed for TwiML <Stream>)
    # e.g. "https://your-ec2-ip-or-domain" — Twilio must reach this over the internet
    webhook_public_url: str = Field(
        default="http://localhost:8000", env="WEBHOOK_PUBLIC_URL"
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chromadb", env="CHROMA_PERSIST_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Returns cached settings instance. Call this everywhere instead of Settings()."""
    return Settings()

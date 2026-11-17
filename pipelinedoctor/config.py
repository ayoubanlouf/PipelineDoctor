"""
pipelinedoctor/config.py
========================
Configuration management using pydantic-settings.
"""

from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Settings ──────────────────────────────────────────────────────
    BIONIC_BASE_URL: str = Field(
        default="http://localhost:7800/v1",
        description="Base URL for local OpenAI-compatible endpoint (Bionic/llama.cpp)",
    )
    BIONIC_MODEL: str = Field(
        default="Qwen3.5 9B",
        description="Model name/id to query",
    )
    LLM_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Timeout in seconds for LLM completions",
    )
    LLM_MAX_TOKENS: int = Field(
        default=2000,
        description="Max tokens budget (accounting for reasoning tokens)",
    )

    # ── GitHub Settings ───────────────────────────────────────────────────
    GITHUB_TOKEN: str = Field(
        default="",
        description="GitHub Personal Access Token or App Installation Token",
    )
    GITHUB_WEBHOOK_SECRET: str = Field(
        default="",
        description="Secret used to sign GitHub webhook payloads (HMAC SHA-256)",
    )

    # ── Agent State Machine ───────────────────────────────────────────────
    MAX_ITERATIONS: int = Field(
        default=3,
        description="Maximum repair iterations before escalating failure",
    )

    # ── Sandbox Settings ──────────────────────────────────────────────────
    DOCKER_IMAGE: str = Field(
        default="python:3.11-slim",
        description="Default Docker image for test verification sandbox",
    )
    SANDBOX_TIMEOUT_SECONDS: int = Field(
        default=180,
        description="Timeout in seconds for container execution",
    )
    SANDBOX_WORK_DIR: str = Field(
        default="/workspace",
        description="Working directory inside the ephemeral container",
    )

    # ── Control Plane API ─────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0", description="API bind host")
    API_PORT: int = Field(default=8000, description="API port")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

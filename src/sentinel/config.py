"""Central configuration. Everything defaults to the free, offline simulation
backend so the platform runs end-to-end with no credentials and no spend."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    # LLM backend: "simulation" (offline, deterministic, free) or "anthropic" (live agents)
    llm_backend: str = "simulation"
    model: str = "claude-opus-4-8"
    max_tokens_per_call: int = 8000
    max_agent_iterations: int = 12
    run_token_budget: int = 400_000

    # Governance
    auto_approve_limit_usd: float = 25_000.0

    # Paths
    data_dir: Path = REPO_ROOT / "data"
    state_dir: Path = REPO_ROOT / ".sentinel"

    # Service
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "sentinel.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()

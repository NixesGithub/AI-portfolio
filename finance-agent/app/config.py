"""Configuración del servicio.

Todo se lee del entorno (12-factor): la misma imagen tiene que servir para
local, staging y producción sin reconstruirse.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Servicio -----------------------------------------------------------
    app_name: str = "finance-agent"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # --- Modelo -------------------------------------------------------------
    anthropic_api_key: str = ""
    model: str = "claude-opus-5"
    max_tokens: int = 4096
    # Techo de tokens del historial que se le manda al modelo. Por encima de
    # esto se recortan los turnos más viejos (ver app/agent/history.py).
    history_max_tokens: int = 12_000
    # Cortafuegos del bucle: sin esto, un modelo que se obstina en llamar a una
    # tool que siempre falla gasta plata hasta que alguien lo mata a mano.
    max_iterations: int = 6
    llm_timeout_seconds: float = 60.0

    # --- Persistencia -------------------------------------------------------
    # SQLite alcanza para un MVP de un solo proceso. El día que haya varias
    # réplicas hay que cambiar la implementación de ConversationStore, no el
    # resto del código: por eso el store es una interfaz.
    database_url: str = "sqlite:///./data/conversations.db"

    # --- Tools --------------------------------------------------------------
    tool_timeout_seconds: float = 20.0
    # Yahoo no publica límites de rate; la caché evita repreguntar lo mismo
    # dentro de una misma conversación.
    quote_cache_ttl_seconds: float = 30.0
    reference_cache_ttl_seconds: float = 900.0
    max_history_rows: int = 60

    # --- Límites de entrada -------------------------------------------------
    max_message_chars: int = 4_000

    @property
    def sqlite_path(self) -> str:
        """Ruta de archivo a partir de database_url."""
        url = self.database_url
        if not url.startswith("sqlite:///"):
            raise ValueError(
                f"Sólo se soporta SQLite en este MVP, no {url!r}. "
                "Para Postgres, implementá ConversationStore contra asyncpg."
            )
        return url.removeprefix("sqlite:///")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings cacheados: se leen una vez por proceso."""
    return Settings()

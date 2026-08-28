"""Configuración de la aplicación.

Todo se lee de variables de entorno (12-factor): la misma imagen de Docker sirve
para local, staging y producción, y lo único que cambia es el entorno.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Servicio -----------------------------------------------------------
    app_name: str = "agente-financiero"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Si se define, /chat exige la cabecera X-API-Key. Vacío = API abierta
    # (cómodo en local, nunca en producción).
    api_key: str = ""

    # --- Modelo -------------------------------------------------------------
    # "anthropic" usa la API real; "fake" es un modelo determinista que permite
    # levantar la API y correr los tests sin credenciales ni red.
    llm_provider: Literal["anthropic", "fake"] = "anthropic"
    llm_model: str = "claude-sonnet-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    anthropic_api_key: str = ""

    # --- Agente -------------------------------------------------------------
    # Vueltas máximas del bucle (una vuelta = una llamada al modelo + sus tools).
    agent_max_iterations: int = 6
    # Presupuesto total de una petición a POST /chat.
    agent_timeout_s: float = 90.0
    tool_timeout_s: float = 15.0
    # Mensajes de historia que se le pasan al modelo en cada turno.
    history_window: int = 40

    # --- Memoria ------------------------------------------------------------
    store_backend: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "/data/conversations.db"
    # Tope de mensajes guardados por conversación (0 = sin tope).
    max_stored_messages: int = 400

    # --- Yahoo Finance ------------------------------------------------------
    yf_cache_ttl_quote_s: int = 60
    yf_cache_ttl_history_s: int = 300
    yf_cache_ttl_profile_s: int = 86_400
    yf_max_history_points: int = 30

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings cacheadas: se leen una vez por proceso."""
    return Settings()

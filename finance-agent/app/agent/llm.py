"""Construcción del modelo."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from ..config import Settings


class MissingCredentialsError(RuntimeError):
    """Falta ANTHROPIC_API_KEY."""


def build_llm(settings: Settings) -> ChatAnthropic:
    if not settings.anthropic_api_key:
        raise MissingCredentialsError(
            "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y completala, "
            "o pasala como variable de entorno al contenedor."
        )

    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
        timeout=settings.llm_timeout_seconds,
        # El SDK ya reintenta 429 y 5xx con backoff exponencial: no hace falta
        # envolverlo en otro reintento nuestro.
        max_retries=2,
    )

"""Dependencias de FastAPI: servicio y autenticación."""

from __future__ import annotations

import hmac

from fastapi import Header, Request

from app.agent.service import ChatService
from app.config import Settings
from app.errors import Unauthorized


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> None:
    """Autenticación por clave, opcional.

    Con ``API_KEY`` vacío la API queda abierta (cómodo en local). En cuanto se
    define, todos los endpoints de negocio la exigen. La comparación es en tiempo
    constante para no filtrar la clave por el tiempo de respuesta.
    """
    settings: Settings = request.app.state.settings
    if not settings.auth_enabled:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise Unauthorized("Falta la cabecera X-API-Key o no es válida.")

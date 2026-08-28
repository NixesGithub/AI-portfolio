"""Dependencias de FastAPI.

El servicio se construye una sola vez en el arranque (ver `app.main`) y se
guarda en `app.state`. Estas funciones sólo lo sacan de ahí: así los tests
pueden sobreescribir la dependencia con un servicio de prueba.
"""

from __future__ import annotations

from fastapi import Request

from ..agent.service import ChatService


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service

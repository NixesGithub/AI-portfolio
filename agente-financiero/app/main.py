"""Punto de entrada de la API.

Las dependencias caras (modelo, tools, almacén) se construyen una vez al arrancar
y viven en ``app.state``: crear un cliente HTTP por petición es una de las formas
más caras y silenciosas de perder latencia.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.models import build_llm
from app.agent.service import ChatService
from app.agent.tools import build_tools
from app.api.routes import router
from app.config import Settings, get_settings
from app.errors import register_exception_handlers
from app.logging import configure_logging, conversation_id_var, request_id_var
from app.memory import ConversationLocks, build_store

logger = logging.getLogger(__name__)

DESCRIPTION = """
Agente conversacional con memoria por conversación y acceso a datos de mercado
de Yahoo Finance.

* `POST /chat` — enviar un mensaje a un hilo (crea el hilo si no se pasa id).
* `GET /chat/{id}` — historial completo de un hilo.
* `DELETE /chat/{id}` — borrar un hilo.

El bucle del agente está definido explícitamente en `app/agent/loop.py`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Se construyen una sola vez: el cliente del modelo mantiene su pool de
        # conexiones y el almacén su esquema.
        store = build_store(settings)
        tools = build_tools(settings)
        llm = build_llm(settings)
        app.state.store = store
        app.state.chat_service = ChatService(
            settings=settings,
            store=store,
            llm=llm,
            tools=tools,
            locks=ConversationLocks(),
        )
        logger.info(
            "servicio arrancado",
            extra={
                "environment": settings.environment,
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "store_backend": settings.store_backend,
                "tools": [tool.name for tool in tools],
                "auth": settings.auth_enabled,
            },
        )
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(
        title="Agente financiero conversacional",
        description=DESCRIPTION,
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        """Id de correlación + log de acceso.

        El cliente puede traer su propio ``X-Request-ID`` (útil cuando hay un
        gateway delante); si no, se genera uno y se devuelve siempre.
        """
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        conversation_token = conversation_id_var.set("-")
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "petición fallida",
                extra={"path": request.url.path, "method": request.method},
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "Error interno.",
                        "request_id": request_id,
                    }
                },
            )
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request_id
        if request.url.path not in ("/health/live", "/health/ready"):
            logger.info(
                "petición",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
        request_id_var.reset(token)
        conversation_id_var.reset(conversation_token)
        return response

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()

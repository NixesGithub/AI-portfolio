"""Logging estructurado.

En producción los logs se leen con una máquina, no con los ojos: por eso el
formato por defecto es JSON de una línea por evento. `request_id` viaja en un
ContextVar para que no haya que pasarlo por parámetro hasta el último rincón
del código.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "conversation_id", default="-"
)

# Atributos que LogRecord trae de fábrica: todo lo demás que aparezca en
# __dict__ es un `extra` que puso quien loguea, y va al JSON.
_RESERVED = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName"}


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.conversation_id = conversation_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "conversation_id": getattr(record, "conversation_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s [%(request_id)s] %(name)s: %(message)s"
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn trae sus propios handlers: los sacamos para que sus mensajes
    # salgan con nuestro formato en vez de duplicarse en dos.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # El access log de uvicorn es un subconjunto del nuestro (que además lleva
    # request_id y duración), así que sólo dejamos pasar sus errores.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # yfinance es ruidoso (avisos de red por cada símbolo que no resuelve).
    logging.getLogger("yfinance").setLevel(logging.ERROR)

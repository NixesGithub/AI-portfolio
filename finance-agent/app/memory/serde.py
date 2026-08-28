"""Serialización de mensajes de LangChain.

Guardar sólo `{"role": ..., "content": ...}` parece suficiente hasta que el
historial tiene una llamada a tool: si se pierden los `tool_calls` de un
AIMessage o el `tool_call_id` de un ToolMessage, la conversación deja de ser
válida para la API del modelo y el turno siguiente falla con un 400.

`messages_to_dict` / `messages_from_dict` son la representación canónica de
LangChain y preservan todo eso, así que persistimos exactamente eso.
"""

from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


def dumps(message: BaseMessage) -> str:
    return json.dumps(messages_to_dict([message])[0], ensure_ascii=False)


def loads(payload: str) -> BaseMessage:
    return messages_from_dict([json.loads(payload)])[0]

"""Tests de la ventana de contexto."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.history import build_prompt_messages, trim_history
from app.memory.base import StoredMessage, utcnow


def stored(message) -> StoredMessage:
    return StoredMessage(
        id="msg_1", conversation_id="c", created_at=utcnow(), message=message
    )


def test_no_recorta_si_cabe():
    messages = [HumanMessage(content=str(i)) for i in range(5)]
    assert trim_history(messages, 10) == messages


def test_se_queda_con_los_ultimos():
    messages = [HumanMessage(content=str(i)) for i in range(10)]
    trimmed = trim_history(messages, 3)
    assert [m.content for m in trimmed] == ["7", "8", "9"]


def test_nunca_empieza_por_un_tool_message_huerfano():
    # El corte cae justo entre el AIMessage con tool_calls y sus resultados.
    messages = [
        HumanMessage(content="hola"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "t", "args": {}, "id": "c1", "type": "tool_call"},
                {"name": "t", "args": {}, "id": "c2", "type": "tool_call"},
            ],
        ),
        ToolMessage(content="r1", tool_call_id="c1"),
        ToolMessage(content="r2", tool_call_id="c2"),
        AIMessage(content="ya está"),
    ]
    trimmed = trim_history(messages, 3)

    assert not isinstance(trimmed[0], ToolMessage)
    assert [m.content for m in trimmed] == ["ya está"]


def test_ventana_cero_devuelve_todo():
    messages = [HumanMessage(content=str(i)) for i in range(4)]
    assert len(trim_history(messages, 0)) == 4


def test_el_prompt_empieza_por_el_system():
    prompt = build_prompt_messages(
        "eres un bot", [stored(HumanMessage(content="hola"))], 10
    )
    assert isinstance(prompt[0], SystemMessage)
    assert prompt[0].content == "eres un bot"
    assert prompt[1].content == "hola"

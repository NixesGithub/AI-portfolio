"""Tests de la memoria: los dos backends cumplen el mismo contrato."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.memory import InMemoryConversationStore, SQLiteConversationStore


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryConversationStore()
    return SQLiteConversationStore(str(tmp_path / "conv.db"))


async def test_guarda_y_recupera_en_orden(store):
    await store.append("c1", [HumanMessage(content="hola"), AIMessage(content="qué tal")])
    await store.append("c1", [HumanMessage(content="bien")])

    history = await store.get("c1")
    assert [m.message.content for m in history] == ["hola", "qué tal", "bien"]
    assert all(m.id for m in history)


async def test_las_conversaciones_estan_aisladas(store):
    await store.append("c1", [HumanMessage(content="soy c1")])
    await store.append("c2", [HumanMessage(content="soy c2")])

    assert [m.message.content for m in await store.get("c1")] == ["soy c1"]
    assert [m.message.content for m in await store.get("c2")] == ["soy c2"]


async def test_meta_y_existencia(store):
    assert await store.meta("nope") is None
    assert await store.exists("nope") is False

    await store.append("c1", [HumanMessage(content="hola")])
    meta = await store.meta("c1")
    assert meta is not None
    assert meta.message_count == 1
    assert meta.created_at <= meta.updated_at


async def test_limit_devuelve_los_mas_recientes(store):
    await store.append("c1", [HumanMessage(content=str(i)) for i in range(5)])
    history = await store.get("c1", limit=2)
    assert [m.message.content for m in history] == ["3", "4"]


async def test_borrar(store):
    await store.append("c1", [HumanMessage(content="hola")])
    assert await store.delete("c1") is True
    assert await store.delete("c1") is False
    assert await store.get("c1") == []


async def test_conserva_el_tipo_y_las_tool_calls(store):
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "yahoo_finance", "args": {"symbol": "AAPL"}, "id": "c1",
             "type": "tool_call"}
        ],
    )
    await store.append("c1", [ai, ToolMessage(content='{"x":1}', tool_call_id="c1")])

    history = await store.get("c1")
    assert history[0].message.type == "ai"
    assert history[0].message.tool_calls[0]["args"] == {"symbol": "AAPL"}
    assert history[1].message.type == "tool"
    assert history[1].message.tool_call_id == "c1"


async def test_el_tope_de_mensajes_descarta_lo_mas_viejo(tmp_path):
    for store in (
        InMemoryConversationStore(max_messages=3),
        SQLiteConversationStore(str(tmp_path / "cap.db"), max_messages=3),
    ):
        await store.append("c1", [HumanMessage(content=str(i)) for i in range(6)])
        history = await store.get("c1")
        assert [m.message.content for m in history] == ["3", "4", "5"]


async def test_sqlite_persiste_entre_instancias(tmp_path):
    path = str(tmp_path / "persist.db")
    first = SQLiteConversationStore(path)
    await first.append("c1", [HumanMessage(content="sobrevivo")])

    second = SQLiteConversationStore(path)
    history = await second.get("c1")
    assert [m.message.content for m in history] == ["sobrevivo"]
    await second.health()

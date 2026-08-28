"""Tests de la memoria por conversación.

Los mismos casos corren contra las dos implementaciones: si el store de SQLite
y el de memoria se comportan distinto, los tests que usan el rápido dejan de
decir algo sobre el que corre en producción.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.memory import (
    ConversationNotFound,
    ConversationStore,
    InMemoryConversationStore,
    SQLiteConversationStore,
)


@pytest.fixture(params=["memoria", "sqlite"])
async def any_store(request, tmp_path):
    if request.param == "memoria":
        store = InMemoryConversationStore()
    else:
        store = await SQLiteConversationStore.create(str(tmp_path / "test.db"))
    yield store
    await store.close()


async def test_cumple_el_protocolo(any_store):
    assert isinstance(any_store, ConversationStore)


async def test_hilo_inexistente(any_store):
    # `load` falla (el cliente pidió algo que no existe) pero `history` no:
    # mandar el primer mensaje de una conversación es el caso normal.
    with pytest.raises(ConversationNotFound):
        await any_store.load("no-existe")
    assert await any_store.history("no-existe") == []


async def test_los_hilos_estan_aislados_por_id(any_store):
    await any_store.append("a", [HumanMessage("soy A")])
    await any_store.append("b", [HumanMessage("soy B")])

    assert [m.content for m in await any_store.history("a")] == ["soy A"]
    assert [m.content for m in await any_store.history("b")] == ["soy B"]


async def test_conserva_el_orden_y_numera(any_store):
    await any_store.append("c", [HumanMessage("uno"), AIMessage("dos")])
    await any_store.append("c", [HumanMessage("tres")])

    conversation = await any_store.load("c")
    assert [r.seq for r in conversation.messages] == [0, 1, 2]
    assert [r.message.content for r in conversation.messages] == ["uno", "dos", "tres"]


async def test_preserva_las_llamadas_a_herramientas(any_store):
    """Si esto se pierde, el turno siguiente falla con un 400 del proveedor."""
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "get_quote", "args": {"symbol": "AAPL"}, "id": "c1"}],
    )
    await any_store.append("d", [
        HumanMessage("precio?"),
        ai,
        ToolMessage(content="100", tool_call_id="c1", name="get_quote"),
    ])

    restored = await any_store.history("d")
    assert restored[1].tool_calls[0]["args"] == {"symbol": "AAPL"}
    assert restored[1].tool_calls[0]["id"] == "c1"
    assert restored[2].tool_call_id == "c1"
    assert restored[2].name == "get_quote"


async def test_append_vacio_no_crea_el_hilo(any_store):
    await any_store.append("e", [])
    with pytest.raises(ConversationNotFound):
        await any_store.load("e")


async def test_sqlite_sobrevive_al_reinicio(tmp_path):
    """La razón de usar SQLite y no un dict: reiniciar el contenedor no borra
    las conversaciones."""
    path = str(tmp_path / "persist.db")

    store = await SQLiteConversationStore.create(path)
    await store.append("f", [HumanMessage("hola"), AIMessage("qué tal")])
    await store.close()

    reopened = await SQLiteConversationStore.create(path)
    conversation = await reopened.load("f")
    assert [m.message.content for m in conversation.messages] == ["hola", "qué tal"]
    await reopened.close()


async def test_sqlite_marca_timestamps(tmp_path):
    store = await SQLiteConversationStore.create(str(tmp_path / "ts.db"))
    await store.append("g", [HumanMessage("uno")])
    first = await store.load("g")
    await store.append("g", [HumanMessage("dos")])
    second = await store.load("g")

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    await store.close()

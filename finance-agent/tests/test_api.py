"""Tests de la API.

Se ejercita la app real (rutas, validación, serialización) con el servicio
inyectado por dependency override. El transporte ASGI no dispara el `lifespan`,
así que no hace falta ni base de datos ni credenciales del modelo.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.api.deps import get_chat_service
from app.main import create_app
from tests.conftest import ScriptedLLM, fake_quote, tool_call


@pytest.fixture
def client_factory(make_service):
    """Devuelve (cliente, llm) con el guion que pida cada test."""

    def _factory(script):
        service, llm = make_service(script)
        app = create_app()
        app.dependency_overrides[get_chat_service] = lambda: service
        client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        )
        return client, llm, service

    return _factory


async def test_health(client_factory):
    client, _, _ = client_factory([])
    async with client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "x-request-id" in response.headers


async def test_post_chat_crea_un_hilo_nuevo(client_factory):
    client, _, _ = client_factory([AIMessage("Hola, ¿en qué te ayudo?")])
    async with client:
        response = await client.post("/chat", json={"message": "hola"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Hola, ¿en qué te ayudo?"
    assert body["conversation_id"]  # generado por el servicio
    assert body["stop_reason"] == "end_turn"
    assert body["steps"] == []


async def test_el_hilo_recuerda_los_turnos_anteriores(client_factory):
    client, llm, _ = client_factory([AIMessage("Encantado, Germán."),
                                     AIMessage("Te llamás Germán.")])
    async with client:
        first = await client.post(
            "/chat", json={"conversation_id": "hilo-1", "message": "me llamo Germán"}
        )
        second = await client.post(
            "/chat", json={"conversation_id": "hilo-1", "message": "cómo me llamo?"}
        )

    assert first.status_code == second.status_code == 200
    assert second.json()["reply"] == "Te llamás Germán."

    # Lo que importa no es la respuesta guionada sino qué vio el modelo: en la
    # segunda llamada tiene que estar el turno anterior completo.
    contenidos = [m.content for m in llm.calls[1]]
    assert "me llamo Germán" in contenidos
    assert "Encantado, Germán." in contenidos


async def test_los_hilos_no_se_mezclan(client_factory):
    client, llm, _ = client_factory([AIMessage("ok"), AIMessage("ok")])
    async with client:
        await client.post("/chat", json={"conversation_id": "uno", "message": "secreto"})
        await client.post("/chat", json={"conversation_id": "dos", "message": "otro"})

    segunda = [m.content for m in llm.calls[1]]
    assert "secreto" not in segunda


async def test_get_chat_devuelve_el_historial(client_factory):
    client, _, _ = client_factory([AIMessage("Cotiza a 100.")])
    async with client:
        await client.post(
            "/chat", json={"conversation_id": "hilo-2", "message": "precio de AAPL?"}
        )
        response = await client.get("/chat/hilo-2")

    body = response.json()
    assert response.status_code == 200
    assert body["conversation_id"] == "hilo-2"
    assert body["message_count"] == 2
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "precio de AAPL?"
    assert body["messages"][0]["seq"] == 0


async def test_el_historial_esconde_los_pasos_internos_salvo_que_se_pidan(
    client_factory,
):
    """El default es lo que alguien espera de un 'historial conversacional':
    los turnos, no la maquinaria."""
    client, _, _ = client_factory([
        AIMessage("", tool_calls=[tool_call("fake_quote", {"symbol": "AAPL"}, "c1")]),
        AIMessage("AAPL cotiza a 100."),
    ])
    async with client:
        await client.post(
            "/chat", json={"conversation_id": "hilo-3", "message": "precio?"}
        )
        limpio = (await client.get("/chat/hilo-3")).json()
        completo = (await client.get("/chat/hilo-3?include_tools=true")).json()

    assert [m["role"] for m in limpio["messages"]] == ["user", "assistant"]
    assert [m["role"] for m in completo["messages"]] == [
        "user", "assistant", "tool", "assistant",
    ]
    assert completo["messages"][1]["tool_calls"][0]["name"] == "fake_quote"
    assert completo["messages"][2]["tool_call_id"] == "c1"


async def test_los_pasos_de_herramientas_se_reportan_en_la_respuesta(client_factory):
    client, _, _ = client_factory([
        AIMessage("", tool_calls=[tool_call("fake_quote", {"symbol": "AAPL"}, "c1")]),
        AIMessage("AAPL cotiza a 100."),
    ])
    async with client:
        response = await client.post("/chat", json={"message": "precio de AAPL?"})

    steps = response.json()["steps"]
    assert len(steps) == 1
    assert steps[0]["tool"] == "fake_quote"
    assert steps[0]["arguments"] == {"symbol": "AAPL"}
    assert steps[0]["ok"] is True


async def test_hilo_inexistente_da_404(client_factory):
    client, _, _ = client_factory([])
    async with client:
        response = await client.get("/chat/no-existe")

    assert response.status_code == 404
    assert "no-existe" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},                                    # falta message
        {"message": ""},                       # vacío
        {"message": "x" * 5000},               # demasiado largo
        {"message": "hola", "conversation_id": "id con espacios"},
        {"message": "hola", "conversation_id": "x" * 100},
    ],
)
async def test_entradas_invalidas_dan_422(client_factory, payload):
    client, _, _ = client_factory([AIMessage("ok")])
    async with client:
        response = await client.post("/chat", json=payload)

    assert response.status_code == 422


async def test_id_invalido_en_la_url_da_422(client_factory):
    client, _, _ = client_factory([])
    async with client:
        response = await client.get("/chat/con%20espacios")

    assert response.status_code == 422


async def test_dos_mensajes_simultaneos_al_mismo_hilo_no_se_pisan(client_factory):
    """Sin el lock por conversación, los dos turnos razonarían sobre el mismo
    historial viejo y uno de los dos se perdería."""
    client, _, service = client_factory([AIMessage("primera"), AIMessage("segunda")])
    async with client:
        await asyncio.gather(
            client.post("/chat", json={"conversation_id": "race", "message": "a"}),
            client.post("/chat", json={"conversation_id": "race", "message": "b"}),
        )
        historial = (await client.get("/chat/race")).json()

    # Cuatro mensajes: las dos preguntas y las dos respuestas, sin solaparse.
    assert historial["message_count"] == 4
    assert [m["role"] for m in historial["messages"]] == [
        "user", "assistant", "user", "assistant",
    ]
    assert [m["seq"] for m in historial["messages"]] == [0, 1, 2, 3]


async def test_el_request_id_del_cliente_se_respeta(client_factory):
    client, _, _ = client_factory([AIMessage("ok")])
    async with client:
        response = await client.post(
            "/chat", json={"message": "hola"}, headers={"x-request-id": "trace-123"}
        )

    assert response.headers["x-request-id"] == "trace-123"

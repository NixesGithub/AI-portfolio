"""Tests de la API: los dos endpoints del enunciado y su comportamiento real."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def post(client: TestClient, message: str, conversation_id: str | None = None, **kwargs):
    payload: dict = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return client.post("/chat", json=payload, **kwargs)


def test_post_chat_crea_un_hilo_y_usa_la_tool(client):
    response = post(client, "¿A cuánto cotiza Apple?")
    assert response.status_code == 200

    body = response.json()
    assert body["conversation_id"].startswith("conv_")
    assert body["is_new_conversation"] is True
    assert body["reply"]
    assert body["stop_reason"] == "final_answer"
    assert [step["tool"] for step in body["tool_steps"]] == ["yahoo_finance"]
    assert body["tool_steps"][0]["ok"] is True
    assert body["tool_steps"][0]["args"]["symbol"] == "AAPL"
    assert "191.25" in body["reply"]  # el precio sale de la tool, no del modelo


def test_get_chat_devuelve_el_historial(client):
    conversation_id = post(client, "Hola").json()["conversation_id"]
    post(client, "¿Y Microsoft?", conversation_id)

    body = client.get(f"/chat/{conversation_id}").json()
    assert body["conversation_id"] == conversation_id
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert body["messages"][0]["content"] == "Hola"
    assert body["created_at"] <= body["updated_at"]


def test_el_historial_oculta_la_traza_de_tools_salvo_que_se_pida(client):
    conversation_id = post(client, "¿Cómo va Apple?").json()["conversation_id"]

    limpio = client.get(f"/chat/{conversation_id}").json()
    assert {m["role"] for m in limpio["messages"]} == {"user", "assistant"}

    completo = client.get(f"/chat/{conversation_id}?include_tools=true").json()
    roles = [m["role"] for m in completo["messages"]]
    assert "tool" in roles
    llamada = next(m for m in completo["messages"] if m["tool_calls"])
    assert llamada["tool_calls"][0]["name"] == "yahoo_finance"
    resultado = next(m for m in completo["messages"] if m["role"] == "tool")
    assert resultado["tool_call_id"] == llamada["tool_calls"][0]["id"]


def test_el_agente_recuerda_dentro_del_hilo(client):
    conversation_id = "hilo-con-memoria"
    post(client, "me llamo Germán", conversation_id)
    reply = post(client, "¿qué te dije?", conversation_id).json()["reply"]

    assert "me llamo Germán" in reply  # el modelo recibe el turno anterior


def test_los_hilos_no_se_mezclan(client):
    post(client, "secreto de A", "hilo-a")
    post(client, "hola", "hilo-b")

    a = client.get("/chat/hilo-a").json()
    b = client.get("/chat/hilo-b").json()
    assert [m["content"] for m in a["messages"] if m["role"] == "user"] == ["secreto de A"]
    assert "secreto de A" not in str(b["messages"])
    assert b["messages"][-1]["role"] == "assistant"


def test_el_cliente_puede_imponer_su_id(client):
    body = post(client, "hola", "pedido-42").json()
    assert body["conversation_id"] == "pedido-42"
    assert client.get("/chat/pedido-42").status_code == 200


def test_conversacion_inexistente_da_404(client):
    response = client.get("/chat/no-existe")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "conversation_not_found"
    assert response.json()["error"]["request_id"]


def test_validaciones_de_entrada(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 5000}).status_code == 422
    assert post(client, "hola", "id con espacios").status_code == 422
    assert client.get("/chat/id%20invalido").status_code == 422

    error = client.post("/chat", json={"message": ""}).json()["error"]
    assert error["code"] == "validation_error"


def test_borrar_una_conversacion(client):
    post(client, "hola", "para-borrar")
    assert client.delete("/chat/para-borrar").status_code == 204
    assert client.get("/chat/para-borrar").status_code == 404
    assert client.delete("/chat/para-borrar").status_code == 404


def test_la_api_key_protege_los_endpoints(settings, fake_yahoo):
    protegido = create_app(settings.model_copy(update={"api_key": "secreta"}))
    with TestClient(protegido) as client:
        assert client.post("/chat", json={"message": "hola"}).status_code == 401
        assert client.get("/chat/lo-que-sea").status_code == 401

        cabecera = {"X-API-Key": "secreta"}
        creada = client.post("/chat", json={"message": "hola"}, headers=cabecera)
        assert creada.status_code == 200
        # La salud queda fuera de la autenticación: la sondea el orquestador.
        assert client.get("/health/ready").status_code == 200

        mala = client.post(
            "/chat", json={"message": "hola"}, headers={"X-API-Key": "otra"}
        )
        assert mala.status_code == 401


def test_id_de_correlacion(client):
    generado = post(client, "hola")
    assert generado.headers["X-Request-ID"]

    propio = client.post(
        "/chat", json={"message": "hola"}, headers={"X-Request-ID": "trace-123"}
    )
    assert propio.headers["X-Request-ID"] == "trace-123"


def test_health(client):
    vivo = client.get("/health/live").json()
    assert vivo["status"] == "ok"
    assert vivo["llm_provider"] == "fake"

    listo = client.get("/health/ready").json()
    assert listo["status"] == "ok"
    assert listo["tools"] == ["yahoo_finance"]


def test_openapi_documenta_los_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/chat" in paths
    assert "/chat/{conversation_id}" in paths
    assert "post" in paths["/chat"]


def test_un_fallo_de_la_tool_no_rompe_la_respuesta(client, monkeypatch):
    from app.agent.tools import yahoo_finance

    def explode(symbol: str):
        raise ConnectionError("Yahoo caído")

    monkeypatch.setattr(yahoo_finance, "_get_ticker", explode)
    body = client.post("/chat", json={"message": "¿Cómo va Apple?"}).json()

    assert body["tool_steps"][0]["ok"] is True  # la tool degrada, no lanza
    assert "no está respondiendo" in body["reply"]

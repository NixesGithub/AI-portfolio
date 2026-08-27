"""Tests del recorte de historial."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.history import prepare


def _turno(n: int) -> list:
    return [HumanMessage(f"pregunta {n} " + "x" * 400), AIMessage(f"respuesta {n}")]


def test_siempre_encabeza_con_el_system_prompt():
    resultado = prepare("SOY EL SISTEMA", _turno(1), max_tokens=10_000)

    assert isinstance(resultado[0], SystemMessage)
    assert resultado[0].content == "SOY EL SISTEMA"


def test_recorta_los_turnos_viejos_cuando_no_entran():
    largo = [m for n in range(20) for m in _turno(n)]

    completo = prepare("sistema", largo, max_tokens=100_000)
    recortado = prepare("sistema", largo, max_tokens=500)

    assert len(recortado) < len(completo)
    # Se conserva lo último, que es lo que da contexto al turno actual.
    assert recortado[-1].content == "respuesta 19"


def test_el_recorte_no_deja_un_tool_message_huerfano():
    """Un ToolMessage sin el AIMessage que lo pidió hace que el proveedor
    devuelva 400 en el turno siguiente."""
    conversacion = [
        HumanMessage("vieja " + "x" * 2000),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage(content="resultado", tool_call_id="c1", name="t"),
        AIMessage("respuesta vieja"),
        HumanMessage("nueva"),
    ]

    recortado = prepare("sistema", conversacion, max_tokens=60)

    tipos = [m.type for m in recortado]
    if "tool" in tipos:
        assert tipos.index("ai") < tipos.index("tool")
    # El primer mensaje después del system prompt es siempre un turno de usuario.
    assert recortado[1].type == "human"


def test_historial_vacio():
    resultado = prepare("sistema", [], max_tokens=1000)
    assert len(resultado) == 1
    assert resultado[0].type == "system"

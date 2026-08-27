# Enunciado

> Construir una API en Python que exponga un agente conversacional con memoria por
> conversación (id) y una tool con capacidad de consultar información financiera de
> Yahoo Finance.
>
> Utilizar Langchain y definir explícitamente la función de `agent_loop`.
>
> Se debe poder mantener distintas conversaciones, cada una con su historial, por id.
>
> Se deben exponer dos endpoints:
> - `POST /chat`: enviar un mensaje a un hilo de conversación
> - `GET /chat/{id}`: debe devolver el historial conversacional de un id en particular
>
> Debe correr en Docker.
>
> Simulando que esto podría ser una app productiva, tomar cualquier decisión de
> diseño que considere adecuada considerando lo pedido.

## Dónde está resuelto cada punto

| Requisito | Dónde |
|---|---|
| API en Python | FastAPI — `app/main.py`, `app/api/routes.py` |
| Agente conversacional | `app/agent/service.py` + `app/agent/prompts.py` |
| Memoria por conversación (id) | `app/memory/` (backends en proceso y SQLite, misma interfaz) |
| Tool de Yahoo Finance | `app/agent/tools/yahoo_finance.py` (`quote`, `history`, `profile`) |
| LangChain | `langchain-core` para mensajes, tools y *tool calling*; `langchain-anthropic` como proveedor |
| `agent_loop` explícito | `app/agent/loop.py` — bucle propio, sin `AgentExecutor` |
| Varias conversaciones aisladas | `conversation_id` en cada llamada; test `test_los_hilos_no_se_mezclan` |
| `POST /chat` | `app/api/routes.py::post_chat` |
| `GET /chat/{id}` | `app/api/routes.py::get_chat` |
| Docker | `Dockerfile` (multi-etapa, usuario sin privilegios, healthcheck) y `docker-compose.yml` |
| Decisiones de producción | Sección «Decisiones de diseño» del [README](README.md) |

## Cómo verificarlo en dos minutos

```bash
docker compose up --build          # o: LLM_PROVIDER=fake docker compose up --build
./scripts/demo.sh                  # dos turnos con memoria + historial + otro hilo
```

Y los tests, que no necesitan red ni credenciales:

```bash
make install && make test
```

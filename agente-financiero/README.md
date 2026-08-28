# Agente financiero conversacional

API en Python que expone un agente de chat con **memoria por conversación** y una
**herramienta de datos de mercado de Yahoo Finance**. El bucle del agente está
escrito a mano (nada de `AgentExecutor`), corre en Docker y está pensado como si
fuese a producción: configuración por entorno, logs estructurados, tiempos de
espera, autenticación opcional, persistencia y tests.

```
POST /chat  ──▶  cargar hilo por id ──▶ ventana de contexto ──▶ agent_loop
                                                                  │
                                            ┌─────────────────────┴───────────┐
                                            │  modelo  ⇄  tool yahoo_finance  │
                                            └─────────────────────┬───────────┘
                                                                  ▼
GET /chat/{id} ◀── historial persistido ◀── guardar turno completo ◀── respuesta
```

## Arranque rápido

```bash
cp .env.example .env      # y pon tu ANTHROPIC_API_KEY
docker compose up --build
```

La API queda en `http://localhost:8000`, con la documentación interactiva en
`http://localhost:8000/docs`.

**¿Sin clave de Anthropic?** Se puede ver el circuito completo igualmente,
poniendo `LLM_PROVIDER=fake` en el `.env` (o sin fichero ninguno):

```bash
docker compose run --rm -e LLM_PROVIDER=fake -p 8000:8000 api
```

`LLM_PROVIDER=fake` sustituye el modelo por uno determinista que sí llama a la
tool y sí usa la memoria. Sirve para revisar la API, para los tests y para CI.

En local, sin Docker:

```bash
make install && make run      # http://localhost:8000/docs
make test                     # 57 tests, sin red ni credenciales
make demo                     # conversación de ejemplo con curl
```

## Los dos endpoints

### `POST /chat` — enviar un mensaje

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "¿A cuánto cotiza Apple y cómo le ha ido este mes?"}'
```

```json
{
  "conversation_id": "conv_9f2c1b7d4e6a0c3b5d8e1f2a",
  "message_id": "msg_3a7d9e1c5b2f4086ad9c1e77",
  "reply": "AAPL cotiza a 191,25 USD (+1,73 % frente al cierre anterior)…",
  "created_at": "2026-08-27T10:31:04.912Z",
  "is_new_conversation": true,
  "stop_reason": "final_answer",
  "iterations": 3,
  "latency_ms": 4120,
  "tool_steps": [
    {"tool": "yahoo_finance", "args": {"symbol": "AAPL", "section": "quote"},
     "ok": true, "latency_ms": 310, "error": null},
    {"tool": "yahoo_finance", "args": {"symbol": "AAPL", "section": "history",
     "period": "1mo", "interval": "1d"}, "ok": true, "latency_ms": 268, "error": null}
  ],
  "usage": {"input_tokens": 1843, "output_tokens": 214, "total_tokens": 2057}
}
```

`conversation_id` es opcional: sin él se abre un hilo nuevo y su id viene en la
respuesta; con él, el agente recuerda todo lo hablado antes. El cliente puede
imponer su propio id (`"conversation_id": "pedido-42"`), que es lo normal cuando
esto se integra con un sistema que ya tiene sus identificadores.

`tool_steps` y `usage` no son adorno: son lo que permite responder «¿por qué
contestó esto?» y «¿cuánto costó?» sin encender un debugger.

### `GET /chat/{id}` — historial

```bash
curl http://localhost:8000/chat/conv_9f2c1b7d4e6a0c3b5d8e1f2a
```

```json
{
  "conversation_id": "conv_9f2c1b7d4e6a0c3b5d8e1f2a",
  "created_at": "2026-08-27T10:30:58.104Z",
  "updated_at": "2026-08-27T10:31:04.912Z",
  "message_count": 6,
  "messages": [
    {"id": "msg_…", "role": "user", "content": "¿A cuánto cotiza Apple…",
     "created_at": "2026-08-27T10:30:58.104Z"},
    {"id": "msg_…", "role": "assistant", "content": "AAPL cotiza a 191,25 USD…",
     "created_at": "2026-08-27T10:31:04.912Z"}
  ]
}
```

Por defecto devuelve la conversación tal y como la vería un usuario. La traza
interna (llamadas a la tool y sus resultados) se pide aparte:

| Parámetro | Qué hace |
|---|---|
| `include_tools=true` | Añade los mensajes de tool y las `tool_calls` del agente |
| `limit=N` | Sólo los N mensajes más recientes |

Y hay dos endpoints más, que no pide el enunciado pero que un servicio real
necesita: `DELETE /chat/{id}` (retención de datos) y `GET /health/live` /
`GET /health/ready` (sondas del orquestador).

## Cómo está montado

```
app/
├── main.py                    Ensamblado de la app, middleware, ciclo de vida
├── config.py                  Configuración por entorno (12-factor)
├── logging.py                 Logs JSON con request_id
├── errors.py                  Errores de dominio → respuestas HTTP
├── api/
│   ├── routes.py              POST /chat, GET /chat/{id}, DELETE, health
│   ├── schemas.py             Contratos de entrada/salida (y el OpenAPI)
│   └── deps.py                Inyección del servicio y API key
├── agent/
│   ├── loop.py       ★        agent_loop: el bucle, explícito
│   ├── service.py             Memoria + prompt + bucle, orquestados
│   ├── history.py             Ventana de contexto
│   ├── prompts.py             Prompt de sistema
│   ├── models.py              Fábrica del modelo (Anthropic / fake)
│   └── tools/
│       ├── yahoo_finance.py   La tool
│       └── cache.py           Caché TTL de las respuestas de Yahoo
└── memory/
    ├── base.py                Interfaz del almacén
    ├── in_memory.py           Backend en proceso
    ├── sqlite.py              Backend persistente
    └── locks.py               Cerrojo por conversación
```

Las capas no se saltan: **HTTP no sabe qué es LangChain, y el bucle no sabe qué
es HTTP**. Entre medias está `ChatService`, que es el único que conoce las dos
cosas. Cambiar de framework web o de proveedor de modelo toca un módulo, no el
proyecto.

## El `agent_loop`

Vive en `app/agent/loop.py` y es un ReAct clásico sobre *tool calling*:

```python
result = await agent_loop(
    llm=llm,                    # modelo con bind_tools
    tools=tools,                # [yahoo_finance]
    messages=prompt_messages,   # system + ventana del historial + mensaje nuevo
    max_iterations=6,
    tool_timeout_s=15.0,
)
```

```
   ┌─────────────────────────────────────────────┐
   │  modelo.ainvoke(conversación)               │
   └───────────────┬─────────────────────────────┘
                   │
        ¿pide tools?├── no ──▶ respuesta final  (stop_reason=final_answer)
                   │ sí
                   ▼
   ejecutar las tool_calls EN PARALELO, con timeout por tool
                   │
   añadir un ToolMessage por cada tool_call (mismo id, mismo orden)
                   │
                   └──▶ volver a empezar, hasta max_iterations
                        y entonces forzar una respuesta sin tools
                                          (stop_reason=max_iterations)
```

Está escrito a mano por la misma razón por la que no se usa un ORM mágico para
una consulta crítica: **lo que hay que controlar está justo ahí dentro**.

- **Presupuesto de vueltas.** `max_iterations` acota coste y latencia. Al
  agotarse no se devuelve un error: se hace una última llamada *sin* tools para
  que el modelo cierre el turno con lo que tenga. El usuario recibe una respuesta
  útil, no un 500.
- **Una tool que falla no tumba la petición.** Excepción, timeout o herramienta
  inexistente se convierten en un `ToolMessage` con `status="error"` y un texto
  que el modelo entiende, así que puede reintentar con otros argumentos o
  explicárselo al usuario.
- **El invariante de los mensajes se respeta siempre.** Cada `AIMessage` con
  `tool_calls` va seguido de un `ToolMessage` por llamada, con su `tool_call_id`
  y en el mismo orden. Romperlo produce 400 del proveedor difíciles de
  diagnosticar; hay un test por cada forma de romperlo.
- **Varias tools de una misma vuelta van en paralelo** (`asyncio.gather`):
  comparar dos valores cuesta lo que el más lento, no la suma.
- **Es puro respecto al almacenamiento.** Recibe mensajes y devuelve mensajes;
  quién los persiste es problema del servicio. Por eso se puede testear entero
  con un modelo de mentira y sin base de datos.
- **Emite eventos** (`llm_start`, `tool_end`, `max_iterations`) por un callback
  opcional: el enganche natural para métricas o para *streaming* por SSE.

## Memoria por conversación

Un `conversation_id` es un hilo: su historial completo, en orden, y aislado del
resto. Dos decisiones que no son evidentes:

**Qué ve el modelo ≠ qué se guarda.** Se guarda todo (incluidas las llamadas a
tools, que son la auditoría de por qué contestó lo que contestó); al modelo se le
pasan los últimos `HISTORY_WINDOW` mensajes. Con el recorte hay una regla
innegociable: la ventana **no puede empezar por un `ToolMessage` huérfano**, o el
proveedor devuelve un 400. Es un fallo que sólo aparece en hilos largos —es
decir, en producción y no en las pruebas—, así que tiene su propio test.

**Un hilo, un turno a la vez.** Dos peticiones simultáneas al mismo id
intercalarían mensajes y dejarían el historial incoherente. Hay un cerrojo *por
conversación*, no global: hilos distintos siguen yendo en paralelo.

Y si el turno falla (el proveedor se cae, se agota el tiempo), **no se persiste
nada**: el usuario reintenta y no le queda un mensaje suyo colgando sin
respuesta.

| Backend | Cuándo | Qué pasa al reiniciar |
|---|---|---|
| `memory` (por defecto en local) | Tests, desarrollo | Se pierde |
| `sqlite` (por defecto en Docker) | Un contenedor con volumen | Sobrevive |

Los dos implementan la misma interfaz (`app/memory/base.py`) y **los dos pasan
exactamente los mismos tests**. Poner Redis o Postgres detrás es escribir una
tercera clase.

## La tool de Yahoo Finance

Una sola herramienta, `yahoo_finance`, con tres secciones:

| `section` | Devuelve |
|---|---|
| `quote` | Precio, variación del día, rango de 52 semanas, capitalización |
| `history` | Evolución en un periodo, con máximos, mínimos y la serie muestreada |
| `profile` | Nombre, sector, industria, país, tamaño, descripción |

Una tool con un esquema tipado en vez de tres tools sueltas: al modelo le cuesta
menos elegir bien entre pocas herramientas bien descritas, y los `Literal` de
`section`, `period` e `interval` evitan de entrada la mayoría de las llamadas
inválidas. Acepta cualquier ticker de Yahoo: `AAPL`, `SAN.MC`, `^GSPC`,
`EURUSD=X`, `BTC-USD`.

Lo que hace que sea utilizable en serio:

- **La salida se normaliza.** El modelo recibe un JSON pequeño y estable, no el
  volcado de `yfinance`: menos tokens y menos ocasiones de alucinar.
- **Las series se recortan.** Un año de velas diarias son 250 filas; se muestrean
  a 30 puntos conservando el máximo, el mínimo y —siempre— el último cierre.
- **Caché TTL** (60 s cotización, 5 min histórico, 24 h ficha): tres preguntas
  seguidas sobre Apple no son tres llamadas a Yahoo.
- **Los errores son texto para el modelo**, no excepciones: un ticker que no
  existe vuelve como `symbol_not_found` con una pista sobre los sufijos de
  mercado, y el agente pregunta en vez de inventarse el precio.
- **`yfinance` es bloqueante**, así que la versión `async` de la tool lo manda a
  un hilo. Un event loop parado en una llamada de red es un servidor que no
  atiende a nadie más.

El prompt de sistema es explícito: cualquier dato de mercado sale de la tool,
nunca de la memoria del modelo, que está desactualizada por definición.

## Decisiones de diseño

**El `conversation_id` lo puede poner el cliente.** Si el que llama ya tiene un
id de ticket, de usuario o de sesión, obligarle a guardar otro más es fricción
gratuita. Se valida con un patrón (`^[A-Za-z0-9._:-]{1,64}$`) y punto.

**Todo lo que sale del proceso tiene tiempo límite**: la tool
(`TOOL_TIMEOUT_S`), el turno completo (`AGENT_TIMEOUT_S`) y el propio modelo
(`LLM_TIMEOUT_S`, con reintentos y *backoff* ante 429/5xx del proveedor). Sin
límites, una dependencia lenta se traduce en trabajadores bloqueados y, de ahí,
en una caída completa.

**Los errores tienen forma fija** — `{"error": {"code", "message", "request_id"}}` —
con códigos estables (`conversation_not_found`, `agent_timeout`,
`upstream_error`, `validation_error`). Un cliente puede programar contra ellos;
contra un texto libre, no.

**Logs JSON con `request_id`** en cada línea, respetando el `X-Request-ID` que
venga del gateway y devolviéndolo siempre. Cada turno deja una línea con
iteraciones, tools usadas, latencia y tokens.

**Autenticación opcional por `API_KEY`** (cabecera `X-API-Key`, comparación en
tiempo constante), desactivada en local y obligatoria en cuanto se define. Las
sondas de salud quedan fuera.

**Un worker por contenedor.** El escalado lo decide el orquestador con réplicas,
no el Dockerfile con procesos. La imagen es multi-etapa, corre como usuario sin
privilegios y trae `HEALTHCHECK`.

**Modelo `fake` de primera clase.** No es un mock de los tests escondido: es un
proveedor más, seleccionable por entorno. Gracias a él los 57 tests corren en CI
en dos segundos sin red ni credenciales, y cualquiera puede levantar el servicio
y ver cómo funciona sin tener cuenta de Anthropic.

## Tests

```bash
make test
```

57 tests, sin red y sin claves: el modelo es el `fake` y Yahoo Finance se
sustituye por un ticker de mentira, así que se ejercita el circuito real entero
salvo las dos fronteras externas.

| Fichero | Qué cubre |
|---|---|
| `test_agent_loop.py` | Vuelta con tool, tools en paralelo, tool que falla, timeout, tool inexistente, corte por iteraciones, tokens, eventos |
| `test_api.py` | Los dos endpoints, aislamiento entre hilos, memoria, `include_tools`, validaciones, 404, API key, `X-Request-ID` |
| `test_service.py` | Serialización por hilo, paralelismo entre hilos, timeout y fallo del proveedor, historial limpio tras un fallo |
| `test_stores.py` | Mismo contrato en los dos backends, tipos y `tool_calls` preservados, tope de mensajes, persistencia |
| `test_yahoo_finance.py` | Normalización, muestreo de series, símbolo inexistente, Yahoo caído, caché, validación del ticker |
| `test_history.py` | Ventana de contexto y el invariante del `ToolMessage` huérfano |

## Configuración

Todo por variables de entorno (ver `.env.example`):

| Variable | Por defecto | Para qué |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` o `fake` |
| `LLM_MODEL` | `claude-sonnet-5` | Modelo de chat |
| `ANTHROPIC_API_KEY` | — | Obligatoria con `LLM_PROVIDER=anthropic` |
| `STORE_BACKEND` | `memory` (`sqlite` en Docker) | Dónde vive el historial |
| `SQLITE_PATH` | `/data/conversations.db` | Fichero del historial |
| `API_KEY` | vacío | Si se define, exige `X-API-Key` |
| `AGENT_MAX_ITERATIONS` | `6` | Vueltas máximas del bucle |
| `AGENT_TIMEOUT_S` | `90` | Presupuesto de un turno |
| `TOOL_TIMEOUT_S` | `15` | Presupuesto de una tool |
| `HISTORY_WINDOW` | `40` | Mensajes que ve el modelo |
| `MAX_STORED_MESSAGES` | `400` | Tope guardado por hilo |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | Logs |

## Qué falta para producción de verdad

Lo que queda fuera del alcance de este ejercicio, en orden de importancia:

1. **Almacén compartido** (Redis o Postgres) y **cerrojo distribuido**. Tal cual
   está, varias réplicas comparten historial sólo si comparten el volumen, y el
   cerrojo por conversación es de proceso. La interfaz ya está preparada; es
   escribir una clase más.
2. **Streaming de la respuesta** por SSE. El bucle ya emite eventos; falta el
   endpoint y consumir el *streaming* del modelo.
3. **Resumen de conversaciones largas**: hoy la ventana descarta lo viejo; lo
   suyo es resumirlo y conservar el resumen como contexto.
4. **Límite de peticiones por clave** y cuota de tokens por conversación, para
   que un cliente no se lleve por delante el presupuesto de todos.
5. **Métricas y trazas** (OpenTelemetry): latencia por vuelta, tasa de error por
   tool, tokens por conversación. Los eventos del bucle son el enganche.
6. **Evaluación del agente**: un conjunto de conversaciones de referencia que se
   ejecute en CI, porque un cambio de prompt puede degradar la calidad sin que
   ningún test unitario se entere.

---

Esto es una demo técnica. La información que devuelve **no es asesoramiento
financiero**, y los datos de Yahoo Finance llegan con retraso y sin garantías.

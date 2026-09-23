# AI-portfolio

This is a portfolio project demonstrating my AI skills.

## Noticias IA diarias (retirado)

El digest diario de noticias de IA que corría acá (Python + GitHub Actions) se
retiró. Ahora vive en [`newsletter-api`](https://github.com/NixesGithub/newsletter-api),
que lo genera, lo publica y lo envía por email y Telegram a los suscriptores.
El código anterior sigue en el historial de git de este repo.

## Finance Agent API

Un agente conversacional con memoria por hilo que consulta datos de mercado de
Yahoo Finance. FastAPI + LangChain, con el bucle del agente escrito a mano en
vez de delegarlo en `AgentExecutor`, y corriendo en Docker.

```
POST /chat        → mensaje a un hilo (lo crea si no existe)
GET  /chat/{id}   → historial de ese hilo
```

Está en [`finance-agent/`](finance-agent/), con el detalle de las decisiones de
diseño —memoria como interfaz, locks por conversación, recorte de historial por
tokens, qué pasa cuando una herramienta falla— en su propio README.

# syntax=docker/dockerfile:1

# Imagen fina y multi-etapa: las dependencias se compilan/instalan una vez y la
# imagen final sólo lleva lo que hace falta para servir.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Compiladores sólo en la etapa de construcción: alguna dependencia de yfinance
# se distribuye como sdist y necesita toolchain. La imagen final no los lleva.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    STORE_BACKEND=sqlite \
    SQLITE_PATH=/data/conversations.db

# curl sólo para el HEALTHCHECK; el resto de la imagen se queda desnuda.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app

# Nada de root dentro del contenedor.
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /data && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/ready || exit 1

# Un worker por contenedor: se escala con réplicas, no con procesos dentro de
# la caja (así el escalado lo decide el orquestador y no el Dockerfile).
# --no-access-log porque el log de acceso ya lo emite la app, en JSON y con
# request_id; el de uvicorn sólo duplicaría líneas.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log"]

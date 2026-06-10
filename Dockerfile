FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data

# Non-root runtime user; state lives under /app/.sentinel
RUN useradd --create-home sentinel && \
    mkdir -p /app/.sentinel && \
    chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["sentinel", "serve"]

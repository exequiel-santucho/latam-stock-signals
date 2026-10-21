FROM python:3.12-slim

# uv for a reproducible install from uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config.py main.py ./
COPY scripts/ scripts/

# data/ and output/ are mounted as volumes so the incremental history survives between runs
VOLUME ["/app/data", "/app/output"]
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "main.py"]

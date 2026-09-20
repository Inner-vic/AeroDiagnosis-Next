ARG PYTHON_BASE_IMAGE=docker.m.daocloud.io/library/python:3.13-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/Inner-vic/AeroDiagnosis-Next"
LABEL org.opencontainers.image.description="Knowledge-enhanced aero-engine diagnosis agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    AERODIAGNOSIS_RUNTIME_DIR=/app/runtime \
    AERODIAGNOSIS_DATABASE_PATH=/app/runtime/data/aerodiagnosis.db \
    AERODIAGNOSIS_VECTOR_BACKEND=sqlite \
    AERODIAGNOSIS_GRAPH_BACKEND=sqlite \
    AERODIAGNOSIS_API_HOST=127.0.0.1 \
    AERODIAGNOSIS_API_PORT=8080 \
    AERODIAGNOSIS_SEED_DEMO_CONTENT=true

ARG UV_VERSION=0.12.2
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PYPI_PACKAGE_MIRROR_URL=https://pypi.tuna.tsinghua.edu.cn
ENV PIP_INDEX_URL=${PYPI_INDEX_URL} \
    UV_DEFAULT_INDEX=${PYPI_INDEX_URL}
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

RUN groupadd --system --gid 10001 aerodiagnosis \
    && useradd --system --uid 10001 --gid aerodiagnosis --home-dir /app aerodiagnosis

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN sed -i "s#https://files.pythonhosted.org#${PYPI_PACKAGE_MIRROR_URL}#g" uv.lock \
    && uv sync --frozen --no-dev --no-editable --extra external-stores \
    && mkdir -p /app/runtime/data \
    && chown -R aerodiagnosis:aerodiagnosis /app/runtime

USER aerodiagnosis

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=2).read()"]

CMD ["python", "-m", "aerodiagnosis.docker_entrypoint"]

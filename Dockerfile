# build: common base shared by prod and dev -- system libs, locked runtime
# dependencies, and the installed project. Nothing dev-only lives here.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# rasterio's manylinux wheel bundles GDAL but still links libexpat; ca-certificates
# is needed for the authenticated HTTPS reads from LP DAAC during seeding.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, in a layer keyed only on the lockfile so it is reused
# across source-only changes. The project itself is installed separately.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# prod: the lean image shipped to production -- CLI entrypoint, no dev deps,
# no tests or seed tooling.
FROM build AS prod
ENTRYPOINT ["hls-composites"]

# dev: local development -- adds dev dependencies (pytest etc.), the test
# suite, dev scripts, and the seed tooling. Not for production.
FROM build AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group seed
COPY tests ./tests
COPY scripts ./scripts
COPY docker ./docker
ENTRYPOINT ["hls-composites"]

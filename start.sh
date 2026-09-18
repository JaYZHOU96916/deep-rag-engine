#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop or Docker Engine, then retry." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "The Docker daemon is not running. Start Docker, then retry." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Replace development passwords and provider credentials before production use."
fi

if grep -q '^POSTGRES_PASSWORD=change-me-before-deploying$' .env; then
  echo "Refusing to start with the example PostgreSQL password. Set POSTGRES_PASSWORD in .env first." >&2
  exit 1
fi

echo "Building and starting Deep-RAG services…"
docker compose --profile application up --build --detach --wait

echo "Verifying storage extensions and API health…"
docker compose exec --no-TTY postgres psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER:-deep_rag}" -d "${POSTGRES_DB:-deep_rag}" -c "SELECT extname FROM pg_extension WHERE extname = 'vector';" >/dev/null

backend_port="${BACKEND_PORT:-8000}"
frontend_port="${FRONTEND_PORT:-3000}"

if command -v curl >/dev/null 2>&1; then
  curl --fail --silent "http://localhost:${backend_port}/healthz" >/dev/null
fi

echo "Deep-RAG is ready."
echo "  Frontend: http://localhost:${frontend_port}"
echo "  Backend:  http://localhost:${backend_port}/docs"

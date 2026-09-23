#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This build script currently produces a macOS Apple Silicon installer." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to build the desktop app." >&2
  exit 1
fi

# Electron Forge 7.11 is not yet compatible with the host's Node 26 runtime.
node_runtime_dir="$project_dir/.venv-desktop/node-runtime"
if [[ ! -x "$node_runtime_dir/node_modules/node/bin/node" ]]; then
  npm install --prefix "$node_runtime_dir" node@22.23.2 --cache /tmp/deep-rag-desktop-npm-cache
fi
export PATH="$node_runtime_dir/node_modules/node/bin:$PATH"
if [[ "$(node -p 'process.versions.node')" != "22.23.2" ]]; then
  echo "The desktop build requires Node.js 22.23.2." >&2
  exit 1
fi

python_binary="${DESKTOP_PYTHON:-}"
if [[ -z "$python_binary" ]]; then
  for candidate in python3.12 python3.13 python3.14; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_binary="$candidate"
      break
    fi
  done
fi
if [[ -z "$python_binary" ]]; then
  echo "Python 3.12+ is required to build the desktop app. Set DESKTOP_PYTHON to its path." >&2
  exit 1
fi

if [[ ! -x .venv-desktop/bin/python ]]; then
  "$python_binary" -m venv .venv-desktop
fi
.venv-desktop/bin/python -m pip install -r backend/requirements-desktop.txt

(
  cd backend
  ../.venv-desktop/bin/pyinstaller \
    --noconfirm --clean --onedir --name deep-rag-api \
    --paths . --hidden-import desktop_api.main \
    --collect-submodules uvicorn --collect-submodules keyring.backends \
    desktop_api/server.py
)

(
  cd frontend
  if [[ ! -d node_modules ]]; then npm ci --cache /tmp/deep-rag-desktop-npm-cache; fi
  DEEP_RAG_DESKTOP_EXPORT=1 NEXT_PUBLIC_API_BASE_URL='' npm run build
)

(
  cd desktop
  npm ci --cache /tmp/deep-rag-desktop-npm-cache
  npm run make
)

installer="$project_dir/desktop/out/make/Deep-RAG.dmg"
if [[ ! -f "$installer" ]]; then
  echo "Electron Forge did not produce the expected installer: $installer" >&2
  exit 1
fi
echo "Desktop installer: $installer"

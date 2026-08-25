#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required. Install it or run: PYTHON_BIN=/path/to/python3.12 ./run.sh" >&2
  exit 1
fi

if [ ! -d "$ROOT/backend/.venv" ]; then
  "$PYTHON_BIN" -m venv "$ROOT/backend/.venv"
fi

"$ROOT/backend/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/backend/.venv/bin/python" -m pip install -r "$ROOT/backend/requirements.txt"

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  npm --prefix "$ROOT/frontend" ci
fi

"$ROOT/backend/.venv/bin/python" "$ROOT/backend/smoke_test.py"

trap 'kill 0' EXIT
"$ROOT/backend/.venv/bin/python" -m uvicorn app.main:app --app-dir "$ROOT/backend" --host 127.0.0.1 --port 8000 &
npm --prefix "$ROOT/frontend" run dev

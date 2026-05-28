#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
VENV_DIR="$PROJECT_DIR/.venv"
DEPS_MARKER="$VENV_DIR/.deps-installed"

if [[ ! -f ".env" ]]; then
  cp ".env.example" ".env"
  echo "Created .env from .env.example. Add your provider API key when you want real LLM output."
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python3"
  fi
  echo "Creating virtual environment with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [[ ! -f "$DEPS_MARKER" || "requirements.txt" -nt "$DEPS_MARKER" ]]; then
  echo "Installing/updating dependencies..."
  pip install -r requirements.txt
  touch "$DEPS_MARKER"
fi

echo ""
echo "Starting Multimodal RAG debug server"
echo "Project: $PROJECT_DIR"
echo "API:     http://$HOST:$PORT"
echo "Docs:    http://$HOST:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop."
echo ""

if [[ "$OPEN_BROWSER" == "1" ]] && command -v open >/dev/null 2>&1; then
  (sleep 2 && open "http://$HOST:$PORT") >/dev/null 2>&1 &
fi

exec uvicorn app.main:app --reload --host "$HOST" --port "$PORT"

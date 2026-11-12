#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install it from https://www.python.org/downloads/ and run this command again."
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating Python environment…"
  python3 -m venv .venv
fi

PYTHON="$PROJECT_DIR/.venv/bin/python"
PIP="$PROJECT_DIR/.venv/bin/pip"
UVICORN="$PROJECT_DIR/.venv/bin/uvicorn"

if ! "$PYTHON" -c 'import fastapi, uvicorn, cv2, ee' >/dev/null 2>&1; then
  echo "Installing TerraScope requirements…"
  "$PIP" install -r requirements.txt
fi

if [[ -z "${EARTH_ENGINE_PROJECT:-}" ]]; then
  export EARTH_ENGINE_PROJECT="satellite-change-ai"
fi

if [[ -z "${EARTH_ENGINE_SERVICE_ACCOUNT_JSON:-}" ]] && ! "$PYTHON" -c 'import ee; ee.data.getAlgorithms()' >/dev/null 2>&1; then
  echo "Google Earth Engine is not authenticated yet. Starting authentication…"
  if [[ -x "$PROJECT_DIR/.venv/bin/earthengine" ]]; then
    "$PROJECT_DIR/.venv/bin/earthengine" authenticate
  elif command -v earthengine >/dev/null 2>&1; then
    earthengine authenticate
  else
    "$PYTHON" -m ee.cli authenticate
  fi
fi

URL="http://127.0.0.1:8000"
echo "Starting TerraScope at $URL"
(
  sleep 2
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  fi
) >/dev/null 2>&1 &
exec "$UVICORN" app:app --host 127.0.0.1 --port 8000

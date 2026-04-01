#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INGEST_MODE="auto"
for arg in "$@"; do
  case "$arg" in
    --ingest)
      INGEST_MODE="always"
      ;;
    --skip-ingest)
      INGEST_MODE="never"
      ;;
    *)
      ;;
  esac
done

if [[ ! -d "$ROOT_DIR/myenv" ]]; then
  python3 -m venv "$ROOT_DIR/myenv"
fi

source "$ROOT_DIR/myenv/bin/activate"

REQ_FILE="requirements.txt"
if [[ ! -f "$ROOT_DIR/$REQ_FILE" && -f "$ROOT_DIR/requirement.txt" ]]; then
  REQ_FILE="requirement.txt"
fi

if [[ ! -f "$ROOT_DIR/$REQ_FILE" ]]; then
  echo "No requirements file found (requirements.txt or requirement.txt)."
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/$REQ_FILE"

KB_DIR="$ROOT_DIR/knowledge_base"
STATE_DIR="$ROOT_DIR/.cache"
STATE_FILE="$STATE_DIR/ingest_state.sha256"

run_ingest_if_needed() {
  if [[ "$INGEST_MODE" == "never" ]]; then
    echo "Skipping ingestion (--skip-ingest)."
    return
  fi

  if [[ "$INGEST_MODE" == "always" ]]; then
    echo "Running ingestion (--ingest)."
    python "$ROOT_DIR/ingest.py"
    mkdir -p "$STATE_DIR"
    current_state="$(python - <<'PY'
import hashlib
import os
root = os.path.join(os.getcwd(), "knowledge_base")
if not os.path.isdir(root):
    print("no_kb")
    raise SystemExit(0)
items = []
for base, _, files in os.walk(root):
    for name in files:
        path = os.path.join(base, name)
        rel = os.path.relpath(path, root)
        stat = os.stat(path)
        items.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")
payload = "\n".join(sorted(items)).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
)"
    printf "%s" "$current_state" > "$STATE_FILE"
    return
  fi

  if [[ ! -d "$KB_DIR" ]]; then
    echo "knowledge_base folder not found. Skipping ingestion."
    return
  fi

  current_state="$(python - <<'PY'
import hashlib
import os
root = os.path.join(os.getcwd(), "knowledge_base")
if not os.path.isdir(root):
    print("no_kb")
    raise SystemExit(0)
items = []
for base, _, files in os.walk(root):
    for name in files:
        path = os.path.join(base, name)
        rel = os.path.relpath(path, root)
        stat = os.stat(path)
        items.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")
payload = "\n".join(sorted(items)).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
)"

  previous_state=""
  if [[ -f "$STATE_FILE" ]]; then
    previous_state="$(cat "$STATE_FILE")"
  fi

  if [[ "$current_state" != "$previous_state" ]]; then
    echo "Knowledge base changed. Running ingestion..."
    python "$ROOT_DIR/ingest.py"
    mkdir -p "$STATE_DIR"
    printf "%s" "$current_state" > "$STATE_FILE"
  else
    echo "Knowledge base unchanged. Skipping ingestion."
  fi
}

run_ingest_if_needed

BACKEND_CMD="cd $ROOT_DIR && $ROOT_DIR/myenv/bin/uvicorn api:app --host 127.0.0.1 --port 8001"
UI_CMD="cd $ROOT_DIR && $ROOT_DIR/myenv/bin/streamlit run chat_ui.py --server.port 8501"

if [[ "$OSTYPE" == darwin* ]]; then
  osascript <<EOF
  tell application "Terminal"
    activate
    do script "$BACKEND_CMD"
    delay 1
    do script "$UI_CMD"
  end tell
EOF
  echo "Backend: http://127.0.0.1:8001"
  echo "UI: http://localhost:8501"
else
  bash -lc "$BACKEND_CMD" &
  bash -lc "$UI_CMD" &
  echo "Backend and UI started in background."
fi

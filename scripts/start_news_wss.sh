#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f ".env.local" ]; then
  set -a
  . ".env.local"
  set +a
fi

if [ "${TS_DISABLE_OPEN_TOKEN_BRIDGES:-false}" = "1" ] || [ "${TS_DISABLE_OPEN_TOKEN_BRIDGES:-false}" = "true" ]; then
  echo "news-wss disabled by TS_DISABLE_OPEN_TOKEN_BRIDGES=${TS_DISABLE_OPEN_TOKEN_BRIDGES}"
  exit 0
fi

SERVICE_NAME="news-wss.service"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl start "$SERVICE_NAME"
  echo "news-wss managed by systemd: $SERVICE_NAME"
  systemctl --no-pager --lines=20 status "$SERVICE_NAME"
  exit $?
fi

PID_FILE="data/trading_system/news_wss.pid"
LOG_FILE="data/trading_system/logs/news_wss_$(date -u +%Y%m%dT%H%M%SZ).log"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

mkdir -p "$(dirname "$LOG_FILE")"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "news-wss already running: pid=$(cat "$PID_FILE")"
  exit 1
fi

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" nohup "$PYTHON_BIN" -m trading_system.cli news-wss >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "news-wss started: pid=$! log=$LOG_FILE python=$PYTHON_BIN"

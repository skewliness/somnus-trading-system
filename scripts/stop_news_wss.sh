#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="news-wss.service"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl stop "$SERVICE_NAME"
  echo "news-wss stopped by systemd: $SERVICE_NAME"
  rm -f "data/trading_system/news_wss.pid"
  exit 0
fi

PID_FILE="data/trading_system/news_wss.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "news-wss not running (no pid file)"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "news-wss stopped: pid=$pid"
else
  echo "news-wss process already gone: pid=$pid"
fi
rm -f "$PID_FILE"

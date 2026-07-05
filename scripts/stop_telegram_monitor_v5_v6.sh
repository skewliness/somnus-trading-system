#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="telegram-monitor-v5-v6.service"
PID_FILE="${TS_TELEGRAM_MONITOR_PID_FILE:-data/trading_system/telegram_monitor/telegram_monitor.pid}"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl stop "$SERVICE_NAME"
fi

pid=""
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  kill "$pid"
  echo "stopped pid=$pid"
else
  echo "not running"
fi

rm -f "$PID_FILE"

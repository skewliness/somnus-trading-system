#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="telegram-monitor-v5-v6.service"
STATE_DIR="${TS_TELEGRAM_MONITOR_DIR:-data/trading_system/telegram_monitor}"
LOG_DIR="${TS_TELEGRAM_MONITOR_LOG_DIR:-$STATE_DIR/logs}"
PID_FILE="${TS_TELEGRAM_MONITOR_PID_FILE:-$STATE_DIR/telegram_monitor.pid}"
LAUNCH_LOG="$LOG_DIR/telegram_monitor.launch.log"

mkdir -p "$LOG_DIR"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl start "$SERVICE_NAME"
  systemctl status "$SERVICE_NAME" --no-pager --lines=12
  exit 0
fi

existing_pid=""
if [ -f "$PID_FILE" ]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$existing_pid" ] && ps -p "$existing_pid" >/dev/null 2>&1; then
  echo "Telegram monitor already running: pid=$existing_pid"
  exit 0
fi

if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.local
  set +a
fi

nohup venv/bin/python scripts/telegram_monitor_v5_v6.py --loop >> "$LAUNCH_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "Telegram monitor started: pid=$pid"
echo "log=$LAUNCH_LOG"
echo "status=scripts/status_telegram_monitor_v5_v6.sh"

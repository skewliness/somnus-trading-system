#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="telegram-monitor-v5-v6.service"
STATE_DIR="${TS_TELEGRAM_MONITOR_DIR:-data/trading_system/telegram_monitor}"
LOG_DIR="${TS_TELEGRAM_MONITOR_LOG_DIR:-$STATE_DIR/logs}"
PID_FILE="${TS_TELEGRAM_MONITOR_PID_FILE:-$STATE_DIR/telegram_monitor.pid}"
STATE_FILE="${TS_TELEGRAM_MONITOR_STATE:-$STATE_DIR/state.json}"
LAUNCH_LOG="$LOG_DIR/telegram_monitor.launch.log"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  echo "systemd=$SERVICE_NAME"
  systemctl status "$SERVICE_NAME" --no-pager --lines=18 || true
  echo
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "state=$STATE_FILE"
    if [ -f "$STATE_FILE" ]; then
      tail -80 "$STATE_FILE"
    fi
    exit 0
  fi
fi

pid=""
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  echo "status=running pid=$pid"
  ps -p "$pid" -o pid,etime,stat,command
else
  echo "status=stopped"
fi

echo "state=$STATE_FILE"
if [ -f "$STATE_FILE" ]; then
  tail -80 "$STATE_FILE"
fi

if [ -f "$LAUNCH_LOG" ]; then
  echo
  echo "log tail:"
  tail -60 "$LAUNCH_LOG"
fi

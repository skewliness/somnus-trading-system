#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="v5-trading.service"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl start "$SERVICE_NAME"
  echo "pipeline 24h managed by systemd: $SERVICE_NAME"
  systemctl --no-pager --lines=30 status "$SERVICE_NAME"
  exit $?
fi

LOG_DIR="${PIPELINE_LOG_DIR:-data/trading_system/logs}"
LOCK_DIR="${PIPELINE_LOCK_DIR:-data/trading_system/pipeline_loop.lock}"
PID_FILE="${PIPELINE_PID_FILE:-data/trading_system/pipeline_loop.pid}"
LAUNCH_LOG="$LOG_DIR/run_pipeline_24h.launch.log"

mkdir -p "$LOG_DIR"

existing_pid=""
if [ -f "$PID_FILE" ]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$existing_pid" ] && ps -p "$existing_pid" >/dev/null 2>&1; then
  echo "pipeline 24h 已在运行: pid=$existing_pid"
  exit 0
fi

if [ -d "$LOCK_DIR" ]; then
  echo "清理残留锁: $LOCK_DIR"
  rm -rf "$LOCK_DIR"
fi

nohup bash scripts/run_pipeline_24h.sh >> "$LAUNCH_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "pipeline 24h 已后台启动: pid=$pid"
echo "启动日志: $LAUNCH_LOG"
echo "交易日志: scripts/tail_trade_activity.sh"

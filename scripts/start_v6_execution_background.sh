#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="v6-execution.service"
STATE_DIR="${TS_V6_EXECUTION_DIR:-data/trading_system/v6_execution}"
LOG_DIR="${TS_V6_EXECUTION_LOG_DIR:-$STATE_DIR/logs}"
PID_FILE="${TS_V6_EXECUTION_PID_FILE:-$STATE_DIR/v6_execution_loop.pid}"
LAUNCH_LOG="$LOG_DIR/v6_execution.launch.log"

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
  echo "V6 execution 已在运行: pid=$existing_pid"
  exit 0
fi

nohup bash scripts/run_v6_execution_loop.sh >> "$LAUNCH_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"

echo "V6 execution 已后台启动: pid=$pid"
echo "启动日志: $LAUNCH_LOG"
echo "状态检查: scripts/status_v6_execution.sh"
echo "停止命令: scripts/stop_v6_execution.sh"

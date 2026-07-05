#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="v6-execution.service"
STATE_DIR="${TS_V6_EXECUTION_DIR:-data/trading_system/v6_execution}"
PID_FILE="${TS_V6_EXECUTION_PID_FILE:-$STATE_DIR/v6_execution_loop.pid}"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl stop "$SERVICE_NAME"
fi

pid=""
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  child_pids="$(pgrep -P "$pid" 2>/dev/null || true)"
  kill -TERM "$pid"
  if [ -n "$child_pids" ]; then
    kill -TERM $child_pids 2>/dev/null || true
  fi
  echo "已发送停止信号: pid=$pid"
else
  echo "没有发现运行中的 V6 execution 进程"
fi

sleep 1

if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  child_pids="$(pgrep -P "$pid" 2>/dev/null || true)"
  if [ -n "$child_pids" ]; then
    kill -KILL $child_pids 2>/dev/null || true
  fi
  kill -KILL "$pid" 2>/dev/null || true
  sleep 1
fi

if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  echo "进程仍在运行，可稍后再查: scripts/status_v6_execution.sh"
else
  rm -f "$PID_FILE"
  echo "V6 execution 已停止"
fi

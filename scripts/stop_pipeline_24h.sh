#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="v5-trading.service"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  sudo systemctl stop "$SERVICE_NAME"
  echo "pipeline 24h stopped by systemd: $SERVICE_NAME"
  rm -f "data/trading_system/pipeline_loop.pid"
  rm -rf "data/trading_system/pipeline_loop.lock"
  exit 0
fi

LOCK_DIR="${PIPELINE_LOCK_DIR:-data/trading_system/pipeline_loop.lock}"
PID_FILE="${PIPELINE_PID_FILE:-data/trading_system/pipeline_loop.pid}"

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
  echo "没有发现运行中的 pipeline 24h 进程"
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
  echo "进程仍在运行，可稍后再查: scripts/status_pipeline_24h.sh"
else
  rm -f "$PID_FILE"
  rm -rf "$LOCK_DIR"
  echo "已停止并清理锁"
fi

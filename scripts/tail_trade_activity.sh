#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

LOG_DIR="${PIPELINE_LOG_DIR:-data/trading_system/logs}"
TAIL_LINES="${TAIL_TRADE_ACTIVITY_LINES:-80}"

tail_pid=""

cleanup() {
  if [ -n "$tail_pid" ]; then
    kill "$tail_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

current_log=""
while true; do
  latest_log="$(ls -t "$LOG_DIR"/trade_activity_*.log 2>/dev/null | head -1 || true)"
  if [ -z "$latest_log" ]; then
    echo "等待交易操作日志生成: $LOG_DIR/trade_activity_*.log"
    sleep 2
    continue
  fi

  if [ "$latest_log" != "$current_log" ]; then
    if [ -n "$tail_pid" ]; then
      kill "$tail_pid" 2>/dev/null || true
      wait "$tail_pid" 2>/dev/null || true
    fi
    current_log="$latest_log"
    echo "正在跟踪最新交易操作日志: $current_log"
    tail -n "$TAIL_LINES" -f "$current_log" &
    tail_pid=$!
  fi

  if [ -n "$tail_pid" ] && ! kill -0 "$tail_pid" 2>/dev/null; then
    wait "$tail_pid" 2>/dev/null || true
    tail_pid=""
    current_log=""
  fi
  sleep 2
done

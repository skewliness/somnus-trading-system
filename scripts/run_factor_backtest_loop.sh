#!/usr/bin/env bash
# Factor backtest loop: runs factor-backtest every 30 minutes in background.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f ".env.local" ]; then
  set -a
  . ".env.local"
  set +a
fi

LOG_DIR="${PIPELINE_LOG_DIR:-data/trading_system/logs}"
PID_FILE="data/trading_system/factor_backtest_loop.pid"
LOCK_FILE="data/trading_system/factor_backtest_loop.lock"
LOG_FILE="$LOG_DIR/factor_backtest_loop.log"
INTERVAL_SECONDS=1800

mkdir -p "$LOG_DIR"

if [ -f "$LOCK_FILE" ]; then
  existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$existing_pid" ] && ps -p "$existing_pid" >/dev/null 2>&1; then
    echo "factor-backtest loop already running: pid=$existing_pid"
    exit 0
  fi
  rm -f "$LOCK_FILE"
fi

echo "$$" > "$PID_FILE"
echo "$$" > "$LOCK_FILE"

cleanup() {
  rm -f "$LOCK_FILE"
  if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) factor-backtest loop started: pid=$$" >> "$LOG_FILE"

while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) running factor-backtest" >> "$LOG_FILE"
  python3 -m trading_system.cli factor-backtest --min-samples 3 >> "$LOG_FILE" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) factor-backtest done, sleeping ${INTERVAL_SECONDS}s" >> "$LOG_FILE"
  sleep "$INTERVAL_SECONDS"
done

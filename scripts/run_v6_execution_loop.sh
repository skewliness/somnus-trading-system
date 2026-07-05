#!/usr/bin/env bash
set -u
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.local
  set +a
fi

INTERVAL_SECONDS="${TS_V6_EXECUTION_INTERVAL_SECONDS:-60}"
SYMBOLS="${TS_V6_SYMBOLS:-BTCUSDT,ETHUSDT}"
STATE_DIR="${TS_V6_EXECUTION_DIR:-data/trading_system/v6_execution}"
LOG_DIR="${TS_V6_EXECUTION_LOG_DIR:-$STATE_DIR/logs}"
PID_FILE="${TS_V6_EXECUTION_PID_FILE:-$STATE_DIR/v6_execution_loop.pid}"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"

cleanup() {
  if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null || true)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
}

terminate() {
  cleanup
  exit 0
}

trap cleanup EXIT
trap terminate INT TERM

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 execution loop started: symbols=$SYMBOLS interval=${INTERVAL_SECONDS}s"

while true; do
  tick_started_epoch="$(date +%s)"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$started_at] V6 execution tick start"

  if venv/bin/python scripts/run_v6_execution.py --symbols "$SYMBOLS" --submit; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 execution tick ok"
  else
    rc=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 execution tick failed: rc=$rc"
  fi

  if venv/bin/python scripts/run_v6_post_trade_reviews.py; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 post-trade review tick ok"
  else
    rc=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 post-trade review tick failed: rc=$rc"
  fi

  elapsed_seconds="$(($(date +%s) - tick_started_epoch))"
  sleep_seconds="$((INTERVAL_SECONDS - elapsed_seconds))"
  if [ "$sleep_seconds" -lt 0 ]; then
    sleep_seconds=0
  fi
  sleep "$sleep_seconds"
done

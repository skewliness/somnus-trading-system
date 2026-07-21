#!/usr/bin/env bash
set -u
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ "${TS_V6_SKIP_ENV_LOCAL_LOAD:-0}" != "1" ] && [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env.local
  set +a
fi

INTERVAL_SECONDS="${TS_V6_LEDGER_RECONCILE_INTERVAL_SECONDS:-60}"
INITIAL_DELAY_SECONDS="${TS_V6_LEDGER_RECONCILE_INITIAL_DELAY_SECONDS:-0}"
INSTANCE_NAME="${TS_V6_INSTANCE_NAME:-demo}"
LOOP_ENABLED="${TS_V6_LEDGER_RECONCILE_LOOP_ENABLED:-1}"
STATE_DIR="${TS_V6_EXECUTION_DIR:-data/trading_system/v6_execution}"
LOG_DIR="${TS_V6_LEDGER_RECONCILE_LOG_DIR:-$STATE_DIR/logs}"
PID_FILE="${TS_V6_LEDGER_RECONCILE_PID_FILE:-$STATE_DIR/v6_ledger_reconcile_loop.pid}"

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

if [ "$LOOP_ENABLED" != "1" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 ledger reconcile loop disabled: TS_V6_LEDGER_RECONCILE_LOOP_ENABLED=$LOOP_ENABLED"
  exit 0
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 ledger reconcile loop started: instance=$INSTANCE_NAME mode=${TS_V6_ORDER_MODE:-dry_run} interval=${INTERVAL_SECONDS}s"

if [ "$INITIAL_DELAY_SECONDS" -gt 0 ] 2>/dev/null; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 ledger reconcile initial delay: instance=$INSTANCE_NAME delay=${INITIAL_DELAY_SECONDS}s"
  sleep "$INITIAL_DELAY_SECONDS"
fi

while true; do
  tick_started_epoch="$(date +%s)"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$started_at] V6 ledger reconcile tick start: instance=$INSTANCE_NAME"

  if venv/bin/python scripts/run_v6_ledger_reconcile.py; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 ledger reconcile tick ok"
  else
    rc=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] V6 ledger reconcile tick failed: rc=$rc"
  fi

  elapsed_seconds="$(($(date +%s) - tick_started_epoch))"
  sleep_seconds="$((INTERVAL_SECONDS - elapsed_seconds))"
  if [ "$sleep_seconds" -lt 0 ]; then
    sleep_seconds=0
  fi
  sleep "$sleep_seconds"
done

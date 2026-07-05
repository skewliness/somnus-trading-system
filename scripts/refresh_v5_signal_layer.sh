#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env.local"
  set +a
fi

export PIPELINE_EXECUTE_REQUESTS=0
export REFRESH_OKX_INPUTS=1
export TS_DISABLE_OPEN_TOKEN_BRIDGES=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [ -z "${SSL_CERT_FILE:-}" ]; then
  CERTIFI_CA="$(venv/bin/python - <<'PY'
try:
    import certifi
    print(certifi.where())
except Exception:
    pass
PY
)"
  if [ -n "$CERTIFI_CA" ]; then
    export SSL_CERT_FILE="$CERTIFI_CA"
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERTIFI_CA}"
  fi
fi
LOG_DIR="${V5_SIGNAL_REFRESH_LOG_DIR:-data/trading_system/logs}"
LOCK_FILE="${V5_SIGNAL_REFRESH_LOCK_FILE:-data/trading_system/v5_signal_refresh.flock}"
LOCK_DIR="${V5_SIGNAL_REFRESH_LOCK_DIR:-data/trading_system/v5_signal_refresh.lockdir}"
LOG_FILE="$LOG_DIR/v5_signal_refresh_$(date -u +%Y%m%dT%H%M%SZ).log"
RUN_PIPELINE="${V5_SIGNAL_REFRESH_RUN_PIPELINE:-0}"

mkdir -p "$LOG_DIR"
LOCK_ACQUIRED_BY_DIR=0
cleanup_lock() {
  if [ "${LOCK_ACQUIRED_BY_DIR:-0}" = "1" ]; then
    rm -rf "$LOCK_DIR"
  fi
}
trap cleanup_lock EXIT
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "v5 signal refresh already running; skip $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_FILE"
    exit 0
  fi
else
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "v5 signal refresh already running; skip $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_FILE"
    exit 0
  fi
  LOCK_ACQUIRED_BY_DIR=1
  printf '%s
' "$$" > "$LOCK_FILE"
fi

{
  echo "### v5 signal refresh start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT_DIR"
  echo "execute_requests=$PIPELINE_EXECUTE_REQUESTS"
  echo "refresh_okx_inputs=$REFRESH_OKX_INPUTS"
  echo "run_pipeline=$RUN_PIPELINE"
  echo
  echo "### refresh Binance AI Select final filter"
  venv/bin/python scripts/refresh_binance_ai_select.py || echo "refresh_binance_ai_select failed; continue with existing bridge file"
  echo
  echo "### refresh Binance smart-signal bridge"
  venv/bin/python scripts/refresh_binance_smart_signal.py || echo "refresh_binance_smart_signal failed; continue with existing bridge file"
  echo
  echo "### refresh OKX bridge inputs"
  venv/bin/python tools/refresh_okx_parallel_inputs.py || echo "refresh_okx_parallel_inputs failed; continue with existing bridge files"
  echo
  if [ "$RUN_PIPELINE" = "1" ] || [ "$RUN_PIPELINE" = "true" ]; then
    if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet v5-trading.service; then
      echo "### skip full V5 pipeline refresh: v5-trading.service is active and owns pipeline cadence"
    else
      echo "### run V5 pipeline signal refresh"
      venv/bin/python -m trading_system.cli pipeline
    fi
  else
    echo "### skip full V5 pipeline refresh: V5_SIGNAL_REFRESH_RUN_PIPELINE=$RUN_PIPELINE"
  fi
  echo
  echo "### v5 signal refresh done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG_FILE" 2>&1

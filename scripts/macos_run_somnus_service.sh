#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/ma7/binance-trading-skill-system"
cd "$ROOT"
export PATH="$ROOT/venv/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export VIRTUAL_ENV="$ROOT/venv"
export PYTHONPATH="$ROOT"
export SSL_CERT_FILE="$ROOT/venv/lib/python3.12/site-packages/certifi/cacert.pem"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7897}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

load_env_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$f"
    set +a
  fi
}

load_env_file "$ROOT/.env.local"

case "${1:-}" in
  telegram-v5)
    load_env_file "$ROOT/.env.telegram_v5"
    export TS_TELEGRAM_MONITOR_DISABLE_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_INTEL_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_SUMMARY=1
    export TS_TELEGRAM_MONITOR_DISABLE_MACRO_REPORT=1
    export TS_TELEGRAM_MONITOR_DISABLE_DAILY_NEWS=1
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/telegram_monitor_v5_v6.py" --loop --scope v5 --state-file data/trading_system/telegram_monitor/state_v5.json
    ;;
  telegram-v6)
    export TS_TELEGRAM_MONITOR_DISABLE_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_INTEL_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_SUMMARY=1
    export TS_TELEGRAM_MONITOR_DISABLE_MACRO_REPORT=1
    export TS_TELEGRAM_MONITOR_DISABLE_DAILY_NEWS=1
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/telegram_monitor_v5_v6.py" --loop --scope v6 --state-file data/trading_system/telegram_monitor/state_v6.json
    ;;
  v5-trading)
    export PIPELINE_INTERVAL_SECONDS="${PIPELINE_INTERVAL_SECONDS:-180}"
    exec /bin/bash "$ROOT/scripts/run_pipeline_24h.sh"
    ;;
  v5-position-management)
    export TS_V5_POSITION_MANAGEMENT_INTERVAL_SECONDS=5
    export TS_V5_POSITION_MANAGEMENT_MIN_REST_SECONDS=1
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/run_v5_position_management_loop.py" --interval-seconds 5 --min-rest-seconds 1 --submit
    ;;
  v5-position-reconcile)
    export TS_V5_POSITION_RECONCILE_INTERVAL_SECONDS=1
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/run_v5_position_reconcile_loop.py" --interval-seconds 1
    ;;
  v5-signal-refresh)
    exec /bin/bash "$ROOT/scripts/refresh_v5_signal_layer.sh"
    ;;
  v6-execution)
    exec /bin/bash "$ROOT/scripts/run_v6_execution_loop.sh"
    ;;
  v6-ledger-reconcile)
    exec /bin/bash "$ROOT/scripts/run_v6_ledger_reconcile_loop.sh"
    ;;
  v6-distill)
    export ROOT_DIR="$ROOT"
    export RUN_BINANCE_CAPTURE=1
    export BINANCE_CAPTURE_MODE=api
    export BINANCE_CURL_FILE=data/trading_system/binance_curl.txt
    export RUN_BITGET_CAPTURE=0
    export RUN_BITGET_IMPORT=0
    export RUN_BUILD_DATASET=1
    export RUN_TRAIN=1
    export INTERVAL_SECONDS="${INTERVAL_SECONDS:-43200}"
    exec /bin/bash "$ROOT/scripts/run_v6_continuous_distill.sh" --loop
    ;;
  xuejian-push)
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/xuejian_proactive_push.py" --loop --interval-seconds 60
    ;;
  news-wss)
    export TS_DISABLE_OPEN_TOKEN_BRIDGES="${TS_DISABLE_OPEN_TOKEN_BRIDGES:-0}"
    exec "$ROOT/venv/bin/python" -m trading_system.cli news-wss
    ;;
  v6-jsonl-rotate)
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/rotate_v6_jsonl_logs.py"
    ;;
  kline-cache-maintenance)
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/maintain_kline_caches.py"
    ;;
  harvest-shadow)
    exec "$ROOT/venv/bin/python" "$ROOT/harvest_shadow_scores.py"
    ;;
  v6-distill-monitor)
    exec /bin/bash "$ROOT/scripts/v6_distill_monitor.sh"
    ;;
  expand-smart-money-watchlist)
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/expand_smart_money_watchlist.py" --output-file data/trading_system/model_training/whale_scan/whales_multi_range_latest.json --time-ranges ALL,90D,30D,7D --min-roi 3.0 --min-win-rate 0.50
    ;;
  *)
    echo "unknown service: ${1:-}" >&2
    exit 64
    ;;
esac



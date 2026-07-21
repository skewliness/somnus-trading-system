#!/usr/bin/env bash
set -euo pipefail
ROOT="${SOMNUS_ROOT:-/Users/ma7/binance-trading-skill-system}"
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

load_env_file "${SOMNUS_ENV_FILE:-$ROOT/.env.local}"

configure_v6_sizing() {
  # All V6 accounts use a fixed 1000 USDT initial notional at 8x. Symbol-level
  # multipliers are applied by the execution engine to entries and scale-ins.
  export TS_V6_BASE_NOTIONAL_USDT=1000
  export TS_V6_MIN_NOTIONAL_USDT=1000
  export TS_V6_DEFAULT_LEVERAGE=8
  export TS_V6_SYMBOL_NOTIONAL_MULTIPLIERS=ZECUSDT:0.60
}

configure_v6_policy() {
  configure_v6_sizing
  export TS_V6_ENTRY_GATE_POLICY_V2_ENABLED=1
  export TS_V6_MIXED_STRUCTURE_MACRO_ARBITRATION_ENABLED=1
  export TS_V6_MIXED_STRUCTURE_PROBABILITY_FALLBACK_MAX_LOSSES=3
  export TS_V6_TELEGRAM_WHALE_BRIDGE_FILE=data/trading_system/skill_bridges/telegram_whale_crypto.json
  export TS_V6_TELEGRAM_WHALE_FALLBACK_ENABLED=1
  export TS_V6_TELEGRAM_WHALE_FALLBACK_MAX_BRIDGE_AGE_SECONDS=180
  export TS_V6_TELEGRAM_WHALE_FALLBACK_MAX_EVENT_AGE_MINUTES=360
  export TS_V6_TELEGRAM_WHALE_FALLBACK_MIN_NOTIONAL_USDT=250000
  export TS_V6_TELEGRAM_WHALE_FALLBACK_MIN_ABS_SCORE=0.75
  export TS_V6_TELEGRAM_WHALE_FALLBACK_MAX_MODEL_GAP=0.015
  export TS_V6_ENTRY_GATE_EVENT_RECORDING_ENABLED=1
  export TS_V6_ENTRY_GATE_MIN_NOTIONAL_MULTIPLIER=1.00
  export TS_V6_ENTRY_GATE_PROBE_STOP_LOSS_PCT=0.015
  # Record Gann structure in every signal, but keep sizing disabled until a
  # causal next-bar A/B run improves PF, drawdown and worst trade together.
  export TS_V6_GANN_POSITION_SIZING_ENABLED=0
  export TS_V6_GANN_POSITION_SIZING_SYMBOLS=BTCUSDT
}

configure_v6_instance() {
  local instance="$1"
  local state_dir

  case "$instance" in
    demo)
      state_dir="data/trading_system/v6_execution"
      export TS_V6_INSTANCE_NAME=demo
      export TS_V6_ORDER_MODE=testnet_order
      export TS_V6_EXECUTION_INITIAL_DELAY_SECONDS=0
      export TS_V6_LEDGER_RECONCILE_INITIAL_DELAY_SECONDS=0
      unset TS_V6_LIVE_TRADING_CONFIRMED
      ;;
    live)
      load_env_file "${SOMNUS_V6_LIVE_ENV_FILE:-$ROOT/.env.v6_live}"
      state_dir="data/trading_system/v6_execution_live"
      export TS_V6_INSTANCE_NAME=live
      export TS_V6_ORDER_MODE=live_order
      # Let Demo refresh the shared public K-line cache first. Live still
      # refreshes it itself if Demo is unavailable or the cache is stale.
      export TS_V6_EXECUTION_INITIAL_DELAY_SECONDS="${TS_V6_LIVE_INITIAL_DELAY_SECONDS:-60}"
      export TS_V6_LEDGER_RECONCILE_INITIAL_DELAY_SECONDS="${TS_V6_LIVE_LEDGER_INITIAL_DELAY_SECONDS:-0}"
      ;;
    live-momo2)
      load_env_file "${SOMNUS_V6_LIVE_MOMO2_ENV_FILE:-$ROOT/.env.v6_live_momo2}"
      state_dir="data/trading_system/v6_execution_live_momo2"
      export TS_V6_INSTANCE_NAME=momo2
      export TS_V6_ORDER_MODE=live_order
      export TS_V6_EXIT_CONTROL_MODE=stop_only_manual_profit
      export TS_V6_MANUAL_EXCHANGE_ACTION_AUDIT_ENABLED=1
      export TS_V6_SUBMIT_PROTECTIVE_STOP=1
      # Keep closePosition protective stops aligned after manual partial exits.
      export TS_V6_RECONCILE_EVERY_RUN=1
      export TS_V6_EXECUTION_INITIAL_DELAY_SECONDS="${TS_V6_LIVE_MOMO2_INITIAL_DELAY_SECONDS:-120}"
      export TS_V6_LEDGER_RECONCILE_INITIAL_DELAY_SECONDS="${TS_V6_LIVE_MOMO2_LEDGER_INITIAL_DELAY_SECONDS:-0}"
      ;;
    *)
      echo "unknown V6 instance: $instance" >&2
      return 64
      ;;
  esac

  # Re-apply after account-specific env loading so execution, reconciliation,
  # readiness checks and Telegram readers share one sizing contract.
  configure_v6_sizing

  # Every mutable execution artifact is instance-local. Public market-data
  # caches and the model remain shared because they contain no account state.
  export TS_V6_EXECUTION_DIR="$state_dir"
  export TS_V6_STATE_FILE="$state_dir/state.json"
  export TS_V6_SIGNALS_FILE="$state_dir/signals.jsonl"
  export TS_V6_ENTRY_GATE_EVENTS_FILE="$state_dir/entry_gate_events.jsonl"
  export TS_V6_ORDERS_FILE="$state_dir/orders.jsonl"
  export TS_V6_EXECUTION_WAL_FILE="$state_dir/execution_wal.jsonl"
  export TS_V6_LATEST_SUMMARY_FILE="$state_dir/latest_run_summary.json"
  export TS_V6_CLOSED_TRADE_REVIEW_QUEUE_FILE="$state_dir/closed_trade_review_queue.jsonl"
  export TS_V6_POST_TRADE_TRAINING_SAMPLES_FILE="$state_dir/post_trade_training_samples.jsonl"
  export TS_V6_MANUAL_EXCHANGE_ACTION_AUDIT_FILE="$state_dir/manual_exchange_actions.jsonl"
  export TS_V6_EXECUTION_LOCK_TARGET="$state_dir/v6_execution"
  export TS_V6_EXECUTION_LOG_DIR="$state_dir/logs"
  export TS_V6_EXECUTION_PID_FILE="$state_dir/v6_execution_loop.pid"
  export TS_V6_LEDGER_RECONCILE_REPORT_FILE="$state_dir/ledger_reconcile_latest.json"
  export TS_V6_LEDGER_RECONCILE_HISTORY_FILE="$state_dir/ledger_reconcile.jsonl"
  export TS_V6_LEDGER_RECONCILE_LOG_DIR="$state_dir/logs"
  export TS_V6_LEDGER_RECONCILE_PID_FILE="$state_dir/v6_ledger_reconcile_loop.pid"
  export TS_V6_SKIP_ENV_LOCAL_LOAD=1
}

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
  telegram-v6-live)
    configure_v6_instance live
    unset TS_V6_LIVE_TRADING_CONFIRMED
    export TS_TELEGRAM_V6_PID_FILE=data/trading_system/v6_execution_live/v6_execution_loop.pid
    export TS_TELEGRAM_V6_SUMMARY_FILE=data/trading_system/v6_execution_live/latest_run_summary.json
    export TS_TELEGRAM_V6_DISPLAY_NAME="V6 LIVE"
    export TS_TELEGRAM_V6_SKIP_DISTILL_ALERTS=1
    export TS_TELEGRAM_MONITOR_DISABLE_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_INTEL_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_SUMMARY=1
    export TS_TELEGRAM_MONITOR_DISABLE_MACRO_REPORT=1
    export TS_TELEGRAM_MONITOR_DISABLE_DAILY_NEWS=1
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/telegram_monitor_v5_v6.py" --loop --scope v6 --state-file data/trading_system/telegram_monitor/state_v6_live.json
    ;;
  telegram-v6-live-momo2)
    configure_v6_instance live-momo2
    unset TS_V6_LIVE_TRADING_CONFIRMED
    export TS_TELEGRAM_V6_PID_FILE=data/trading_system/v6_execution_live_momo2/v6_execution_loop.pid
    export TS_TELEGRAM_V6_SUMMARY_FILE=data/trading_system/v6_execution_live_momo2/latest_run_summary.json
    export TS_TELEGRAM_V6_DISPLAY_NAME="V6 LIVE momo2"
    export TS_TELEGRAM_V6_MANUAL_TAKE_PROFIT=1
    export TS_TELEGRAM_V6_SKIP_DISTILL_ALERTS=1
    export TS_TELEGRAM_MONITOR_DISABLE_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_INTEL_COMMANDS=1
    export TS_TELEGRAM_MONITOR_DISABLE_SUMMARY=1
    export TS_TELEGRAM_MONITOR_DISABLE_MACRO_REPORT=1
    export TS_TELEGRAM_MONITOR_DISABLE_DAILY_NEWS=1
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/telegram_monitor_v5_v6.py" --loop --scope v6 --state-file data/trading_system/telegram_monitor/state_v6_live_momo2.json
    ;;
  v5-trading)
    export PIPELINE_INTERVAL_SECONDS="${PIPELINE_INTERVAL_SECONDS:-180}"
    export TS_V6_GANN_POSITION_SIZING_ENABLED=0
    exec /bin/bash "$ROOT/scripts/run_pipeline_24h.sh"
    ;;
  v5-position-management)
    export TS_V5_POSITION_MANAGEMENT_INTERVAL_SECONDS=5
    export TS_V5_POSITION_MANAGEMENT_MIN_REST_SECONDS=1
    export TS_V6_GANN_POSITION_SIZING_ENABLED=0
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
    configure_v6_policy
    configure_v6_instance demo
    exec /bin/bash "$ROOT/scripts/run_v6_execution_loop.sh"
    ;;
  v6-execution-live)
    configure_v6_policy
    configure_v6_instance live
    unset TS_V6_LIVE_TRADING_CONFIRMED
    "$ROOT/venv/bin/python" "$ROOT/scripts/check_v6_live_readiness.py"
    export TS_V6_LIVE_TRADING_CONFIRMED=YES_I_UNDERSTAND_REAL_MONEY_RISK
    exec /bin/bash "$ROOT/scripts/run_v6_execution_loop.sh"
    ;;
  v6-execution-live-momo2)
    configure_v6_policy
    configure_v6_instance live-momo2
    unset TS_V6_LIVE_TRADING_CONFIRMED
    "$ROOT/venv/bin/python" "$ROOT/scripts/check_v6_live_readiness.py"
    export TS_V6_LIVE_TRADING_CONFIRMED=YES_I_UNDERSTAND_REAL_MONEY_RISK
    exec /bin/bash "$ROOT/scripts/run_v6_execution_loop.sh"
    ;;
  v6-live-readiness)
    configure_v6_policy
    configure_v6_instance live
    unset TS_V6_LIVE_TRADING_CONFIRMED
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/check_v6_live_readiness.py"
    ;;
  v6-live-readiness-momo2)
    configure_v6_policy
    configure_v6_instance live-momo2
    unset TS_V6_LIVE_TRADING_CONFIRMED
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/check_v6_live_readiness.py"
    ;;
  v6-entry-gate-labeler)
    exec "$ROOT/venv/bin/python" "$ROOT/scripts/label_v6_entry_gate_events.py"
    ;;
  v6-ledger-reconcile)
    configure_v6_instance demo
    exec /bin/bash "$ROOT/scripts/run_v6_ledger_reconcile_loop.sh"
    ;;
  v6-ledger-reconcile-live)
    configure_v6_instance live
    unset TS_V6_LIVE_TRADING_CONFIRMED
    exec /bin/bash "$ROOT/scripts/run_v6_ledger_reconcile_loop.sh"
    ;;
  v6-ledger-reconcile-live-momo2)
    configure_v6_instance live-momo2
    unset TS_V6_LIVE_TRADING_CONFIRMED
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
  telegram-whale)
    export TS_TELEGRAM_WHALE_CHANNEL="${TS_TELEGRAM_WHALE_CHANNEL:-hyperdashflows}"
    export TS_TELEGRAM_WHALE_POLL_SECONDS="${TS_TELEGRAM_WHALE_POLL_SECONDS:-30}"
    exec "$ROOT/venv/bin/python" -m trading_system.telegram_whale_listener
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

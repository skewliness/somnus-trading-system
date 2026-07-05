#!/usr/bin/env bash
set -u
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env.local"
  set +a
fi

INTERVAL_SECONDS="${PIPELINE_INTERVAL_SECONDS:-180}"
MAX_SECONDS="${PIPELINE_MAX_SECONDS:-86400}"
LOG_DIR="${PIPELINE_LOG_DIR:-data/trading_system/logs}"
LOCK_DIR="${PIPELINE_LOCK_DIR:-data/trading_system/pipeline_loop.lock}"
LOCK_FILE="${PIPELINE_LOCK_FILE:-data/trading_system/pipeline_loop.flock}"
PID_FILE="${PIPELINE_PID_FILE:-data/trading_system/pipeline_loop.pid}"
LOG_FILE="$LOG_DIR/pipeline_24h_$(date -u +%Y%m%dT%H%M%SZ).log"
TRADE_LOG_FILE="${TRADE_ACTIVITY_LOG_FILE:-$LOG_DIR/trade_activity_$(date -u +%Y%m%dT%H%M%SZ).log}"
EXECUTE_REQUESTS="${PIPELINE_EXECUTE_REQUESTS:-0}"
REFRESH_OKX_INPUTS="${REFRESH_OKX_INPUTS:-1}"
USE_V6_EXECUTION_OVERLAY="${TS_V5_USE_V6_EXECUTION_OVERLAY:-1}"
RUN_TIMEOUT_SECONDS="${PIPELINE_RUN_TIMEOUT_SECONDS:-900}"
EXECUTION_TIMEOUT_SECONDS="${PIPELINE_EXECUTION_TIMEOUT_SECONDS:-300}"
V5_EXECUTION_STATE_DIR="${TS_V5_EXECUTION_STATE_DIR:-data/trading_system/v5_v6_execution}"
V5_MAIN_EXECUTION_FLAG="${TS_V5_MAIN_EXECUTION_FLAG:-$V5_EXECUTION_STATE_DIR/v5_main_execution_requested.flag}"
V5_MAIN_EXECUTION_LOCK_TIMEOUT_SECONDS="${TS_V5_MAIN_EXECUTION_LOCK_TIMEOUT_SECONDS:-600}"

mkdir -p "$LOG_DIR"

if [ -d "$LOCK_DIR" ]; then
  echo "Removing legacy pipeline directory lock: $LOCK_DIR" >&2
  rm -rf "$LOCK_DIR"
fi

LOCK_ACQUIRED_BY_DIR=0
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another pipeline loop appears to be running: $LOCK_FILE" >&2
    exit 1
  fi
  : > "$LOCK_FILE"
  printf '%s
' "$$" 1>&9
else
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another pipeline loop appears to be running: $LOCK_DIR" >&2
    exit 1
  fi
  LOCK_ACQUIRED_BY_DIR=1
  printf '%s
' "$$" > "$LOCK_FILE"
fi

cleanup() {
  if [ -f "$PID_FILE" ] && [ "$(cat "$PID_FILE" 2>/dev/null)" = "$$" ]; then
    rm -f "$PID_FILE"
  fi
  rm -f "$V5_MAIN_EXECUTION_FLAG" 2>/dev/null || true
  if [ "${LOCK_ACQUIRED_BY_DIR:-0}" = "1" ]; then
    rm -rf "$LOCK_DIR"
  fi
}

stop_loop() {
  # Recursively kill active pipeline child process tree if running
  if [ -n "${active_pipeline_pid:-}" ] && kill -0 "$active_pipeline_pid" 2>/dev/null; then
    kill_descendants "$active_pipeline_pid"
    sleep 2
    if kill -0 "$active_pipeline_pid" 2>/dev/null; then
      kill_descendants_force "$active_pipeline_pid"
    fi
  fi
  cleanup
  exit 130
}

kill_descendants() {
  local pid="$1"
  local children
  children="$(pgrep -P "$pid" 2>/dev/null || true)"
  for child in $children; do
    kill_descendants "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
}

kill_descendants_force() {
  local pid="$1"
  local children
  children="$(pgrep -P "$pid" 2>/dev/null || true)"
  for child in $children; do
    kill_descendants_force "$child"
  done
  kill -9 "$pid" 2>/dev/null || true
}

active_pipeline_pid=""

print_order_status() {
  if [ "${USE_V6_EXECUTION_OVERLAY:-0}" = "1" ] || [ "${USE_V6_EXECUTION_OVERLAY:-0}" = "true" ]; then
    python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/trading_system/v5_v6_execution/latest_v5_v6_execution_report.json")
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("本轮V5-V6执行状态: 无法读取覆盖执行报告")
    raise SystemExit(0)

intent_count = int(report.get("intent_count") or 0)
result_count = int(report.get("result_count") or 0)
results = [item for item in report.get("results") or [] if isinstance(item, dict)]
silent_statuses = {
    "reconciled_missing_exchange_position",
    "submitted_no_position_delta",
    "status_unknown_no_position",
    "skipped_protective_stop_immediate_trigger",
    "skipped_percent_price_retry_deviation",
    "skipped_percent_price_retry_counterparty_filter",
    "skipped_percent_price_retry_failed",
    "deferred_percent_price_close_deviation",
    "deferred_percent_price_close_counterparty_filter",
    "deferred_percent_price_close_failed",
}
def is_silent_result(item):
    status = str(item.get("status") or "")
    text = str(item.get("error") or item.get("original_error") or item.get("retry_error") or "").lower()
    return (
        status in silent_statuses
        or '"code":-1121' in text
        or '"code": -1121' in text
        or "invalid symbol" in text
        or item.get("telegram_silent") is True
        or item.get("maintenance_only") is True
    )
visible_results = [item for item in results if not is_silent_result(item)]
failed = sum(1 for item in visible_results if str(item.get("status") or "") == "failed")
submitted = sum(1 for item in visible_results if str(item.get("status") or "") in {"submitted", "submitted_unprotected"})
positions = report.get("positions") or []
if intent_count == 0 and result_count == 0:
    print(f"本轮V5-V6执行: 无新订单；V5-V6持仓={len(positions)}。")
elif not visible_results:
    print(f"本轮V5-V6执行: 无新有效订单；V5-V6持仓={len(positions)}。")
else:
    parts = []
    for item in visible_results:
        symbol = str(item.get("symbol") or "--")
        action = str(item.get("action") or "--")
        status = str(item.get("status") or "unknown")
        parts.append(f"{symbol} {action} {status}")
    detail = "；".join(parts[:5])
    print(f"本轮V5-V6执行: 意图={intent_count} 结果={result_count} 已提交={submitted} 失败={failed}；{detail}；V5-V6持仓={len(positions)}。")
PY
    return
  fi
  python3 - <<'PY'
import json
from pathlib import Path

report_path = Path("data/trading_system/pipeline_run_report.json")
try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
except Exception:
    print("本轮币安订单状态: 无法读取 pipeline 报告")
    raise SystemExit(0)

bridge = report.get("execution_bridge_summary") if isinstance(report, dict) else {}
if not isinstance(bridge, dict):
    bridge = {}

request_file = str(bridge.get("request_file") or "")
intent_count = int(bridge.get("intent_count") or 0)
blocked_count = int(bridge.get("blocked_intent_count") or 0)
if not request_file:
    print("本轮没有需要发送到币安的订单；正在进入下一轮训练/扫描准备。")
    raise SystemExit(0)

result_path = Path("data/trading_system/execution_results") / f"{Path(request_file).stem}_result.json"
try:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
except Exception:
    print(f"本轮币安订单状态: 等待执行结果 {result_path}")
    raise SystemExit(0)

status = str(payload.get("status") or "unknown")
result_count = int(payload.get("result_count") or len(payload.get("results") or []))
if intent_count == 0 and result_count == 0:
    print("本轮没有需要发送到币安的订单；正在进入下一轮训练/扫描准备。")
elif result_count == 0:
    print(f"本轮没有实际发送到币安的订单；状态={status} 候选操作={intent_count} 拦截={blocked_count}。")
else:
    failed = sum(1 for item in payload.get("results") or [] if isinstance(item, dict) and item.get("status") == "failed")
    print(f"本轮币安执行: 状态={status} 结果={result_count} 失败={failed}；正在进入下一轮训练/扫描准备。")
PY
}

cooldown_with_countdown() {
  local remaining="$1"
  local next_run_at="$2"
  if [ "$remaining" -le 0 ]; then
    return
  fi

  local total_minutes=$(((remaining + 59) / 60))
  echo "===== 冷却${total_minutes}分钟 next=$next_run_at =====" >> "$LOG_FILE"
  echo "===== 冷却${total_minutes}分钟 next=$next_run_at =====" >> "$TRADE_LOG_FILE"

  while [ "$remaining" -gt 0 ]; do
    if [ -f "data/trading_system/news_urgent_trigger" ]; then
      rm -f "data/trading_system/news_urgent_trigger"
      echo "===== urgent news trigger detected, breaking cooldown =====" >> "$LOG_FILE"
      echo "===== 紧急新闻触发，提前开始下一轮扫描 =====" >> "$TRADE_LOG_FILE"
      break
    fi
    if [ "$remaining" -ge 60 ]; then
      sleep 60
      remaining=$((remaining - 60))
    else
      sleep "$remaining"
      remaining=0
    fi
  done
}

trap cleanup EXIT
trap stop_loop INT TERM

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"
run_count=0
echo "$$" > "$PID_FILE"

{
  echo "===== pipeline 24h loop started: $started_at ====="
  echo "root=$ROOT_DIR"
  echo "interval_seconds=$INTERVAL_SECONDS"
  echo "max_seconds=$MAX_SECONDS"
  echo "execute_requests=$EXECUTE_REQUESTS"
  echo "refresh_okx_inputs=$REFRESH_OKX_INPUTS"
  echo "v5_use_v6_execution_overlay=$USE_V6_EXECUTION_OVERLAY"
  echo "run_timeout_seconds=$RUN_TIMEOUT_SECONDS"
  echo "execution_timeout_seconds=$EXECUTION_TIMEOUT_SECONDS"
  echo "pid=$$"
  echo "pid_file=$PID_FILE"
  echo "log_file=$LOG_FILE"
  echo "trade_activity_log=$TRADE_LOG_FILE"
} >> "$LOG_FILE"

{
  echo "===== trade activity log started: $started_at ====="
  echo "root=$ROOT_DIR"
  echo "pipeline_log=$LOG_FILE"
} >> "$TRADE_LOG_FILE"

echo "V4 pipeline started: pid=$$ interval=${INTERVAL_SECONDS}s log=$LOG_FILE"

while true; do
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - started_epoch))
  if [ "$MAX_SECONDS" -gt 0 ] && [ "$elapsed" -ge "$MAX_SECONDS" ]; then
    echo "===== max runtime reached: elapsed=${elapsed}s =====" >> "$LOG_FILE"
    break
  fi

  run_count=$((run_count + 1))
  run_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "" >> "$LOG_FILE"
  echo "===== run #$run_count started: $run_started elapsed=${elapsed}s =====" >> "$LOG_FILE"
  {
    echo ""
    echo "===== run #$run_count started: $run_started elapsed=${elapsed}s ====="
  } >> "$TRADE_LOG_FILE"
  echo "--- run #$run_count $run_started ---"

  if [ "$REFRESH_OKX_INPUTS" = "1" ] || [ "$REFRESH_OKX_INPUTS" = "true" ]; then
    echo "===== run #$run_count refresh OKX bridge inputs =====" >> "$LOG_FILE"
    python3 tools/refresh_okx_parallel_inputs.py >> "$LOG_FILE" 2>&1 || {
      echo "refresh_okx_parallel_inputs exited non-zero; continuing with existing bridge files" >> "$LOG_FILE"
    }
  fi

  status_file="$LOG_DIR/.pipeline_status_${$}_${run_count}"
  (
    python3 -m trading_system.cli pipeline >> "$LOG_FILE" 2>&1
    echo "${PIPESTATUS[0]}" > "$status_file"
  ) &
  pipeline_pid=$!
  active_pipeline_pid=$pipeline_pid
  heartbeat_seconds=0
  while kill -0 "$pipeline_pid" 2>/dev/null; do
    sleep 60
    heartbeat_seconds=$((heartbeat_seconds + 60))
    if kill -0 "$pipeline_pid" 2>/dev/null; then
      heartbeat_minutes=$(((heartbeat_seconds + 59) / 60))
      echo "===== run #$run_count 扫描/训练中：已运行约${heartbeat_minutes}分钟 =====" >> "$TRADE_LOG_FILE"
      if [ "${RUN_TIMEOUT_SECONDS:-0}" -gt 0 ] && [ "$heartbeat_seconds" -ge "$RUN_TIMEOUT_SECONDS" ]; then
        echo "===== run #$run_count pipeline timeout after ${heartbeat_seconds}s; killing child tree =====" >> "$LOG_FILE"
        echo "===== run #$run_count 扫描/训练超时，终止本轮并进入下一轮 =====" >> "$TRADE_LOG_FILE"
        kill_descendants "$pipeline_pid"
        sleep 2
        if kill -0 "$pipeline_pid" 2>/dev/null; then
          kill_descendants_force "$pipeline_pid"
        fi
        echo 124 > "$status_file"
        break
      fi
    fi
  done
  wait "$pipeline_pid" 2>/dev/null || true
  active_pipeline_pid=""
  status="$(cat "$status_file" 2>/dev/null || echo 1)"
  rm -f "$status_file"

  if [ "$EXECUTE_REQUESTS" = "1" ] || [ "$EXECUTE_REQUESTS" = "true" ]; then
    if [ "$USE_V6_EXECUTION_OVERLAY" = "1" ] || [ "$USE_V6_EXECUTION_OVERLAY" = "true" ]; then
      echo "===== run #$run_count execute V5 candidates with V6 trading system using V5 key =====" >> "$LOG_FILE"
      echo "===== run #$run_count V5候选使用V6交易系统，并使用V5 key下单 =====" >> "$TRADE_LOG_FILE"
      execute_status_file="$LOG_DIR/.v5_execute_status_${$}_${run_count}"
      mkdir -p "$(dirname "$V5_MAIN_EXECUTION_FLAG")"
      printf '%s\n' "$$ $run_count $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$V5_MAIN_EXECUTION_FLAG"
      (
        python3 scripts/run_v5_v6_execution.py --submit --lock-timeout-seconds "$V5_MAIN_EXECUTION_LOCK_TIMEOUT_SECONDS" 2>&1 | tee -a "$LOG_FILE" "$TRADE_LOG_FILE"
        echo "${PIPESTATUS[0]}" > "$execute_status_file"
      ) &
      execute_pid=$!
      execute_seconds=0
      while kill -0 "$execute_pid" 2>/dev/null; do
        sleep 10
        execute_seconds=$((execute_seconds + 10))
        if [ "${EXECUTION_TIMEOUT_SECONDS:-0}" -gt 0 ] && [ "$execute_seconds" -ge "$EXECUTION_TIMEOUT_SECONDS" ]; then
          echo "===== run #$run_count execution timeout after ${execute_seconds}s; killing child tree =====" >> "$LOG_FILE"
          echo "===== run #$run_count V5执行超时，终止本轮执行段 =====" >> "$TRADE_LOG_FILE"
          kill_descendants "$execute_pid"
          sleep 2
          if kill -0 "$execute_pid" 2>/dev/null; then
            kill_descendants_force "$execute_pid"
          fi
          echo 124 > "$execute_status_file"
          break
        fi
      done
      wait "$execute_pid" 2>/dev/null || true
      rm -f "$V5_MAIN_EXECUTION_FLAG" 2>/dev/null || true
      execute_status="$(cat "$execute_status_file" 2>/dev/null || echo 1)"
      rm -f "$execute_status_file"
    else
      echo "===== run #$run_count execute pending requests =====" >> "$LOG_FILE"
      echo "===== run #$run_count 正在发送待执行订单到币安 =====" >> "$TRADE_LOG_FILE"
      python3 -m trading_system.cli execute-pending-requests >> "$LOG_FILE" 2>&1
      execute_status=${PIPESTATUS[0]}
    fi
    if [ "$execute_status" -ne 0 ]; then
      echo "execution step exited non-zero; continuing after interval" >> "$LOG_FILE"
    fi
  fi

  echo "===== run #$run_count summary =====" >> "$LOG_FILE"
  python3 scripts/summarize_pipeline_run.py --report data/trading_system/pipeline_run_report.json >> "$LOG_FILE" 2>&1

  echo "===== run #$run_count trade activity =====" >> "$LOG_FILE"
  echo "===== run #$run_count trade activity =====" >> "$TRADE_LOG_FILE"
  python3 scripts/summarize_trade_activity.py \
    --report data/trading_system/pipeline_run_report.json \
    --state data/trading_system/paper_state.json \
    --results-dir data/trading_system/execution_results \
    2>&1 | tee -a "$LOG_FILE" "$TRADE_LOG_FILE"

  run_finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "===== run #$run_count finished: $run_finished status=$status =====" >> "$LOG_FILE"

  if [ "$status" -ne 0 ]; then
    echo "pipeline exited non-zero; continuing after interval" >> "$LOG_FILE"
  fi

  next_run_epoch=$(($(date +%s) + INTERVAL_SECONDS))
  next_run_at="$(date -u -r "$next_run_epoch" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$next_run_epoch" +%Y-%m-%dT%H:%M:%SZ)"
  echo "===== run #$run_count sleeping: ${INTERVAL_SECONDS}s next_run=$next_run_at =====" >> "$LOG_FILE"
  print_order_status 2>&1 | tee -a "$LOG_FILE" "$TRADE_LOG_FILE"

  cooldown_with_countdown "$INTERVAL_SECONDS" "$next_run_at"
done

echo "===== pipeline 24h loop stopped: $(date -u +%Y-%m-%dT%H:%M:%SZ) runs=$run_count =====" >> "$LOG_FILE"
echo "===== trade activity log stopped: $(date -u +%Y-%m-%dT%H:%M:%SZ) runs=$run_count =====" >> "$TRADE_LOG_FILE"

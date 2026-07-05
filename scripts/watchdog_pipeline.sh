#!/usr/bin/env bash
# Pipeline watchdog: auto-detect problems and self-heal.
# Designed to run via launchd every 10 minutes.
# Only sends macOS notification when auto-recovery fails.
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env.local"
  set +a
fi

PID_FILE="${PIPELINE_PID_FILE:-data/trading_system/pipeline_loop.pid}"
LOCK_DIR="${PIPELINE_LOCK_DIR:-data/trading_system/pipeline_loop.lock}"
LOG_DIR="${PIPELINE_LOG_DIR:-data/trading_system/logs}"
DECISION_FILE="data/trading_system/model_training/paper_replay_autolearn/paper_decision_samples.jsonl"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
HEALTH_JSON="data/trading_system/observation_health.json"
FACTOR_BT_PID_FILE="data/trading_system/factor_backtest_loop.pid"
HF_BT_PID_FILE="data/trading_system/hf_backtest_loop.pid"

MAX_STALE_MINUTES=30
MAX_RESTART_ATTEMPTS=3
RESTART_COOLDOWN_FILE="$LOG_DIR/.watchdog_restart_count"

mkdir -p "$LOG_DIR"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [watchdog] $*" >> "$WATCHDOG_LOG"
}

notify() {
  local title="$1"
  local msg="$2"
  osascript -e "display notification \"$msg\" with title \"$title\"" 2>/dev/null || true
  log "NOTIFY: $title - $msg"
}

service_exists() {
  systemctl list-unit-files "$1" >/dev/null 2>&1
}

service_is_active() {
  service_exists "$1" && systemctl is-active --quiet "$1"
}

start_service_if_available() {
  local service_name="$1"
  if service_exists "$service_name"; then
    systemctl start "$service_name" >> "$WATCHDOG_LOG" 2>&1
    return $?
  fi
  return 1
}

get_restart_count() {
  if [ -f "$RESTART_COOLDOWN_FILE" ]; then
    local ts count
    ts="$(head -1 "$RESTART_COOLDOWN_FILE" 2>/dev/null || echo 0)"
    count="$(tail -1 "$RESTART_COOLDOWN_FILE" 2>/dev/null || echo 0)"
    local now_epoch
    now_epoch="$(date +%s)"
    local age=$(( now_epoch - ts ))
    # Reset counter if last restart was more than 1 hour ago
    if [ "$age" -gt 3600 ]; then
      echo 0
    else
      echo "$count"
    fi
  else
    echo 0
  fi
}

increment_restart_count() {
  local count
  count="$(get_restart_count)"
  count=$(( count + 1 ))
  printf '%s\n%s\n' "$(date +%s)" "$count" > "$RESTART_COOLDOWN_FILE"
  echo "$count"
}

reset_restart_count() {
  rm -f "$RESTART_COOLDOWN_FILE"
}

is_pipeline_running() {
  if [ ! -f "$PID_FILE" ]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
  if [ -z "$pid" ]; then
    return 1
  fi
  if ps -p "$pid" >/dev/null 2>&1; then
    return 0
  fi
  return 1
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

kill_pipeline_tree() {
  local parent_pid="$1"
  log "Killing pipeline tree (recursive): parent=$parent_pid"
  kill_descendants "$parent_pid"
  sleep 3
  # Force kill any survivors
  if kill -0 "$parent_pid" 2>/dev/null; then
    log "TERM did not stop tree, sending SIGKILL"
    kill_descendants_force "$parent_pid"
    sleep 1
  fi
}

restart_pipeline() {
  local count
  count="$(get_restart_count)"
  if [ "$count" -ge "$MAX_RESTART_ATTEMPTS" ]; then
    log "ERROR: restart count $count >= $MAX_RESTART_ATTEMPTS in last hour, giving up"
    notify "Pipeline 自愈失败" "已尝试重启 ${count} 次仍失败，需要人工介入"
    return 1
  fi

  log "Attempting restart (attempt $(( count + 1 ))/$MAX_RESTART_ATTEMPTS)"

  # Clean stale lock
  if [ -d "$LOCK_DIR" ]; then
    log "Removing stale lock: $LOCK_DIR"
    rm -rf "$LOCK_DIR"
  fi

  # Clean stale pid file
  if [ -f "$PID_FILE" ]; then
    log "Removing stale pid file"
    rm -f "$PID_FILE"
  fi

  # Start pipeline
  bash scripts/start_pipeline_24h_background.sh >> "$WATCHDOG_LOG" 2>&1
  local exit_code=$?

  if [ "$exit_code" -ne 0 ]; then
    log "start_pipeline_24h_background.sh exited with code $exit_code"
    increment_restart_count
    return 1
  fi

  sleep 5

  if is_pipeline_running; then
    log "Pipeline restarted successfully"
    reset_restart_count
    return 0
  else
    log "Pipeline failed to start after restart attempt"
    increment_restart_count
    return 1
  fi
}

check_decision_freshness() {
  if [ ! -f "$DECISION_FILE" ]; then
    echo "missing"
    return
  fi
  local mtime_epoch now_epoch age_minutes
  mtime_epoch="$(stat -f %m "$DECISION_FILE" 2>/dev/null || stat -c %Y "$DECISION_FILE" 2>/dev/null || echo 0)"
  now_epoch="$(date +%s)"
  age_minutes=$(( (now_epoch - mtime_epoch) / 60 ))
  echo "$age_minutes"
}

# --- Main watchdog logic ---

log "--- watchdog check start ---"
had_problem=0
restarted_this_round=0

# Check 1: Is pipeline process alive?
if ! is_pipeline_running; then
  log "Pipeline is NOT running"
  had_problem=1
  restart_pipeline
  if [ $? -eq 0 ]; then
    log "Self-healed: pipeline restarted"
    restarted_this_round=1
  fi
else
  log "Pipeline process: OK"
fi

# Check 2: Are decision samples fresh?
# Skip kill/restart if we already restarted this round — give new process time to produce data
stale_minutes="$(check_decision_freshness)"
if [ "$stale_minutes" = "missing" ]; then
  log "WARN: decision samples file missing"
  had_problem=1
elif [ "$stale_minutes" -gt "$MAX_STALE_MINUTES" ]; then
  had_problem=1
  if [ "$restarted_this_round" -eq 1 ]; then
    log "WARN: decision samples stale (${stale_minutes}m) but pipeline just restarted, skipping kill"
  else
    log "WARN: decision samples stale (${stale_minutes}m > ${MAX_STALE_MINUTES}m)"
    # Grace period: if pipeline process is young (< 10 min), don't kill it yet
    skip_kill=0
    if is_pipeline_running; then
      local_pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
      if [ -n "$local_pid" ]; then
        pid_age_seconds="$(( $(date +%s) - $(stat -f %m "$PID_FILE" 2>/dev/null || echo "$(date +%s)") ))"
        if [ "$pid_age_seconds" -lt 600 ]; then
          log "Pipeline process young (${pid_age_seconds}s), grace period — skipping kill"
          skip_kill=1
        fi
      fi
      if [ "$skip_kill" -eq 0 ]; then
        if [ -n "$local_pid" ]; then
          kill_pipeline_tree "$local_pid"
        fi
        restart_pipeline
        if [ $? -eq 0 ]; then
          log "Self-healed: killed stale pipeline and restarted"
        fi
      fi
    else
      restart_pipeline
    fi
  fi
else
  log "Decision samples freshness: OK (${stale_minutes}m)"
fi

# Only reset restart counter when entire check passes clean
if [ "$had_problem" -eq 0 ]; then
  reset_restart_count
fi

# Check 3: Run health check and update JSON
python3 scripts/check_observation_health.py --json-only > /dev/null 2>&1

# Check 4: Is factor-backtest loop alive?
factor_bt_alive=0
if service_is_active "factor-backtest-loop.service"; then
  factor_bt_alive=1
elif [ -f "$FACTOR_BT_PID_FILE" ]; then
  factor_pid="$(cat "$FACTOR_BT_PID_FILE" 2>/dev/null || true)"
  if [ -n "$factor_pid" ] && ps -p "$factor_pid" >/dev/null 2>&1; then
    factor_bt_alive=1
  fi
fi
if [ "$factor_bt_alive" -eq 0 ]; then
  log "factor-backtest loop NOT running, restarting"
  rm -f "data/trading_system/factor_backtest_loop.lock"
  if ! start_service_if_available "factor-backtest-loop.service"; then
    nohup bash scripts/run_factor_backtest_loop.sh >> "$LOG_DIR/factor_backtest_loop.launch.log" 2>&1 &
  fi
  sleep 2
  if service_is_active "factor-backtest-loop.service" || { [ -f "$FACTOR_BT_PID_FILE" ] && ps -p "$(cat "$FACTOR_BT_PID_FILE" 2>/dev/null)" >/dev/null 2>&1; }; then
    log "factor-backtest loop restarted OK"
  else
    log "ERROR: factor-backtest loop failed to restart"
  fi
else
  log "factor-backtest loop: OK"
fi

# Check 5: Is hf-backtest loop alive?
hf_bt_alive=0
if service_is_active "hf-backtest-loop.service"; then
  hf_bt_alive=1
elif [ -f "$HF_BT_PID_FILE" ]; then
  hf_pid="$(cat "$HF_BT_PID_FILE" 2>/dev/null || true)"
  if [ -n "$hf_pid" ] && ps -p "$hf_pid" >/dev/null 2>&1; then
    hf_bt_alive=1
  fi
fi
if [ "$hf_bt_alive" -eq 0 ]; then
  log "hf-backtest loop NOT running, restarting"
  rm -f "data/trading_system/hf_backtest_loop.lock"
  if ! start_service_if_available "hf-backtest-loop.service"; then
    nohup bash scripts/run_hf_backtest_loop.sh >> "$LOG_DIR/hf_backtest_loop.launch.log" 2>&1 &
  fi
  sleep 2
  if service_is_active "hf-backtest-loop.service" || { [ -f "$HF_BT_PID_FILE" ] && ps -p "$(cat "$HF_BT_PID_FILE" 2>/dev/null)" >/dev/null 2>&1; }; then
    log "hf-backtest loop restarted OK"
  else
    log "ERROR: hf-backtest loop failed to restart"
  fi
else
  log "hf-backtest loop: OK"
fi

# Check 6: Is news-wss listener alive?
NEWS_WSS_PID_FILE="data/trading_system/news_wss.pid"
news_wss_alive=0
if [ "${TS_DISABLE_OPEN_TOKEN_BRIDGES:-false}" = "1" ] || [ "${TS_DISABLE_OPEN_TOKEN_BRIDGES:-false}" = "true" ]; then
  log "news-wss listener disabled by TS_DISABLE_OPEN_TOKEN_BRIDGES=${TS_DISABLE_OPEN_TOKEN_BRIDGES}"
elif service_is_active "news-wss.service"; then
  news_wss_alive=1
elif [ -f "$NEWS_WSS_PID_FILE" ]; then
  news_pid="$(cat "$NEWS_WSS_PID_FILE" 2>/dev/null || true)"
  if [ -n "$news_pid" ] && ps -p "$news_pid" >/dev/null 2>&1; then
    news_wss_alive=1
  fi
fi
if [ "${TS_DISABLE_OPEN_TOKEN_BRIDGES:-false}" = "1" ] || [ "${TS_DISABLE_OPEN_TOKEN_BRIDGES:-false}" = "true" ]; then
  :
elif [ "$news_wss_alive" -eq 0 ]; then
  log "news-wss listener NOT running, restarting"
  if ! start_service_if_available "news-wss.service"; then
    bash scripts/start_news_wss.sh >> "$WATCHDOG_LOG" 2>&1
  fi
  sleep 2
  if service_is_active "news-wss.service" || { [ -f "$NEWS_WSS_PID_FILE" ] && ps -p "$(cat "$NEWS_WSS_PID_FILE" 2>/dev/null)" >/dev/null 2>&1; }; then
    log "news-wss listener restarted OK"
  else
    log "ERROR: news-wss listener failed to restart"
  fi
else
  log "news-wss listener: OK"
fi

log "--- watchdog check end ---"

#!/usr/bin/env bash
set -u

ROOT="/Users/ma7/binance-trading-skill-system"
HERMES_HOME="${HERMES_HOME:-/Users/ma7/.hermes}"
LOG_DIR="$ROOT/data/trading_system/logs"
LOG_FILE="$LOG_DIR/hermes_gateway_maintenance.log"
LABEL="ai.hermes.gateway"
FD_THRESHOLD="${HERMES_FD_RESTART_THRESHOLD:-220}"
MODE="${1:-full}"
LOCK_DIR="$HERMES_HOME/runtime/hermes-gateway-maintenance.lock"

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG_FILE"
}

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_DIR")"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    trap release_lock EXIT INT TERM
    return 0
  fi

  local owner=""
  owner="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    log "maintenance_skip reason=already_running owner=$owner mode=$MODE"
    return 1
  fi

  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "maintenance_skip reason=lock_unavailable mode=$MODE"
    return 1
  fi
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
  trap release_lock EXIT INT TERM
}

release_lock() {
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

delete_old_tmp_files() {
  local deleted=0
  local dir
  for dir in "$HERMES_HOME/sessions" "$HERMES_HOME/weixin/accounts" "$HERMES_HOME/tmp"; do
    [ -d "$dir" ] || continue
    while IFS= read -r file; do
      [ -n "$file" ] || continue
      rm -f "$file" && deleted=$((deleted + 1))
    done < <(find "$dir" -type f \( -name '*.tmp' -o -name '.*.tmp' \) -mtime +1 -print 2>/dev/null)
  done
  log "tmp_cleanup deleted=$deleted"
}

gateway_pid() {
  pgrep -f 'hermes_cli.main gateway run' | head -1 || true
}

fd_count_for_pid() {
  local pid="$1"
  lsof -p "$pid" 2>/dev/null | wc -l | tr -d ' '
}

restart_gateway() {
  local reason="$1"
  local uid
  local domain
  local old_pid
  local new_pid
  local new_fds
  uid="$(id -u)"
  domain="user/$uid/$LABEL"
  old_pid="$(gateway_pid)"
  log "gateway_restart reason=$reason domain=$domain old_pid=${old_pid:-missing}"

  if ! launchctl kickstart -k "$domain" >> "$LOG_FILE" 2>&1; then
    log "gateway_restart_failed reason=$reason domain=$domain"
    return 1
  fi

  sleep 5
  new_pid="$(gateway_pid)"
  if [ -z "$new_pid" ]; then
    log "gateway_after_restart pid=missing"
    return 1
  fi

  new_fds="$(fd_count_for_pid "$new_pid")"
  log "gateway_after_restart old_pid=${old_pid:-missing} pid=$new_pid fd=$new_fds"
  if [ -n "$old_pid" ] && [ "$new_pid" = "$old_pid" ]; then
    log "gateway_restart_failed reason=pid_unchanged pid=$new_pid"
    return 1
  fi
}

main() {
  acquire_lock || return 0

  if [ "$MODE" != "--health-only" ]; then
    delete_old_tmp_files
  fi

  local pid
  pid="$(gateway_pid)"
  if [ -z "$pid" ]; then
    restart_gateway "missing_process"
    return $?
  fi

  local fds
  fds="$(fd_count_for_pid "$pid")"
  log "gateway_check mode=$MODE pid=$pid fd=$fds threshold=$FD_THRESHOLD"

  if [ "${fds:-0}" -gt "$FD_THRESHOLD" ]; then
    restart_gateway "fd_threshold"
    return $?
  fi
}

main "$@"

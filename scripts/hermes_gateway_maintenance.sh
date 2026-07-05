#!/usr/bin/env bash
set -u

ROOT="/Users/ma7/binance-trading-skill-system"
HERMES_HOME="${HERMES_HOME:-/Users/ma7/.hermes}"
LOG_DIR="$ROOT/data/trading_system/logs"
LOG_FILE="$LOG_DIR/hermes_gateway_maintenance.log"
LABEL="ai.hermes.gateway"
FD_THRESHOLD="${HERMES_FD_RESTART_THRESHOLD:-220}"

mkdir -p "$LOG_DIR"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG_FILE"
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
  local uid
  uid="$(id -u)"
  log "gateway_restart reason=$1"
  launchctl kickstart -k "gui/$uid/$LABEL" >> "$LOG_FILE" 2>&1 || true
  sleep 5
  local new_pid
  new_pid="$(gateway_pid)"
  if [ -n "$new_pid" ]; then
    log "gateway_after_restart pid=$new_pid fd=$(fd_count_for_pid "$new_pid")"
  else
    log "gateway_after_restart pid=missing"
  fi
}

main() {
  delete_old_tmp_files

  local pid
  pid="$(gateway_pid)"
  if [ -z "$pid" ]; then
    restart_gateway "missing_process"
    return 0
  fi

  local fds
  fds="$(fd_count_for_pid "$pid")"
  log "gateway_check pid=$pid fd=$fds threshold=$FD_THRESHOLD"

  if [ "${fds:-0}" -gt "$FD_THRESHOLD" ]; then
    restart_gateway "fd_threshold"
  fi
}

main "$@"

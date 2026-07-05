#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="v5-trading.service"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  echo "systemd=$SERVICE_NAME"
  systemctl --no-pager --lines=30 status "$SERVICE_NAME" || true
fi

LOG_DIR="${PIPELINE_LOG_DIR:-data/trading_system/logs}"
LOCK_DIR="${PIPELINE_LOCK_DIR:-data/trading_system/pipeline_loop.lock}"
PID_FILE="${PIPELINE_PID_FILE:-data/trading_system/pipeline_loop.pid}"

pid=""
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  echo "状态: 运行中 pid=$pid"
  ps -p "$pid" -o pid,etime,stat,command
else
  echo "状态: 未运行"
fi

if [ -d "$LOCK_DIR" ]; then
  echo "锁文件: 存在 $LOCK_DIR"
else
  echo "锁文件: 无"
fi

pipeline_log="$(ls -t "$LOG_DIR"/pipeline_24h_*.log 2>/dev/null | head -1 || true)"
trade_log="$(ls -t "$LOG_DIR"/trade_activity_*.log 2>/dev/null | head -1 || true)"

echo "最新主日志: ${pipeline_log:-无}"
echo "最新交易日志: ${trade_log:-无}"

if [ -n "$pipeline_log" ]; then
  echo ""
  echo "主日志末尾:"
  tail -n 30 "$pipeline_log"
fi

if [ -n "$trade_log" ]; then
  echo ""
  echo "交易日志末尾:"
  tail -n 40 "$trade_log"
fi

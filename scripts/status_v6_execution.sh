#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="v6-execution.service"
STATE_DIR="${TS_V6_EXECUTION_DIR:-data/trading_system/v6_execution}"
LOG_DIR="${TS_V6_EXECUTION_LOG_DIR:-$STATE_DIR/logs}"
PID_FILE="${TS_V6_EXECUTION_PID_FILE:-$STATE_DIR/v6_execution_loop.pid}"
LATEST_SUMMARY="${TS_V6_LATEST_SUMMARY_FILE:-$STATE_DIR/latest_run_summary.json}"

service_active=0
if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  echo "systemd=$SERVICE_NAME"
  systemctl status "$SERVICE_NAME" --no-pager --lines=18 || true
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    service_active=1
  fi
  echo
fi

pid=""
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
fi

if [ "$service_active" = "1" ]; then
  echo "状态: 运行中 systemd=$SERVICE_NAME"
elif [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
  echo "状态: 运行中 pid=$pid"
  ps -p "$pid" -o pid,etime,stat,command
else
  echo "状态: 未运行"
fi

echo "PID文件: ${PID_FILE}"
echo "日志目录: ${LOG_DIR}"
echo "最新摘要: ${LATEST_SUMMARY}"

if [ -f "$LATEST_SUMMARY" ]; then
  echo ""
  echo "最新摘要:"
  venv/bin/python - "$LATEST_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
signals = payload.get("signals") or []
print({
    "generated_at": payload.get("generated_at"),
    "order_mode": payload.get("order_mode"),
    "submit": payload.get("submit"),
    "signal_count": len(signals),
    "intent_count": len(payload.get("intents") or []),
    "result_count": len(payload.get("results") or []),
    "positions": payload.get("positions") or [],
})
for signal in signals[:5]:
    print({
        "symbol": signal.get("symbol"),
        "side_bias": signal.get("side_bias"),
        "long_probability": signal.get("long_probability"),
        "short_probability": signal.get("short_probability"),
        "wait_probability": signal.get("wait_probability"),
        "price": signal.get("price"),
    })
PY
fi

launch_log="$LOG_DIR/v6_execution.launch.log"
if [ -f "$launch_log" ]; then
  echo ""
  echo "日志末尾:"
  tail -n 80 "$launch_log"
fi

#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

SERVICE_NAME="news-wss.service"
PID_FILE="data/trading_system/news_wss.pid"
HEALTH_FILE="data/trading_system/news_wss_health.json"
BRIDGE_FILE="data/trading_system/skill_bridges/opennews_wss.json"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  echo "systemd=$SERVICE_NAME"
  systemctl --no-pager --lines=30 status "$SERVICE_NAME" || true
else
  pid=""
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    pid="$(cat "$PID_FILE")"
  else
    pid="$(pgrep -f "trading_system.cli news-wss" | head -n 1 || true)"
  fi

  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "状态: 运行中 pid=$pid"
    ps -p "$pid" -o pid,etime,stat,command || true
  else
    echo "状态: 停止"
  fi
fi

for file in "$HEALTH_FILE" "$BRIDGE_FILE"; do
  echo
  echo "文件: $file"
  if [ -f "$file" ]; then
    ls -lh "$file"
    head -c 1200 "$file"
    echo
  else
    echo "不存在"
  fi
done

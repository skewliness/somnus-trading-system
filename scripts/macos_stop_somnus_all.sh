#!/usr/bin/env bash
set -u
LABELS=(
  com.somnus.telegram-v5
  com.somnus.telegram-v6
  com.somnus.telegram-v6-live
  com.somnus.telegram-v6-live-momo2
  com.somnus.v5-trading
  com.somnus.v5-position-management
  com.somnus.v5-position-reconcile
  com.somnus.v6-execution
  com.somnus.v6-execution-live
  com.somnus.v6-execution-live-momo2
  com.somnus.v6-entry-gate-labeler
  com.somnus.v6-ledger-reconcile
  com.somnus.v6-ledger-reconcile-live
  com.somnus.v6-ledger-reconcile-live-momo2
  com.somnus.v6-distill
  com.somnus.v5-signal-refresh
  com.somnus.v6-jsonl-rotate
  com.somnus.kline-cache-maintenance
  com.somnus.telegram-whale
  com.somnus.harvest-shadow
  com.somnus.v6-distill-monitor
  com.somnus.expand-smart-money-watchlist
)
for label in "${LABELS[@]}"; do
  sudo launchctl bootout system "/Library/LaunchDaemons/$label.plist" 2>/dev/null || true
  launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/$label.plist" 2>/dev/null || true
done

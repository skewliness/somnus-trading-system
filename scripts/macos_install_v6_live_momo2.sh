#!/usr/bin/env bash
set -euo pipefail

ROOT="${SOMNUS_ROOT:-/Users/ma7/binance-trading-skill-system}"
LIVE_ENV="${SOMNUS_V6_LIVE_MOMO2_ENV_FILE:-$ROOT/.env.v6_live_momo2}"
LABELS=(
  com.somnus.v6-ledger-reconcile-live-momo2
  com.somnus.v6-execution-live-momo2
  com.somnus.telegram-v6-live-momo2
)

if [[ ! -f "$LIVE_ENV" ]]; then
  echo "missing V6 momo2 credentials file: $LIVE_ENV" >&2
  exit 78
fi
chmod 600 "$LIVE_ENV"

echo "Running signed read-only Binance Futures mainnet readiness check for momo2..."
"$ROOT/scripts/macos_run_somnus_service.sh" v6-live-readiness-momo2

for label in "${LABELS[@]}"; do
  source_plist="$ROOT/launchd/$label.plist"
  target_plist="/Library/LaunchDaemons/$label.plist"
  if [[ ! -f "$source_plist" ]]; then
    echo "missing launchd definition: $source_plist" >&2
    exit 78
  fi
  sudo install -o root -g wheel -m 644 "$source_plist" "$target_plist"
  sudo launchctl bootout "system/$label" 2>/dev/null || true
  sudo launchctl bootstrap system "$target_plist"
  sudo launchctl enable "system/$label"
done

sleep 3
for label in "${LABELS[@]}"; do
  echo "### $label"
  sudo launchctl print "system/$label" | grep -E "state =|pid =|last exit code" || true
done

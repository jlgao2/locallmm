#!/usr/bin/env bash
# Raises the macOS GPU memory limit to 40 GB so MLX has more headroom for
# large contexts. Installs a LaunchDaemon so it persists across reboots.
#
# Run with sudo:   sudo bash macos/install-gpu-limit.sh
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "Please run with sudo:  sudo bash $0"; exit 1; }

SRC="$(cd "$(dirname "$0")" && pwd)/com.locallmm.gpulimit.plist"
DEST="/Library/LaunchDaemons/com.locallmm.gpulimit.plist"

cp "$SRC" "$DEST"
chown root:wheel "$DEST"
chmod 644 "$DEST"
launchctl load -w "$DEST" 2>/dev/null || launchctl bootstrap system "$DEST" 2>/dev/null || true
sleep 1

echo "iogpu.wired_limit_mb is now: $(sysctl -n iogpu.wired_limit_mb) MB"
echo "Done — GPU memory limit set to 40 GB; it will re-apply automatically on every reboot."

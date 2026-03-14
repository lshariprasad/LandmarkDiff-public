#!/usr/bin/env bash
# watch_backup_checkpoints.sh - Periodically back up Phase A checkpoints
#
# Polls the checkpoint directory every INTERVAL seconds and runs
# backup_checkpoints.sh when new checkpoint-* dirs appear.
#
# Usage:
#   nohup ./scripts/watch_backup_checkpoints.sh &> logs/backup_watcher.log &
#   ./scripts/watch_backup_checkpoints.sh --interval 600  # check every 10 min
#   ./scripts/watch_backup_checkpoints.sh --stop          # kill running watcher

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIDFILE="$WORK_DIR/logs/backup_watcher.pid"
INTERVAL=300  # default: check every 5 minutes

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --stop)
            if [[ -f "$PIDFILE" ]]; then
                pid=$(cat "$PIDFILE")
                if kill -0 "$pid" 2>/dev/null; then
                    kill "$pid"
                    echo "Stopped backup watcher (PID $pid)."
                else
                    echo "Watcher PID $pid not running."
                fi
                rm -f "$PIDFILE"
            else
                echo "No pidfile found at $PIDFILE."
            fi
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--interval SECONDS] [--stop]"
            exit 1
            ;;
    esac
done

mkdir -p "$WORK_DIR/logs"

# Write pidfile
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

SRC_DIR="${CKPT_SRC:-$WORK_DIR/outputs/phaseA_v3_curriculum}"
export CKPT_SRC="$SRC_DIR"
BACKUP_SCRIPT="$SCRIPT_DIR/backup_checkpoints.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Backup watcher started (PID $$, interval ${INTERVAL}s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watching: $SRC_DIR"

last_count=0

while true; do
    if [[ -d "$SRC_DIR" ]]; then
        # Count checkpoint dirs
        count=$(find "$SRC_DIR" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)

        if [[ "$count" -gt "$last_count" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] New checkpoint detected ($last_count -> $count). Running backup."
            bash "$BACKUP_SCRIPT" 2>&1
            last_count=$count
        fi
    fi

    sleep "$INTERVAL"
done

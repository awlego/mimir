#!/usr/bin/env bash
# job-status [job]: state of the current (or named) job and the end of its log.
. "$(dirname "$0")/common.sh"
if job="$(running_job)"; then
  pid="$(cat "$STATE/job.pid")"
  echo "RUNNING: $job (pid $pid, up $(ps -o etime= -p "$pid" | tr -d ' '))"
else
  job="$(cat "$STATE/job.name" 2>/dev/null || true)"
  echo "No mimir job running.${job:+ Last job: $job.}"
fi
name="${1:-$job}"
[ -n "$name" ] || exit 0
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || die "Bad job name."
[ -f "runs/$name/DONE" ] && echo "DONE: runs/$name/DONE exists."
ckpt="$(ls -dt runs/"$name"/checkpoint-* 2>/dev/null | head -1 || true)"
[ -n "$ckpt" ] && echo "Latest checkpoint: $ckpt ($(date -r "$ckpt" '+%F %T'))"
log="$STATE/logs/$name.log"
[ -f "$log" ] || exit 0
echo "--- end of $log ---"
# Progress bars redraw with \r; show each redraw as its own line.
tail -c 20000 "$log" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 30

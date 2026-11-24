#!/usr/bin/env bash
# Stop the running job gracefully: SIGTERM, so it checkpoints and exits. Kills it only after MIMIR_STOP_WAIT seconds.
. "$(dirname "$0")/common.sh"
job="$(running_job)" || { echo "No mimir job running."; exit 0; }
pid="$(cat "$STATE/job.pid")"
wait="${MIMIR_STOP_WAIT:-600}"
echo "Asking '$job' (pid $pid) to checkpoint and stop; waiting up to ${wait}s."
kill -TERM "$pid"
for ((i = 0; i < wait; i += 5)); do
  kill -0 "$pid" 2>/dev/null || { echo "Stopped after ${i}s. Run it again to resume."; exit 0; }
  sleep 5
done
echo "Still running after ${wait}s: killing it. Work since its last checkpoint is lost."
kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"

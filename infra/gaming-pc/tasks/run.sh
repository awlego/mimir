#!/usr/bin/env bash
# run <job>: start infra/gaming-pc/jobs/<job>.sh detached, logging to runs/_gaming-pc/logs/<job>.log.
# Refuses if a job is running or the GPU is busy. Running a stopped job again resumes it.
. "$(dirname "$0")/common.sh"
name="${1:-}"
[[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || die "Give a job name. Jobs: $(cd infra/gaming-pc/jobs && ls *.sh | sed 's/\.sh$//' | tr '\n' ' ')"
[ -f "infra/gaming-pc/jobs/$name.sh" ] || die "No job '$name'. Jobs: $(cd infra/gaming-pc/jobs && ls *.sh | sed 's/\.sh$//' | tr '\n' ' ')"
if job="$(running_job)"; then die "Job '$job' is already running. Stop it first."; fi
[ -f "runs/$name/DONE" ] && { echo "Job '$name' already finished (runs/$name/DONE exists). Delete that file to run it again."; exit 0; }
used="$(gpu_mem_used)"
[ "$used" -le "$GPU_BUSY_MIB" ] || die "GPU busy: ${used} MiB in use (threshold ${GPU_BUSY_MIB}). Not starting."
log="$STATE/logs/$name.log"
echo "=== $(date '+%F %T') run $name at $(git rev-parse --short HEAD) ===" >> "$log"
export PYTHONPATH="$REPO/infra/gaming-pc${PYTHONPATH:+:$PYTHONPATH}"   # for graceful.py
# setsid + nohup: the job outlives this task and wsl.exe; stdin/stdout detached so the task returns.
setsid nohup bash "infra/gaming-pc/jobs/$name.sh" >> "$log" 2>&1 < /dev/null &
echo $! > "$STATE/job.pid"
echo "$name" > "$STATE/job.name"
sleep 5
running_job > /dev/null || { tail -n 20 "$log"; die "Job '$name' exited within 5 s; log above."; }
echo "Started '$name' (pid $!). Log: runs/_gaming-pc/logs/$name.log"

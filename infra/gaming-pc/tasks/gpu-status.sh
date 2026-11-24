#!/usr/bin/env bash
# The GPU at a glance, and whether a job may start.
. "$(dirname "$0")/common.sh"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,power.draw --format=csv
used="$(gpu_mem_used)"
if job="$(running_job)"; then
  echo "Mimir job running: $job (pid $(cat "$STATE/job.pid"))"
elif [ "$used" -gt "$GPU_BUSY_MIB" ]; then
  echo "BUSY: ${used} MiB in use (threshold ${GPU_BUSY_MIB}), not by a mimir job. Alex is probably using the GPU."
else
  echo "FREE: ${used} MiB in use (threshold ${GPU_BUSY_MIB}). A job can start."
fi
# WSL's nvidia-smi can't list processes (WDDM); the Windows one can, without memory figures.
win=/mnt/c/Windows/System32/nvidia-smi.exe
if [ -x "$win" ]; then
  echo "Windows processes on the GPU (for information; the desktop always shows up here):"
  "$win" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | sed 's/^/  compute: /' || true
  "$win" 2>/dev/null | sed -n '/Processes:/,$p' | grep -E '^\| +[0-9]' | awk '{print "  " $0}' || true
fi

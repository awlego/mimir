# Shared setup for the gaming-pc task scripts. Sourced, not run.
# Heimdall starts tasks through wsl.exe --exec: no login shell, no .bashrc.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export PATH="$REPO/env3.12/bin:$HOME/.local/bin:/usr/lib/wsl/lib:/usr/local/bin:/usr/bin:/bin"
STATE="$REPO/runs/_gaming-pc"   # pid file, current job name, logs (runs/ is gitignored)
mkdir -p "$STATE/logs"
# Machine-local settings (not committed), e.g. MIMIR_GPU_BUSY_MIB=2500.
[ -f "$REPO/infra/gaming-pc/local.env" ] && . "$REPO/infra/gaming-pc/local.env"
GPU_BUSY_MIB="${MIMIR_GPU_BUSY_MIB:-2500}"   # VRAM in use above this = someone else is on the GPU

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the running job's name and returns 0, or returns 1 if none is running.
running_job() {
  [ -f "$STATE/job.pid" ] || return 1
  local pid; pid="$(cat "$STATE/job.pid")"
  if kill -0 "$pid" 2>/dev/null; then cat "$STATE/job.name"; return 0; fi
  return 1
}

gpu_mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

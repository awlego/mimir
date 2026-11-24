# Setting up the gaming PC as mimir's overnight GPU runner

This guide sets up `gaming-pc` (Windows, RTX 5080 with 16 GB) to run mimir's GPU jobs inside WSL2. Heimdall then drives them with six fixed tasks: `sync`, `gpu-status`, `run`, `job-status`, `stop` and `push-results`. Expect about an hour, most of it downloads.

Each step says what you should see. If you don't see it, use the "If not" note under the step. `PS>` means a Windows PowerShell prompt, and `$` means the Ubuntu (WSL) shell.

Versions below were checked on 2026-11-23. The sources are at the end.

## How it fits together

- Heimdall's machine agent already runs on Windows. It is the `gaming-pc` machine, platform `win32`, online.
- Each task is a `wsl.exe … --exec bash <script>` command, so the work happens in Ubuntu.
- `run` starts a job detached inside WSL, so it outlives the task.
- `stop` sends the job SIGTERM, so it saves a checkpoint before exiting. Running the same job again resumes it.
- Jobs write to `~/mimir/runs/<job>/`, which git ignores.
- `push-results` commits only small result files, to the branch `gaming-pc-results`. The steward merges that branch.

---

## 1. NVIDIA driver (Windows only)

Install the latest **Game Ready** (or Studio) driver for the RTX 5080 from <https://www.nvidia.com/drivers/>, or through the NVIDIA app. This Windows driver is the only NVIDIA driver you need. It also serves CUDA inside WSL.

**Never install an NVIDIA driver inside Ubuntu.** That means no `nvidia-driver-*`, no `cuda-drivers`, no `nvidia-open`, and no `cuda` or `cuda-13-x` meta-packages. A Linux driver overwrites the driver that WSL maps in from Windows [1].

The PyTorch build used below runs on CUDA 13.2. CUDA 13.x needs driver **580 or newer**, and 13.2 belongs to the R595 driver branch [2]. Install the newest driver offered; it should be at least 595.

```powershell
PS> nvidia-smi
```

**You should see:**
- `NVIDIA GeForce RTX 5080` with 16303 MiB or so.
- A `Driver Version` of 580 or newer (595 or newer to match CUDA 13.2).
- `CUDA Version: 13.x` with x ≥ 2.

**If not:**
- `nvidia-smi` not found: reboot after installing the driver.
- An old version: update in the NVIDIA app, under Drivers.

## 2. WSL2 with Ubuntu 24.04

In an **administrator** PowerShell:

```powershell
PS> wsl --install -d Ubuntu-24.04
```

Reboot if asked. Ubuntu then opens and asks for a Linux username and password. Remember the username: this guide calls it `LINUXUSER`.

Then, in a normal PowerShell:

```powershell
PS> wsl --update
PS> wsl -l -v
```

**You should see:** `Ubuntu-24.04   Running (or Stopped)   2`. The last column is the WSL version, and it must be `2`.

**If not:**
- Version 1: run `wsl --set-version Ubuntu-24.04 2`.
- "Virtual Machine Platform" or virtualization errors: enable virtualization (SVM or VT-x) in the BIOS, then rerun `wsl --install`.
- A different distro name, such as `Ubuntu`: replace `Ubuntu-24.04` everywhere below, including in `machine-workspace.json`.

### Keep Ubuntu running when no window is open

By default WSL shuts a distro down 15 seconds after it goes idle, which would kill an overnight job. `instanceIdleTimeout = -1` turns that off [3]. Create `C:\Users\<you>\.wslconfig` (in Notepad: `notepad $env:USERPROFILE\.wslconfig`):

```ini
[general]
instanceIdleTimeout=-1

[wsl2]
# WSL gets half the PC's RAM by default. Uncomment to give it more for dataset work:
# memory=24GB
```

```powershell
PS> wsl --shutdown
PS> wsl -d Ubuntu-24.04 --exec true
```

Then close every Ubuntu window, wait two minutes, and run `PS> wsl -l -v`.

**You should see:** `Ubuntu-24.04 Running`.

**If it says `Stopped`:**
- Check that the file is named exactly `.wslconfig` and not `.wslconfig.txt`, and that `wsl --update` ran. Older WSL versions ignore `instanceIdleTimeout`.
- As a fallback, add a Task Scheduler task that runs at your logon (not as SYSTEM): `wsl.exe -d Ubuntu-24.04 --exec sleep infinity`.

## 3. Base tools in Ubuntu

Open Ubuntu from the Start menu. Then:

```bash
$ sudo apt update && sudo apt install -y git tmux curl build-essential gh
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ exec bash          # picks up ~/.local/bin on PATH
$ uv --version && gh --version && nvidia-smi
```

**You should see:**
- A `uv` and a `gh` version.
- The same `nvidia-smi` table as on Windows, here served from `/usr/lib/wsl/lib`.
- In WSL, the process list at the bottom is empty or says it isn't supported. That is expected: NVML can't list processes in WSL [1].

**If not:**
- `nvidia-smi: command not found`: run `ls /usr/lib/wsl/lib/nvidia-smi`. If the file is missing, the Windows driver is too old or WSL is version 1. Go back to steps 1 and 2.
- `Failed to initialize NVML`: run `PS> wsl --shutdown` and reopen Ubuntu.
- `gh` missing from apt: install it following <https://github.com/cli/cli/blob/trunk/docs/install_linux.md>.

## 4. GitHub login and the repo

Keep the repo in the Linux home folder, **not** under `/mnt/c`. Windows drives are slow from WSL and break file watching.

```bash
$ gh auth login            # GitHub.com → HTTPS → "Authenticate Git with your GitHub credentials": Yes → browser
$ gh auth setup-git
$ git config --global user.name "Alex Wilson"
$ git config --global user.email "<the email you commit with>"
$ gh repo clone awlego/mimir ~/mimir
$ cd ~/mimir && git status
```

**You should see:** `On branch main … up to date with 'origin/main'`. `gh auth status` should say `Logged in to github.com account awlego`.

**If not:**
- The browser doesn't open: open the URL `gh` prints in Windows and type the code shown.
- `push` asks for a password later: rerun `gh auth setup-git`.

## 5. Python venv and PyTorch for sm_120

The RTX 5080 is Blackwell, compute capability 12.0 (`sm_120`). PyTorch wheels built with CUDA 12.8 or newer include sm_120 kernels; maintainers confirm this from PyTorch 2.7 on [4]. Older builds, such as `cu126`, load and then fail on the GPU with "no kernel image is available".

`torch==2.15.0` from PyPI is built on CUDA 13.2 on Linux [5], so no special index is needed. `infra/gaming-pc/requirements-gpu.txt` pins the whole stack: torch 2.15.0, transformers 5.24.0, peft 0.22.0, trl 1.18.2, accelerate 1.16.0, datasets 5.2.0 and huggingface_hub 2.4.0. All of them need Python ≥ 3.11 [5].

```bash
$ cd ~/mimir
$ uv venv env3.12 --python 3.12
$ uv pip install --python env3.12 -r infra/gaming-pc/requirements-gpu.txt
$ env3.12/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

**You should see:** `2.15.0+cu132 13.2 True NVIDIA GeForce RTX 5080 [... 'sm_120' ...]`. The exact suffix may differ. What matters is `True` and `sm_120` in the list.

**If not:**
- `False`: CUDA isn't reaching WSL. Check that `nvidia-smi` works in Ubuntu (step 3), then run `PS> wsl --shutdown`.
- `sm_120` missing: an old or CPU wheel got installed. Run `uv pip install --python env3.12 --reinstall torch==2.15.0`.
- If PyPI ever serves a CPU wheel: run `uv pip install --python env3.12 torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132`. That index carries 2.15.0, and uv's `--torch-backend=cu1xx` flag does the same [6].
- `CUDA driver version is insufficient`: update the Windows driver (step 1).

Packages the plan lists for later can be added when a job needs them: lm-eval, scikit-learn, sae-lens, and vLLM in WSL only. bitsandbytes is optional, and its Blackwell support is unconfirmed.

## 6. Hugging Face login and the Gemma licence

1. In a browser, logged in to Hugging Face, open <https://huggingface.co/google/gemma-3-4b-it> and accept the licence. The repo is public, but its files are gated behind Google's licence [7]. Do the same for Qwen3-4B later only if its page asks.
2. Create a **Read** token at <https://huggingface.co/settings/tokens>.
3. Log in:

```bash
$ env3.12/bin/hf auth login       # paste the token; answer "n" to adding it as a git credential
$ env3.12/bin/hf auth whoami
```

**You should see:** your Hugging Face username [8].

**If not:**
- `hf: not found`: step 5 didn't finish, because `hf` comes with huggingface_hub.
- A 401 or 403 error in step 7 later: the licence wasn't accepted with the same account as the token.

## 7. Smoke test

```bash
$ cd ~/mimir && env3.12/bin/python infra/gaming-pc/smoke_test.py
```

The first run downloads the model, several GB, into `~/.cache/huggingface`.

**You should see:**
- The torch and CUDA versions, with `sm_120` in the arch list.
- `bf16 matmul ok: True`.
- `loaded Gemma3ForConditionalGeneration`.
- A one-sentence answer about model organisms.
- `VRAM:` about 8 GiB allocated and peak, of about 15.9 GiB.
- `SMOKE TEST PASSED`.

Please paste the whole output into Heimdall: it is our first real number for this card.

**If not:**
- `no sm_120 kernels`: go back to step 5.
- `OutOfMemoryError`: something else is using the GPU, such as a game or a browser with hardware video. Close it and rerun.
- `GatedRepoError` or a 403: go back to step 6.
- A very slow download: optionally run `uv pip install --python env3.12 hf_xet` first, then rerun.

## 8. Heimdall workspace and tasks

### First, check the bridge from Windows

This is how Heimdall will start jobs. Close every Ubuntu window, then in PowerShell:

```powershell
PS> wsl.exe -d Ubuntu-24.04 --exec bash /home/LINUXUSER/mimir/infra/gaming-pc/tasks/run.sh dummy
```

Wait two minutes, still with no Ubuntu window open, then run:

```powershell
PS> wsl.exe -d Ubuntu-24.04 --exec bash /home/LINUXUSER/mimir/infra/gaming-pc/tasks/job-status.sh
```

**You should see:**
- The first command prints `Started 'dummy'` and returns after about 5 seconds.
- The second prints `RUNNING: dummy`, with log lines past `step 100/120` or `DONE`.

Then clean up with `rm -r ~/mimir/runs/dummy` in Ubuntu.

**If not:**
- `--exec` unknown: run `wsl --update`.
- `GPU busy`: your desktop uses more than the 2500 MiB default. Set the threshold (step 9, first part), then retry.
- The job is gone and the log stops early: the distro was shut down. Recheck the `.wslconfig` part of step 2.

### Then add the workspace

The machine agent on Windows reads `C:\Users\<you>\.heimdall\machine.json`. That is `%USERPROFILE%\.heimdall\machine.json`, the file the first `node machine-agent.mjs --hub … --token …` created.

1. Open it in Notepad: `PS> notepad $env:USERPROFILE\.heimdall\machine.json`.
2. Copy the `"mimir": { … }` block from `infra/gaming-pc/machine-workspace.json` into its `"workspaces"` object, next to `"demo"`, with a comma between the two.
   - Open the file in Windows at `\\wsl.localhost\Ubuntu-24.04\home\LINUXUSER\mimir\infra\gaming-pc\machine-workspace.json`.
3. Replace every `LINUXUSER` with your Linux username: 7 places, including the `folder` line.
4. Save. Keep `hub`, `token` and everything else as they are.

The current agent reloads the file by itself within a few seconds, and its console says `Config reloaded.`
- The agent on this PC is an older version: Heimdall shows it without realm support.
- If nothing reloads, stop the agent (Ctrl+C in its window) and start it again.
- Better, download a fresh `machine-agent.mjs` from Heimdall (Machines, Add shows the `curl` line) and start that. It keeps the same `machine.json`.

**Check:** ask Heimdall (the steward) to run `gpu-status` and `list_files` on `gaming-pc`/`mimir`. You should see the GPU table and `FREE: …`, and the repo's top-level files.

**If not:**
- `Config not reloaded: Unexpected token`: the JSON has a typo, often a missing comma or a single `\` where it needs `\\`.
- `No such file or folder` for every task: the agent can't see `\\wsl.localhost\…`.
  - Open that path in Explorer. If Explorer can't either, start Ubuntu once and retry.
  - If Explorer works but the agent still fails, try `\\\\wsl$\\Ubuntu-24.04\\home\\LINUXUSER\\mimir` as the `folder`.
  - If neither works, set `folder` to any plain Windows folder, such as `"C:\\Users\\<you>\\.heimdall\\mimir"` (create it). The tasks still work, because they use absolute WSL paths. Only Heimdall's `list_files` and `read_file` lose sight of the repo; `job-status` still shows the logs.
- `couldn't start: spawn wsl.exe ENOENT`: replace `"wsl.exe"` with `"C:\\Windows\\System32\\wsl.exe"`.

## 9. Calibrate "GPU busy" and try the runner

WSL can't see which Windows programs use the GPU. So `run` decides the GPU is busy when **VRAM in use** is above a threshold. The default is 2500 MiB. Set it for your desktop:

1. With your usual apps open but nothing heavy, run `bash infra/gaming-pc/tasks/gpu-status.sh` and note the MiB in use, for example 1200.
2. Then:

```bash
$ echo 'MIMIR_GPU_BUSY_MIB=2500' > ~/mimir/infra/gaming-pc/local.env   # idle figure + ~1300
```

`local.env` stays out of git.

Then try the full cycle with the GPU-free test job:

```bash
$ cd ~/mimir
$ bash infra/gaming-pc/tasks/run.sh dummy        # Started 'dummy' (pid …)
$ bash infra/gaming-pc/tasks/job-status.sh       # RUNNING: dummy … step 10/120, checkpointed
$ bash infra/gaming-pc/tasks/stop.sh             # Stopped after 5s. Run it again to resume.
$ bash infra/gaming-pc/tasks/run.sh dummy        # log shows "starting at step N/120" with N > 0
```

The same tasks also work from Heimdall. After about two minutes, `job-status` shows `DONE`. Then delete the test output with `rm -r ~/mimir/runs/dummy`.

### When you want the GPU back

Stop the job, which costs at most the time since its last checkpoint (about 20 minutes). Then run it again later. There are three ways to stop it:
- Ask Heimdall to run `stop`.
- In Ubuntu, run `bash ~/mimir/infra/gaming-pc/tasks/stop.sh`.
- Use a desktop shortcut whose target is `wsl.exe -d Ubuntu-24.04 --exec bash /home/LINUXUSER/mimir/infra/gaming-pc/tasks/stop.sh`.

`run` refuses to start while a game holds more VRAM than the threshold. But a job that is already running does **not** step aside by itself when you start a game.

## 10. Overnight settings (Windows)

- **Sleep:** Settings → System → Power → Screen and sleep: set "When plugged in, put my device to sleep after" to **Never**. The screen may still turn off.
- **Windows Update restarts:** set Active hours to cover the night, or pause updates around long runs. A restart kills the job, which then resumes from its checkpoint when restarted.
- **After a reboot:** log in once. WSL and the machine agent run as your user, not as a service.

---

## Files here

| File | What it is |
|---|---|
| `SETUP.md` | This guide |
| `machine-workspace.json` | The `mimir` workspace block for `%USERPROFILE%\.heimdall\machine.json` on Windows |
| `requirements-gpu.txt` | Pinned GPU Python stack |
| `smoke_test.py` | Step 7 |
| `tasks/*.sh` | The six Heimdall tasks, plus `common.sh` |
| `jobs/` | One `<name>.sh` per runnable job, and the rules a job must follow (`jobs/README.md`) |
| `graceful.py` | `StopOnSigterm` callback and `last_checkpoint()` for HF Trainer jobs |
| `dummy_job.py` | GPU-free test job |
| `local.env` | Not committed. Machine-local settings, such as `MIMIR_GPU_BUSY_MIB` and `MIMIR_STOP_WAIT` |

## Sources

1. NVIDIA, *CUDA on WSL User Guide* (CUDA 13.5 docs, updated 2026-11-02): install only the Windows driver, and "Do not install any Linux display driver in WSL"; don't install `cuda-drivers`/`nvidia-open`; "`nvidia-smi` has a limited feature set on WSL"; GPU utilization and active compute process queries are not supported; `nvidia-smi` is in `/usr/lib/wsl/lib`. <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
2. NVIDIA, *CUDA Toolkit Release Notes* (13.5): Table 2 maps CUDA 13.2 to driver branch R595 and 13.0 to R580; Table 3 gives "13.x >= 580" for minor-version compatibility; CUDA 12.8 Update 1 needs Windows ≥ 572.61. <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>. See also <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>.
3. Microsoft, *Advanced settings configuration in WSL*: in `.wslconfig` `[general]`, `instanceIdleTimeout` defaults to 15000 ms, "Set to -1 to disable auto shutdown". <https://learn.microsoft.com/en-us/windows/wsl/wsl-config>
4. PyTorch Forums, *PyTorch support for sm120*: a maintainer says PyTorch 2.7.0 added Blackwell support in wheels built with CUDA 12.8, and that any recent release built with CUDA 12.8+ supports it. <https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099>. `get_arch_list()` in steps 5 and 7 is the real check.
5. PyPI metadata, checked 2026-11-23: `torch 2.15.0` requires `cuda-toolkit[…]==13.2.2` on Linux, and its latest version is 2.15.0. transformers 5.24.0, peft 0.22.0, trl 1.18.2 and accelerate 1.16.0 require Python ≥ 3.11. <https://pypi.org/project/torch/>. The full pinned set also resolves for `x86_64-unknown-linux-gnu`/Python 3.12 (`uv pip compile`).
6. PyTorch wheel index, checked 2026-11-23: `https://download.pytorch.org/whl/cu132/torch/` carries `torch-2.15.0+cu132` (cu130 tops out at 2.14.1). uv's PyTorch guide documents `uv pip install torch --torch-backend=cu130`-style backend selection. <https://docs.astral.sh/uv/guides/integration/pytorch/>
7. Hugging Face, *google/gemma-3-4b-it*: "publicly accessible, but you have to accept the conditions to access its files". <https://huggingface.co/google/gemma-3-4b-it>
8. Hugging Face, *Command Line Interface* guide: `hf auth login` and `hf auth whoami`. <https://huggingface.co/docs/huggingface_hub/guides/cli>

# Jobs

Each `<name>.sh` here is one job that `run <name>` can start. The rules:

- **One idempotent command.** It runs from the repo root with `env3.12` first on PATH. It writes everything to `runs/<name>/`, and a second run resumes from the latest checkpoint or shard there.
- **`exec` the command.** Then `stop` sends SIGTERM straight to Python, not to a bash wrapper that would hold the signal.
- **On SIGTERM, checkpoint and exit.** For an HF `Trainer`, add `StopOnSigterm()` from `graceful.py` (jobs have `infra/gaming-pc` on PYTHONPATH) to its callbacks and pass `resume_from_checkpoint` when a checkpoint exists. Its docstring has an example. Also checkpoint every 15–20 minutes (`save_steps`), in case the PC loses power.
- **Write `runs/<name>/DONE` on success.** `run` won't start a job that has one.
- **Keep results small.** `push-results` copies `.json .jsonl .csv .md .txt .png` files under 1 MB from `runs/<name>/` (not from `checkpoint*` folders).

`dummy.sh` follows all these rules without using the GPU. Use it to try the runner.

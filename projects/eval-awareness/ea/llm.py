"""One small interface for the cheap LLM steps (rewrites and judges).

Two backends:
- "claude-code": headless Claude Code (`claude -p`), which draws on a Claude
  subscription. Used for the original runs.
- "anthropic": the anthropic SDK with ANTHROPIC_API_KEY, so others can
  reproduce the work without Claude Code.

Every result is cached on disk, keyed by a hash of (model, system, prompt,
schema, sample index), so reruns are free and interrupted jobs resume.
Neither backend exposes temperature control here, so outputs are sampled
non-deterministically; use `sample` > 0 to draw a repeat for agreement checks.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from .paths import LLM_CACHE

MODEL_ALIAS = "haiku"
SDK_MODEL_ID = "claude-haiku-5-5"
CALL_LOG = LLM_CACHE / "calls.jsonl"
_log_lock = threading.Lock()


def _key(model: str, system: str, prompt: str, schema: dict | None, sample: int) -> str:
    blob = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "schema": schema, "sample": sample},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return LLM_CACHE / key[:2] / f"{key}.json"


def _backend() -> str:
    forced = os.environ.get("EA_LLM_BACKEND")
    if forced:
        return forced
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "claude-code"


def _call_claude_code(system: str, prompt: str, schema: dict | None) -> dict:
    cmd = [
        "claude", "-p",
        "--model", MODEL_ALIAS,
        "--system-prompt", system,
        "--tools", "",
        "--setting-sources", "",
        "--no-session-persistence",
        "--output-format", "json",
    ]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr[-500:]}")
    j = json.loads(proc.stdout)
    if j.get("is_error"):
        raise RuntimeError(f"claude -p error: {j.get('subtype')} {j.get('result', '')[:300]}")
    model_ids = list(j.get("modelUsage", {}).keys())
    out = j.get("structured_output") if schema is not None else j.get("result")
    if schema is not None and out is None:
        out = json.loads(j["result"])
    return {"output": out, "model_id": ",".join(model_ids), "cost_usd": j.get("total_cost_usd", 0.0)}


def _call_anthropic(system: str, prompt: str, schema: dict | None) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    kwargs: dict[str, Any] = dict(
        model=SDK_MODEL_ID,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if schema is not None:
        kwargs["tools"] = [{"name": "answer", "description": "Return the answer.", "input_schema": schema}]
        kwargs["tool_choice"] = {"type": "tool", "name": "answer"}
    msg = client.messages.create(**kwargs)
    if schema is not None:
        out = next(b.input for b in msg.content if b.type == "tool_use")
    else:
        out = "".join(b.text for b in msg.content if b.type == "text")
    return {"output": out, "model_id": msg.model, "cost_usd": None}


def llm_call(system: str, prompt: str, schema: dict | None = None, sample: int = 0,
             retries: int = 3) -> dict:
    """Return {"output", "model_id", "cost_usd", "cached"}. `output` is a str, or a dict if `schema`."""
    key = _key(MODEL_ALIAS, system, prompt, schema, sample)
    path = _cache_path(key)
    if path.exists():
        rec = json.loads(path.read_text())
        rec["cached"] = True
        return rec
    backend = _backend()
    fn = _call_claude_code if backend == "claude-code" else _call_anthropic
    err: Exception | None = None
    for attempt in range(retries):
        try:
            rec = fn(system, prompt, schema)
            break
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"llm_call failed after {retries} tries") from err
    rec["backend"] = backend
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.replace(path)
    with _log_lock:
        with CALL_LOG.open("a") as f:
            f.write(json.dumps({"t": time.time(), "key": key, "model_id": rec["model_id"],
                                "cost_usd": rec["cost_usd"], "backend": backend}) + "\n")
    rec["cached"] = False
    return rec


def llm_map(fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 8) -> list:
    """Run fn over items with a thread pool, preserving order. Failed items return None."""
    items = list(items)

    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001
            print(f"[llm_map] item failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(safe, items))


def call_summary() -> dict:
    """Count of uncached calls made so far and their summed cost (USD equivalent)."""
    if not CALL_LOG.exists():
        return {"calls": 0, "cost_usd": 0.0, "model_ids": []}
    n, cost, ids = 0, 0.0, set()
    for line in CALL_LOG.read_text().splitlines():
        r = json.loads(line)
        n += 1
        cost += r.get("cost_usd") or 0.0
        ids.add(r.get("model_id"))
    return {"calls": n, "cost_usd": round(cost, 4), "model_ids": sorted(i for i in ids if i)}

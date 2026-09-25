"""
[I06] LLM backend - one call() for all 7 LLM agents. No API keys.

    result = llm.call(stage="analyzer", tier="heavy", system=..., user=...,
                      schema=AnalysisModel, mock={...}, run_id=ctx["run_id"])

Backends:
  claude   DEFAULT. Claude does the stage inside chat/Cowork, no API.
           The stage writes data/runs/<run_id>/<stage>.prompt.md and pauses the run
           (HandoffPending). Claude reads the prompt, writes <stage>.response.json
           next to it, and the run resumes. The runner skill (R01) automates this loop.
  mock     returns the agent's own mock dict, validated against the schema.
           Free, instant, no network. Used by every test.
  ollama   OPTIONAL, for later. Calls a local Ollama at OLLAMA_HOST with the JSON
           schema as the required output format. Not used unless you route to it.

Every backend's output is validated against the same Pydantic schema, so an
agent never sees malformed data. Ollama gets exactly one repair turn: it is
shown its own bad output plus the validation errors and asked to fix it.

Every call is logged to data/runs/<run_id>/llm_log.jsonl.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from config.settings import BACKEND, OLLAMA_HOST, ROOT, RUNS_DIR
from core.base_agent import AgentError, HandoffPending

MODELS_FILE = ROOT / "config" / "models.yaml"
TIERS = ("heavy", "mid")


class LLMError(AgentError):
    """An LLM call could not produce valid output. Message says why."""


# ---------- routing ----------

def load_config() -> dict:
    return yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8")) or {}


def backend_for(stage: str, cfg: dict | None = None) -> str:
    if BACKEND == "mock":
        return "mock"
    cfg = cfg if cfg is not None else load_config()
    return (cfg.get("stages") or {}).get(stage, BACKEND)


# ---------- validation ----------

def _parse(raw: str | dict, schema: type[BaseModel]) -> dict:
    """JSON text or dict -> validated plain dict. Raises LLMError with readable reasons."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        raise LLMError(f"not valid JSON: {e.msg} at line {e.lineno} col {e.colno}")
    try:
        return schema.model_validate(data).model_dump()
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or '(root)'}: {err['msg']}"
                             for err in e.errors()[:8])
        raise LLMError(f"does not match schema: {problems}")


def _log(run_id: str, entry: dict) -> None:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "llm_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------- backends ----------

def _mock(stage: str, schema: type[BaseModel], mock: Any) -> tuple[dict, dict]:
    if mock is None:
        raise LLMError(f"{stage}: no mock output provided for mock mode")
    return _parse(mock, schema), {}


def handoff_paths(run_id: str, stage: str) -> tuple[Path, Path]:
    d = RUNS_DIR / run_id
    return d / f"{stage}.prompt.md", d / f"{stage}.response.json"


def _rel(p: Path) -> str:
    """Project-relative path: the same string works on Windows, in Cowork and in chat."""
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _handoff(stage: str, system: str, user: str, schema: type[BaseModel],
             run_id: str) -> tuple[dict, dict]:
    prompt_path, response_path = handoff_paths(run_id, stage)
    if response_path.exists():
        try:
            return _parse(response_path.read_text(encoding="utf-8"), schema), \
                {"response_file": response_path.name}
        except LLMError as e:
            raise LLMError(f"{response_path.name} {e}. Fix the file and re-run.")

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(
        f"# Claude stage: {stage}\n\n"
        f"Run `{run_id}`. You are doing this pipeline stage yourself. Follow the System\n"
        f"and Task sections exactly, then save ONLY the JSON result to:\n\n"
        f"`{_rel(response_path)}`\n\n"
        "No markdown fences, no commentary. It must validate against the schema below.\n\n"
        f"## System\n\n{system}\n\n"
        f"## Task\n\n{user}\n\n"
        f"## Output schema (JSON Schema)\n\n```json\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}\n```\n",
        encoding="utf-8")
    raise HandoffPending(f"waiting for Claude: prompt at {_rel(prompt_path)}, "
                         f"save the answer as {response_path.name}")


def _ollama_request(payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _ollama(stage: str, tier: str, system: str, user: str,
            schema: type[BaseModel], cfg: dict) -> tuple[dict, dict]:
    model = cfg["tiers"][tier]["ollama_model"]
    ocfg = cfg.get("ollama", {})
    retries, timeout = ocfg.get("retries", 3), ocfg.get("timeout_s", 300)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    meta = {"model": model, "attempts": 0, "repaired": False, "tokens_in": 0, "tokens_out": 0}

    def ask(msgs: list[dict]) -> str:
        payload = {"model": model, "messages": msgs, "stream": False,
                   "format": schema.model_json_schema(),
                   "options": {"temperature": ocfg.get("temperature", 0.4)}}
        for attempt in range(retries):
            meta["attempts"] += 1
            try:
                out = _ollama_request(payload, timeout)
                meta["tokens_in"] += out.get("prompt_eval_count", 0)
                meta["tokens_out"] += out.get("eval_count", 0)
                return out["message"]["content"]
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:200]
                raise LLMError(f"Ollama returned HTTP {e.code}: {body} "
                               f"(is the model '{model}' pulled? run: ollama pull {model})")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == retries - 1:
                    raise LLMError(f"Ollama not reachable at {OLLAMA_HOST} after {retries} "
                                   f"tries ({e}). Is Ollama running?")
                time.sleep(2 ** attempt)
        raise LLMError("unreachable")  # pragma: no cover

    raw = ask(messages)
    try:
        return _parse(raw, schema), meta
    except LLMError as first:
        meta["repaired"] = True                    # one repair turn, then give up
        fix = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your output was rejected: {first}. "
                                        "Return corrected JSON only, matching the schema."}]
        try:
            return _parse(ask(fix), schema), meta
        except LLMError as second:
            raise LLMError(f"{stage}: invalid output after 1 repair turn: {second}")


# ---------- public entry point ----------

def call(stage: str, tier: str, system: str, user: str, schema: type[BaseModel],
         mock: Any = None, run_id: str = "adhoc") -> dict:
    if tier not in TIERS:
        raise LLMError(f"unknown tier {tier!r}; use one of {TIERS}")
    cfg = load_config()
    backend = backend_for(stage, cfg)
    t0 = time.perf_counter()
    entry = {"stage": stage, "tier": tier, "backend": backend,
             "prompt_chars": len(system) + len(user)}
    try:
        if backend == "mock":
            result, meta = _mock(stage, schema, mock)
        elif backend == "claude":
            result, meta = _handoff(stage, system, user, schema, run_id)
        elif backend == "ollama":
            result, meta = _ollama(stage, tier, system, user, schema, cfg)
        else:
            raise LLMError(f"unknown backend {backend!r}")
        entry.update(meta, ok=True, response_chars=len(json.dumps(result)))
        return result
    except HandoffPending:
        entry.update(ok=None, waiting=True)
        raise
    except LLMError as e:
        entry.update(ok=False, error=str(e)[:300])
        raise
    finally:
        entry["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        _log(run_id, entry)


# ---------- manual checkpoint ----------

class _Demo(BaseModel):
    summary: str
    keywords: list[str]


if __name__ == "__main__":
    print(f"backend for 'demo': {backend_for('demo')}")
    try:
        out = call(stage="demo", tier="mid", run_id="i06-check",
                   system="You summarise code repositories in one sentence.",
                   user="Repo: a deliberately vulnerable RAG app for testing prompt injection. "
                        "Return a one-sentence summary and 3 keywords.",
                   schema=_Demo,
                   mock={"summary": "A lab RAG app built to be attacked.",
                         "keywords": ["rag", "prompt-injection", "lab"]})
        print(json.dumps(out, indent=2))
    except HandoffPending as e:
        print(f"WAITING  {e}")
    except LLMError as e:
        print(f"FAILED   {e}")
    print(f"log      {(RUNS_DIR / 'i06-check' / 'llm_log.jsonl').as_posix()}")
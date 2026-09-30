"""
[I06] LLM backend - one call() for all 7 LLM agents. No API keys.

    result = llm.call(stage="analyzer", tier="heavy", system=..., user=...,
                      schema=AnalysisModel, mock={...}, run_id=ctx["run_id"])

Backends:
  claude   DEFAULT. Claude does the stage inside chat/Cowork, no API.
           The stage writes data/runs/<run_id>/<stage>.prompt.md and pauses the run
           (HandoffPending). Claude reads the prompt, writes <stage>.response.json
           next to it, and the run resumes. The runner skill (R01) automates this loop.
  claude_cli
           AUTOMATIC. Runs the local Claude Code CLI ("claude -p") for the stage, logged
           in with your Claude plan (no API key). Same prompt as the handoff, answer
           streamed live to <stage>.live.txt, saved as <stage>.response.json. One repair
           turn if the JSON does not validate. Tools are disabled for these calls and they
           run in an empty temp folder, so Claude only answers; it cannot touch files.
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
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from config.settings import BACKEND, CLAUDE_BIN, OLLAMA_HOST, ROOT, RUNS_DIR
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
        f"# Content OS stage: {stage} (run `{run_id}`)\n\n"
        "Answer this ONE stage only. Do not start, resume or run any pipeline, skill or tool\n"
        "to do it: just follow the System and Task sections and produce the JSON result.\n"
        "- In a normal chat: reply with ONLY the JSON object.\n"
        f"- With file access to this project: save ONLY the JSON to `{_rel(response_path)}`.\n\n"
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


# ---------- claude_cli: Claude Code on this machine ----------

CLI_SYSTEM = ("You are one stage of a content pipeline. Follow the instructions in the "
              "message exactly and reply with a single JSON object and nothing else.")
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0     # no console pop-ups on Windows


def claude_bin() -> str | None:
    """Full path of the Claude Code CLI, or None if it is not installed."""
    found = shutil.which(CLAUDE_BIN)
    return found or (CLAUDE_BIN if Path(CLAUDE_BIN).is_file() else None)


def _cli_prompt(system: str, user: str, schema: type[BaseModel]) -> str:
    return (f"## System\n\n{system}\n\n## Task\n\n{user}\n\n"
            f"## Output\n\nReply with ONLY one JSON object that validates against this JSON "
            f"Schema. No prose before or after it, no code fences.\n\n"
            f"{json.dumps(schema.model_json_schema(), indent=1)}\n")


def _extract_json(text: str) -> str:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
    return t


def _run_claude(prompt: str, model: str, timeout: int, live: Path) -> str:
    exe = claude_bin()
    if not exe:
        raise LLMError("Claude Code CLI not found. Install it (npm install -g @anthropic-ai/claude-code), "
                       "run `claude` once in a terminal to log in, then retry. "
                       "Or switch the dashboard to Manual mode.")
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--tools", "", "--model", model, "--no-session-persistence", "--strict-mcp-config",
           "--system-prompt", CLI_SYSTEM]
    killed = threading.Event()
    with tempfile.TemporaryDirectory(prefix="cos_cli_", ignore_cleanup_errors=True) as cwd, \
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err, text=True, encoding="utf-8", errors="replace",
                                creationflags=NO_WINDOW)
        timer = threading.Timer(timeout, lambda: (killed.set(), proc.kill()))
        timer.start()
        result = None
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
            with open(live, "a", encoding="utf-8") as out:
                for line in proc.stdout:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "stream_event":
                        delta = (ev.get("event") or {}).get("delta") or {}
                        piece = delta.get("text") or delta.get("thinking")
                        if piece:
                            out.write(piece)
                            out.flush()
                    elif ev.get("type") == "result":
                        result = ev
            proc.wait()
        finally:
            timer.cancel()
        err.seek(0)
        stderr = err.read()[-400:]
    if killed.is_set():
        raise LLMError(f"Claude Code took longer than {timeout}s (claude_cli.timeout_s in models.yaml)")
    if result is None:
        raise LLMError(f"Claude Code exited ({proc.returncode}) without an answer. "
                       f"If it is not logged in, run `claude` once in a terminal. {stderr.strip()}")
    if result.get("is_error"):
        raise LLMError(f"Claude Code error: {str(result.get('result') or result.get('subtype'))[:300]}")
    return result.get("result") or ""


def _claude_cli(stage: str, tier: str, system: str, user: str, schema: type[BaseModel],
                cfg: dict, run_id: str) -> tuple[dict, dict]:
    prompt_path, response_path = handoff_paths(run_id, stage)
    if response_path.exists():                 # already answered (resume / re-run): reuse it
        return _handoff(stage, system, user, schema, run_id)
    model = (cfg.get("tiers", {}).get(tier) or {}).get("claude_model", "opus")
    timeout = int((cfg.get("claude_cli") or {}).get("timeout_s", 600))
    prompt = _cli_prompt(system, user, schema)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    live = prompt_path.with_name(f"{stage}.live.txt")
    live.write_text("", encoding="utf-8")
    meta = {"model": model, "repaired": False}

    raw = _run_claude(prompt, model, timeout, live)
    try:
        result = _parse(_extract_json(raw), schema)
    except LLMError as first:
        meta["repaired"] = True
        with open(live, "a", encoding="utf-8") as out:
            out.write(f"\n\n--- answer rejected ({first}); asking for a fix ---\n\n")
        fix = (f"{prompt}\n\n## Your previous answer was rejected\n\n{first}\n\n"
               f"Previous answer:\n{raw[:20000]}\n\nReturn the corrected JSON object only.")
        try:
            result = _parse(_extract_json(_run_claude(fix, model, timeout, live)), schema)
        except LLMError as second:
            raise LLMError(f"{stage}: invalid output after 1 repair turn: {second}")
    response_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result, meta


# ---------- public entry point ----------

def call(stage: str, tier: str, system: str, user: str, schema: type[BaseModel],
         mock: Any = None, run_id: str = "adhoc") -> dict:
    if tier not in TIERS:
        raise LLMError(f"unknown tier {tier!r}; use one of {TIERS}")
    cfg = load_config()
    backend = backend_for(stage, cfg)
    t0 = time.perf_counter()
    entry = {"stage": stage, "tier": tier, "backend": backend,
             "prompt_chars": len(system) + len(user),
             "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        if backend == "mock":
            result, meta = _mock(stage, schema, mock)
        elif backend == "claude":
            result, meta = _handoff(stage, system, user, schema, run_id)
        elif backend == "claude_cli":
            result, meta = _claude_cli(stage, tier, system, user, schema, cfg, run_id)
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

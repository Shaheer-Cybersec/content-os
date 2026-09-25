"""
[I02] base agent - the contract all 11 agents follow.

An agent declares:
  agent_id  tracking tag, e.g. "A01"
  name      stage name, e.g. "repo_ingest"
  requires  context keys it needs before it can run
  produces  the ONE context key it writes its output into
  uses_llm  True for the 7 LLM agents, False for the 4 static ones

An agent implements run(ctx) -> dict.
The orchestrator only ever calls execute(ctx), which:
  - checks inputs exist
  - times the run
  - catches every error and returns a StageRecord (never crashes the pipeline)
  - rejects output that is not JSON-serializable
"""
from __future__ import annotations

import json
import time

from core.schemas import StageRecord


class AgentError(Exception):
    """The agent could not do its job. The message must say why, in plain words."""


class SkipStage(Exception):
    """The agent decided it has nothing to do (e.g. A09 reviser when there are no flags)."""


class BaseAgent:
    agent_id: str = "A00"
    name: str = "base"
    requires: tuple[str, ...] = ()
    produces: str = ""
    uses_llm: bool = False

    def run(self, ctx: dict) -> dict:
        raise NotImplementedError(f"{self.name} must implement run()")

    def execute(self, ctx: dict) -> dict:
        """Run the agent safely. Returns a StageRecord as a plain dict."""
        missing = [k for k in self.requires if k not in ctx]
        if missing:
            return self._record("failed", 0, f"missing context keys: {missing}")

        t0 = time.perf_counter()
        status, error = "failed", None
        try:
            output = self.run(ctx)
            # Guard: output must survive json.dumps. Raw bytes (e.g. embedding
            # blobs) corrupted the run context in the first build.
            json.dumps(output)
            ctx[self.produces] = output
            status = "success"
        except SkipStage as e:
            status, error = "skipped", str(e) or None
        except AgentError as e:
            error = str(e)
        except TypeError as e:
            error = f"output not JSON-serializable: {e}"
        except Exception as e:  # a real bug: keep the type so it's debuggable
            error = f"{type(e).__name__}: {e}"
        ms = int((time.perf_counter() - t0) * 1000)
        return self._record(status, ms, error)

    def _record(self, status: str, ms: int, error: str | None) -> dict:
        return StageRecord(id=self.agent_id, agent=self.name, status=status,
                           duration_ms=ms, error=error).model_dump()
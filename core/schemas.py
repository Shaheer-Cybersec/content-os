"""
[I02] schemas - the data shapes every agent agrees on.

Evidence is the spine of Content OS: every factual claim in a post must carry
the file and commit it came from. I05 later checks those exist in the repo.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Evidence(BaseModel):
    """One claim, bound to where it came from in the repo."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(min_length=1)        # "Retrieval uses a top-k of 5"
    source_path: str = Field(min_length=1)  # "app/retriever.py"
    source_ref: str = Field(min_length=1)   # commit SHA, e.g. "38ae3eb"

    @field_validator("source_path")
    @classmethod
    def repo_relative(cls, v: str) -> str:
        v = v.replace("\\", "/")            # Windows paths -> repo style
        if v.startswith("/") or ":" in v:
            raise ValueError("source_path must be relative to the repo root")
        if ".." in v.split("/"):
            raise ValueError("source_path must not contain '..'")
        return v


class StageRecord(BaseModel):
    """What execute() reports for one agent run. Written to the run log."""
    id: str                                  # "A01"
    agent: str                               # "repo_ingest"
    status: Literal["success", "failed", "skipped"]
    duration_ms: int = 0
    error: Optional[str] = None
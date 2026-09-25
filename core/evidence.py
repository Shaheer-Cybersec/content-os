"""
[I05] evidence validator - catches invented files and commits, with no LLM involved.

Every claim a model writes must cite a real file (source_path) and a real
commit (source_ref) from the repo that A01 ingested. This module checks both
against A01's output. A model can make up a convincing sentence; it cannot make
a file appear in the git tree.

Fail-closed: if something cannot be verified, it counts as invalid.
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from core.schemas import Evidence

SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _known_paths(repo: dict) -> set[str]:
    return {f["path"] for f in repo.get("tree", [])}


def _known_shas(repo: dict) -> list[str]:
    shas = [c["sha"].lower() for c in repo.get("commits", [])]
    if repo.get("head_sha"):
        shas.append(repo["head_sha"].lower())
    return shas


def check_one(raw: dict, paths: set[str], shas: list[str], truncated: bool) -> str | None:
    """Return None if the claim is valid, otherwise the reason it is not."""
    try:
        ev = Evidence(**raw)
    except (ValidationError, TypeError) as e:
        first = e.errors()[0]["msg"] if isinstance(e, ValidationError) else str(e)
        return f"malformed evidence: {first}"

    if ev.source_path not in paths:
        if truncated:
            return f"cannot verify path (repo tree was truncated): {ev.source_path}"
        return f"path not in repo: {ev.source_path}"

    ref = ev.source_ref.lower()
    if not SHA_RE.match(ref):
        return f"source_ref is not a commit SHA: {ev.source_ref}"
    if not any(s.startswith(ref) for s in shas):
        return f"unknown commit: {ev.source_ref}"
    return None


def validate(claims: list[dict], repo: dict) -> dict:
    """Split claims into valid and invalid.
    Returns {"ok": bool, "valid": [...], "invalid": [{"evidence": ..., "reason": ...}]}."""
    paths, shas = _known_paths(repo), _known_shas(repo)
    truncated = bool(repo.get("tree_truncated"))
    valid, invalid = [], []
    for raw in claims:
        reason = check_one(raw, paths, shas, truncated)
        if reason is None:
            valid.append(raw)
        else:
            invalid.append({"evidence": raw, "reason": reason})
    return {"ok": not invalid, "valid": valid, "invalid": invalid}

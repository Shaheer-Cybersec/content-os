"""
[I01] settings - every path and setting in one place.

Rules:
  - No other module hardcodes a path. Everything imports from here.
  - Priority: real environment variable > .env file > default below.
  - There are no API keys in this project. BACKEND picks where LLM stages run.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root = the folder that contains config/
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader: KEY=VALUE lines, '#' comments.
    setdefault() means a real environment variable always wins."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# ---------- paths ----------
DATA_DIR = ROOT / "data"                  # runtime state, gitignored
CACHE_DIR = DATA_DIR / "repo_cache"       # A01 output, one JSON per commit SHA
RUNS_DIR = DATA_DIR / "runs"              # run state + handoff prompt/response files
OBSIDIAN_DIR = DATA_DIR / "obsidian"      # A11 read-only mirror for browsing
DB_PATH = DATA_DIR / "content_os.db"      # I03 SQLite database
OUTPUTS_DIR = ROOT / "outputs"            # A10 finished post packages
VOICE_DIR = ROOT / "memory" / "voice"     # L01 your real posts, gitignored

# ---------- LLM backend ----------
BACKENDS = ("mock", "handoff", "ollama")
BACKEND = os.getenv("CONTENTOS_BACKEND", "mock").strip().lower()
if BACKEND not in BACKENDS:
    raise ValueError(
        f"CONTENTOS_BACKEND={BACKEND!r} is not valid. Use one of: {', '.join(BACKENDS)}"
    )

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")

# Create runtime folders on import so no module has to check first
for _d in (DATA_DIR, CACHE_DIR, RUNS_DIR, OBSIDIAN_DIR, OUTPUTS_DIR, VOICE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"ROOT         {ROOT}")
    print(f"BACKEND      {BACKEND}")
    print(f"OLLAMA_HOST  {OLLAMA_HOST}")
    print(f"DB_PATH      {DB_PATH}")
    for d in (DATA_DIR, CACHE_DIR, RUNS_DIR, OBSIDIAN_DIR, OUTPUTS_DIR, VOICE_DIR):
        print(f"{'ok' if d.is_dir() else 'MISSING':<12} {d.relative_to(ROOT)}")
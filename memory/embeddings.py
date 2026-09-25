"""
[I04] embeddings - turn text into vectors so A04 can spot duplicate angles.

Fail-closed rule (defect #5 in the first build):
  If the model can't load, max_similarity() returns None - never a guessed score.
  A04 then marks angles 'unchecked' instead of accepting or rejecting at random.

Vectors are plain lists of floats while in Python, and float32 bytes (BLOB)
inside SQLite. They never go into the run context.
"""
from __future__ import annotations

import math
from array import array

from config.settings import EMBED_MODEL, EMBEDDINGS_ENABLED

_model = None
_why_unavailable: str | None = None


def _load():
    """Load the model once, on first use. Returns None if it can't."""
    global _model, _why_unavailable
    if _model is not None or _why_unavailable is not None:
        return _model
    if not EMBEDDINGS_ENABLED:
        _why_unavailable = "disabled by CONTENTOS_EMBEDDINGS=off"
        return None
    try:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy
        _model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:  # not installed, no internet on first download, etc.
        _why_unavailable = f"{type(e).__name__}: {e}"
    return _model


def available() -> bool:
    return _load() is not None


def why_unavailable() -> str | None:
    _load()
    return _why_unavailable


def embed(text: str) -> list[float] | None:
    """Unit-length vector for text, or None if the model is unavailable."""
    model = _load()
    if model is None:
        return None
    return model.encode(text, normalize_embeddings=True).tolist()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity: 1.0 = same meaning, ~0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def max_similarity(text: str, past: list[list[float]]) -> float | None:
    """Highest similarity between text and any past vector.
    0.0 when there is nothing to compare against. None when we cannot check."""
    if not past:
        return 0.0
    v = embed(text)
    if v is None:
        return None
    return max(cosine(v, p) for p in past)


def to_blob(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def from_blob(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


if __name__ == "__main__":
    pairs = [
        ("near-duplicate",
         "I built a deliberately vulnerable RAG app to test prompt injection.",
         "I built an intentionally vulnerable RAG application to test prompt injection."),
        ("same topic, different angle",
         "I built a deliberately vulnerable RAG app to test prompt injection.",
         "Why top-k retrieval size changes how easy indirect injection is."),
        ("unrelated",
         "I built a deliberately vulnerable RAG app to test prompt injection.",
         "My favourite biryani spot in Rawalpindi closes at midnight."),
    ]
    if not available():
        print(f"embeddings OFF -> {why_unavailable()}")
        print(f"max_similarity returns: {max_similarity('anything', [[1.0, 0.0]])}")
    else:
        for label, a, b in pairs:
            s = cosine(embed(a), embed(b))
            print(f"{s:.2f}  {label}")
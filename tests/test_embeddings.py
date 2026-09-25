"""[I04] checkpoint tests. Math and fail-closed tests need no model; the last test
uses the real model and is skipped if sentence-transformers is not installed."""
import pytest

from memory import embeddings as E


def test_cosine_basics():
    assert E.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert E.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert E.cosine([0, 0], [1, 0]) == 0.0

def test_blob_roundtrip():
    v = [0.25, -0.5, 1.0]
    assert E.from_blob(E.to_blob(v)) == pytest.approx(v)

def test_nothing_to_compare_is_zero():
    assert E.max_similarity("anything", []) == 0.0

def test_fail_closed_when_model_missing(monkeypatch):
    monkeypatch.setattr(E, "_load", lambda: None)
    assert E.embed("x") is None
    assert E.max_similarity("x", [[1.0, 0.0]]) is None   # None, never a fake score

def test_real_model_ranks_duplicates_highest():
    pytest.importorskip("sentence_transformers")
    if not E.available():
        pytest.skip(E.why_unavailable())
    base = "I built a deliberately vulnerable RAG app to test prompt injection."
    dup = E.embed("I built an intentionally vulnerable RAG application to test prompt injection.")
    other = E.embed("My favourite biryani spot in Rawalpindi closes at midnight.")
    assert E.max_similarity(base, [dup]) > 0.82
    assert E.max_similarity(base, [other]) < 0.30
"""Shared test setup: every test runs with the mock LLM backend.
Real runs default to Claude; tests must never write handoff prompts or wait on anyone."""
import pytest

from core import llm


@pytest.fixture(autouse=True)
def mock_backend(monkeypatch):
    monkeypatch.setattr(llm, "BACKEND", "mock")
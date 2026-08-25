"""Shared fixtures.

The backend lives one directory up; tests run with the backend dir on
``sys.path`` (configured via ``pytest.ini`` rootdir) so ``import app.*`` works.
The ``client`` fixture redirects storage into a tmp dir so API tests never
touch the real ``backend/data`` tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A FastAPI TestClient with storage redirected to ``tmp_path``."""
    from fastapi.testclient import TestClient

    from app import main, storage
    from app.llm_agent import LlmUnavailable

    def offline_llm(*args, **kwargs):
        del args, kwargs
        raise LlmUnavailable("LLM disabled by the hermetic API test fixture")

    projects = tmp_path / "projects"
    uploads = tmp_path / "uploads"
    projects.mkdir()
    uploads.mkdir()

    # storage.* functions read these module globals; main imported them by value.
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(storage, "PROJECTS_DIR", projects, raising=True)
    monkeypatch.setattr(storage, "UPLOADS_DIR", uploads, raising=True)
    monkeypatch.setattr(main, "PROJECTS_DIR", projects, raising=True)
    monkeypatch.setattr(main, "UPLOADS_DIR", uploads, raising=True)
    monkeypatch.setattr(main, "plan_operations_llm", offline_llm, raising=True)

    with TestClient(main.app) as test_client:
        yield test_client

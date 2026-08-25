"""Persistence-layer coverage for ``app.storage`` (redirected to tmp)."""
from __future__ import annotations

import pytest

from app import storage
from app.models import Operation


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point storage at an isolated tmp tree for the duration of a test."""
    projects = tmp_path / "projects"
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "PROJECTS_DIR", projects)
    monkeypatch.setattr(storage, "UPLOADS_DIR", uploads)
    return storage


def test_create_project_writes_files_and_default_ir(store):
    project = store.create_project("demo")
    assert project.name == "demo"
    pdir = store.PROJECTS_DIR / project.project_id
    assert (pdir / "project.json").exists()
    assert (pdir / "output.dxf").exists()   # write_exports ran
    assert (pdir / "preview.svg").exists()
    assert len(project.ir.entities) == 4    # default_ir baseline


def test_save_then_load_round_trips(store):
    project = store.create_project("demo")
    project.history.append(Operation(operation="delete_entity", entity_id="hole_1"))
    store.save_project(project)

    loaded = store.load_project(project.project_id)
    assert loaded.project_id == project.project_id
    assert [op.operation for op in loaded.history] == ["delete_entity"]


def test_save_updates_timestamp(store):
    project = store.create_project("demo")
    first = project.updated_at
    reloaded = store.save_project(store.load_project(project.project_id))
    assert reloaded.updated_at >= first


def test_load_missing_project_raises(store):
    with pytest.raises(FileNotFoundError):
        store.load_project("does-not-exist")


def test_list_projects_returns_newest_first(store):
    a = store.create_project("a")
    b = store.create_project("b")
    listed = store.list_projects()
    ids = {p.project_id for p in listed}
    assert {a.project_id, b.project_id} <= ids


def test_list_projects_skips_corrupt_entries(store):
    good = store.create_project("good")
    bad_dir = store.PROJECTS_DIR / "broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "project.json").write_text("{ not valid json", encoding="utf-8")

    listed = store.list_projects()
    ids = {p.project_id for p in listed}
    assert good.project_id in ids          # corrupt entry is skipped, not fatal
    assert "broken" not in ids

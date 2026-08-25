from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .cad_layers import normalize_cad_layers
from .exporter import export_dxf, export_svg
from .models import DrawingIR, ProjectState, default_ir

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROJECTS_DIR = DATA_DIR / "projects"
UPLOADS_DIR = DATA_DIR / "uploads"


def init_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def load_project(project_id: str) -> ProjectState:
    payload = json.loads(project_file(project_id).read_text(encoding="utf-8"))
    project = ProjectState.model_validate(payload)
    normalize_cad_layers(project.ir)
    _sync_mechanical_snapshots(project)
    return project


def save_project(project: ProjectState) -> ProjectState:
    normalize_cad_layers(project.ir)
    project.updated_at = datetime.now(timezone.utc)
    write_project_file(project)
    write_exports(project)
    return project


def save_project_metadata(project: ProjectState) -> ProjectState:
    """Persist project JSON without regenerating preview/export artifacts."""
    normalize_cad_layers(project.ir)
    project.updated_at = datetime.now(timezone.utc)
    write_project_file(project)
    return project


def write_project_file(project: ProjectState) -> None:
    pdir = project_dir(project.project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    target = project_file(project.project_id)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(target)


def create_project(name: str, ir: DrawingIR | None = None, source_image: str | None = None) -> ProjectState:
    init_dirs()
    project_id = uuid4().hex[:10]
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id=project_id,
        name=name,
        created_at=now,
        updated_at=now,
        source_image=source_image,
        ir=ir or default_ir(),
    )
    return save_project(project)


def list_projects() -> list[ProjectState]:
    init_dirs()
    projects: list[ProjectState] = []
    for path in sorted(PROJECTS_DIR.glob("*/project.json"), reverse=True):
        try:
            project = ProjectState.model_validate(json.loads(path.read_text(encoding="utf-8")))
            normalize_cad_layers(project.ir)
            _sync_mechanical_snapshots(project)
            projects.append(project)
        except Exception:
            continue
    return projects


def write_exports(project: ProjectState) -> None:
    pdir = project_dir(project.project_id)
    export_dxf(project.ir, pdir / "output.dxf", project.mechanical_ir)
    export_svg(project.ir, pdir / "preview.svg")


def _sync_mechanical_snapshots(project: ProjectState) -> None:
    """Hydrate the unified semantic snapshot when loading pre-v1 projects."""
    if project.mechanical_ir.dimensions:
        project.mechanical_dimensions = project.mechanical_ir.dimensions
        return
    if project.mechanical_dimensions:
        project.mechanical_ir.units = project.ir.units
        project.mechanical_ir.dimensions = project.mechanical_dimensions

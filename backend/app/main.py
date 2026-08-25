from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .agent import plan_operations
from .agent_eval import append_agent_eval_report, load_agent_eval_dataset, run_agent_eval
from .agent_runtime import append_agent_task_run, run_agent_task
from .agent_tools import build_default_tool_registry
from .arrow_cv import detect_arrowheads_from_reference
from .cad_ops import apply_operations, ids_diff_payload
from .dimension_benchmark import (
    apply_dimension_correction,
    evaluate_dimension_benchmark,
    seed_dimension_ground_truth,
)
from .dimension_render import render_dimension_binding_arrowheads
from .dimension_semantics import detect_dimension_bindings
from .llm_agent import LlmUnavailable, plan_mechanical_operation_llm, plan_operations_llm
from .mechanical_drive import (
    MechanicalDriveError,
    execute_mechanical_operation,
    is_undo_command,
    plan_mechanical_drive_deterministic,
    undo_last_mechanical_transaction,
)
from .mechanical_edit import EDIT_INTENT_RE, plan_mechanical_dimension_edit, sync_mechanical_dimension_edit
from .mechanical_ir import build_mechanical_drawing_ir
from .models import (
    AgentEvalDatasetResponse,
    AgentEvalRequest,
    AgentEvalResponse,
    AgentTaskRequest,
    AgentTaskResponse,
    AgentToolCatalogResponse,
    AnalyzeResponse,
    CadPipelineRequest,
    CadPipelineResponse,
    CadPipelineStep,
    ChatRequest,
    ChatResponse,
    CreateProjectRequest,
    DiffItem,
    DimensionBenchmarkResponse,
    DimensionBenchmarkSeedRequest,
    DimensionCorrectionRequest,
    DimensionSemanticsResponse,
    ExportPaths,
    OcrRequest,
    OcrResponse,
    Operation,
    ProjectState,
    ReconstructionRegion,
    ScanCadReconstructionResponse,
    ScanPromotionResponse,
    SectionReconstructionResponse,
    SemanticRepairRequest,
    SemanticRepairResponse,
    TableOcrResponse,
    TableReconstructionResponse,
    VectorizerBenchmarkResponse,
    VectorizerToolResult,
    VectorReconstructionResponse,
)
from .ocr import run_project_ocr
from .promote import promote_scan_primitives
from .reconstruct import reconstruct_tables_from_reference
from .reference import _upload_url_to_path, analyze_reference, save_reference_upload
from .scan_cad import reconstruct_scan_cad_from_reference
from .scan_eval import evaluate_scan_structure
from .section_cv import reconstruct_section_from_reference
from .semantic_repair import append_semantic_repair_run, run_semantic_repair_agent
from .storage import (
    PROJECTS_DIR,
    UPLOADS_DIR,
    create_project,
    init_dirs,
    list_projects,
    load_project,
    project_dir,
    save_project,
    save_project_metadata,
)
from .structure_eval import StructureEvalReport, evaluate_structure
from .table_ocr import extract_table_ocr_from_reference
from .table_text import render_table_ocr_cells_into_ir
from .title_block import extract_title_block_cells, render_title_block_cells_into_ir
from .vector_external import export_vector_pdf_assets
from .vector_extract import reconstruct_vector_from_reference
from .vectorizer_benchmark import run_project_vectorizer_benchmark

# Load backend/.env (gitignored) so LLM keys are available without exporting them.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dirs()
    yield


app = FastAPI(title="Vibe CAD MVP", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLACE_DRAWING_OPERATIONS = {"create_plate", "create_spur_gear_drawing"}
AGENT_TOOL_REGISTRY = build_default_tool_registry()


def would_replace_vector_import(project: ProjectState, operations: list[Operation]) -> bool:
    return (
        project.source_kind == "vector_pdf"
        and project.source_file is not None
        and any(operation.operation in REPLACE_DRAWING_OPERATIONS for operation in operations)
    )


def _pipeline_step(name: str, status: str, detail: str = "") -> CadPipelineStep:
    return CadPipelineStep(name=name, status=status, detail=detail)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/projects")
def api_list_projects():
    return list_projects()


@app.post("/api/projects")
def api_create_project(request: CreateProjectRequest):
    project = create_project(request.name)
    if request.prompt.strip():
        operations, _ = plan_operations(request.prompt, project.ir)
        if operations:
            try:
                project.ir, diffs = apply_operations(project.ir, operations)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            project.history.extend(operations)
            project.diffs = diffs
            project = save_project(project)
    return project


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    try:
        return load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@app.get("/api/projects/{project_id}/eval", response_model=StructureEvalReport)
def api_project_eval(project_id: str) -> StructureEvalReport:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return evaluate_structure(project.ir)


@app.get("/api/projects/{project_id}/eval/scan", response_model=StructureEvalReport)
def api_project_scan_eval(project_id: str) -> StructureEvalReport:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    source_image_path = _upload_url_to_path(project.source_image, UPLOADS_DIR) if project.source_image else None
    return evaluate_scan_structure(project.ir, source_image_path)


@app.get("/api/projects/{project_id}/eval/dimensions", response_model=DimensionBenchmarkResponse)
def api_dimension_benchmark(project_id: str) -> DimensionBenchmarkResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return DimensionBenchmarkResponse(project=project, report=evaluate_dimension_benchmark(project))


@app.post("/api/projects/{project_id}/benchmark/dimensions/seed", response_model=DimensionBenchmarkResponse)
def api_seed_dimension_benchmark(
    project_id: str,
    request: DimensionBenchmarkSeedRequest | None = None,
) -> DimensionBenchmarkResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    payload = request or DimensionBenchmarkSeedRequest()
    project = seed_dimension_ground_truth(project, payload.targets, replace=payload.replace)
    project = save_project_metadata(project)
    return DimensionBenchmarkResponse(project=project, report=evaluate_dimension_benchmark(project))


@app.put("/api/projects/{project_id}/benchmark/dimensions/correction", response_model=DimensionBenchmarkResponse)
def api_correct_dimension_benchmark(
    project_id: str,
    request: DimensionCorrectionRequest,
) -> DimensionBenchmarkResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    try:
        project = apply_dimension_correction(project, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    project = save_project(project)
    return DimensionBenchmarkResponse(project=project, report=evaluate_dimension_benchmark(project))


@app.post(
    "/api/projects/{project_id}/agent/dimensions/repair",
    response_model=SemanticRepairResponse,
)
def api_run_semantic_repair(
    project_id: str,
    request: SemanticRepairRequest,
) -> SemanticRepairResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if not project.dimension_ground_truth:
        raise HTTPException(status_code=400, detail="Dimension ground truth is not initialized")

    snapshot_payload = project.model_dump_json(indent=2)
    repaired, report, run = run_semantic_repair_agent(project, request)
    snapshot_relative = Path("semantic_repair_snapshots") / f"{run.id}.json"
    snapshot_path = project_dir(project_id) / snapshot_relative
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(snapshot_payload, encoding="utf-8")
    run.snapshot_file = snapshot_relative.as_posix()
    append_semantic_repair_run(repaired, run)
    repaired = save_project(repaired)
    return SemanticRepairResponse(project=repaired, report=report, run=run)


@app.post(
    "/api/projects/{project_id}/agent/dimensions/repair/{run_id}/rollback",
    response_model=SemanticRepairResponse,
)
def api_rollback_semantic_repair(project_id: str, run_id: str) -> SemanticRepairResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    run = next((item for item in project.semantic_repair_runs if item.id == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="Semantic repair run not found")
    active_runs = [item for item in project.semantic_repair_runs if item.rolled_back_at is None]
    if not active_runs or active_runs[-1].id != run.id:
        raise HTTPException(status_code=400, detail="Only the latest active semantic repair run can be rolled back")
    if not run.snapshot_file:
        raise HTTPException(status_code=400, detail="Semantic repair snapshot is unavailable")

    root = project_dir(project_id).resolve()
    snapshot_path = (root / run.snapshot_file).resolve()
    if root not in snapshot_path.parents or not snapshot_path.exists():
        raise HTTPException(status_code=400, detail="Semantic repair snapshot is invalid")
    restored = ProjectState.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
    rolled_back_run = run.model_copy(
        update={
            "rolled_back_at": datetime.now(timezone.utc),
            "stopped_reason": "rolled_back",
        }
    )
    append_semantic_repair_run(restored, rolled_back_run)
    restored = save_project(restored)
    report = evaluate_dimension_benchmark(restored)
    return SemanticRepairResponse(project=restored, report=report, run=rolled_back_run)


@app.get("/api/agent/tools", response_model=AgentToolCatalogResponse)
def api_agent_tool_catalog() -> AgentToolCatalogResponse:
    return AgentToolCatalogResponse(tools=AGENT_TOOL_REGISTRY.definitions())


@app.get("/api/agent/evals/dataset", response_model=AgentEvalDatasetResponse)
def api_agent_eval_dataset() -> AgentEvalDatasetResponse:
    version, cases = load_agent_eval_dataset()
    return AgentEvalDatasetResponse(version=version, cases=cases)


@app.post(
    "/api/projects/{project_id}/agent/evals",
    response_model=AgentEvalResponse,
)
def api_run_agent_eval(project_id: str, request: AgentEvalRequest) -> AgentEvalResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    try:
        report = run_agent_eval(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    append_agent_eval_report(project, report)
    project = save_project_metadata(project)
    return AgentEvalResponse(project=project, report=report)


@app.post(
    "/api/projects/{project_id}/agent/tasks",
    response_model=AgentTaskResponse,
)
def api_run_agent_task(project_id: str, request: AgentTaskRequest) -> AgentTaskResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    snapshot_payload = project.model_dump_json(indent=2)
    result, run = run_agent_task(project, request, AGENT_TOOL_REGISTRY)
    snapshot_relative = Path("agent_task_snapshots") / f"{run.id}.json"
    snapshot_path = project_dir(project_id) / snapshot_relative
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(snapshot_payload, encoding="utf-8")
    run.snapshot_file = snapshot_relative.as_posix()
    append_agent_task_run(result, run)
    result = save_project(result)
    return AgentTaskResponse(project=result, run=run)


@app.post(
    "/api/projects/{project_id}/agent/tasks/{run_id}/rollback",
    response_model=AgentTaskResponse,
)
def api_rollback_agent_task(project_id: str, run_id: str) -> AgentTaskResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    run = next((item for item in project.agent_task_runs if item.id == run_id), None)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent task run not found")
    active_runs = [item for item in project.agent_task_runs if item.rolled_back_at is None]
    if not active_runs or active_runs[-1].id != run.id:
        raise HTTPException(status_code=400, detail="Only the latest active agent task can be rolled back")
    if not run.snapshot_file:
        raise HTTPException(status_code=400, detail="Agent task snapshot is unavailable")

    root = project_dir(project_id).resolve()
    snapshot_path = (root / run.snapshot_file).resolve()
    if root not in snapshot_path.parents or not snapshot_path.exists():
        raise HTTPException(status_code=400, detail="Agent task snapshot is invalid")
    restored = ProjectState.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    rolled_back_run = run.model_copy(
        update={
            "status": "rolled_back",
            "summary": f"已回滚任务：{run.goal}",
            "rolled_back_at": now,
            "completed_at": now,
        }
    )
    append_agent_task_run(restored, rolled_back_run)
    restored = save_project(restored)
    return AgentTaskResponse(project=restored, run=rolled_back_run)


@app.post("/api/projects/{project_id}/chat", response_model=ChatResponse)
def api_chat(project_id: str, request: ChatRequest):
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    if is_undo_command(request.message):
        project, reply, diffs = undo_last_mechanical_transaction(project)
        if diffs:
            project = save_project(project)
        return ChatResponse(project=project, reply=reply, operations=[], diffs=diffs)

    drive_plan = plan_mechanical_drive_deterministic(request.message, project)
    if drive_plan is None and project.mechanical_ir.dimensions and EDIT_INTENT_RE.search(request.message):
        try:
            drive_plan, _ = plan_mechanical_operation_llm(request.message, project)
        except LlmUnavailable:
            drive_plan = None
    if drive_plan is not None:
        try:
            result = execute_mechanical_operation(project, drive_plan, request.message)
        except MechanicalDriveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        project = save_project(result.project)
        return ChatResponse(
            project=project,
            reply=result.reply,
            operations=result.operations,
            diffs=result.diffs,
        )

    mechanical_edit = plan_mechanical_dimension_edit(request.message, project)
    if mechanical_edit is not None:
        operations = mechanical_edit.operations
        reply = mechanical_edit.reply
    else:
        # Try the LLM planner; fall back to the deterministic parser when no
        # provider/key is configured or the call fails (offline-safe).
        try:
            operations, reply = plan_operations_llm(request.message, project.ir)
        except LlmUnavailable:
            operations, reply = plan_operations(request.message, project.ir)
    if would_replace_vector_import(project, operations):
        return ChatResponse(
            project=project,
            reply=(
                "当前项目已经绑定矢量 PDF。这个命令会用内置模板替换整张图，"
                "从而覆盖刚才 PDF->DXF 的导入结果，所以我先不执行。"
                "如果想重新导入原 PDF，请点 Vectorize；如果想画模板图，请先点 New 新建项目。"
            ),
            operations=[],
            diffs=[],
        )
    if operations:
        try:
            project.ir, diffs = apply_operations(project.ir, operations)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if mechanical_edit is not None:
            sync_mechanical_dimension_edit(project, mechanical_edit)
        project.history.extend(operations)
        project.diffs = diffs
        project = save_project(project)
    else:
        diffs = []
    return ChatResponse(project=project, reply=reply, operations=operations, diffs=diffs)


@app.post("/api/projects/{project_id}/upload")
async def api_upload(project_id: str, file: UploadFile = File(...)):
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        reference = save_reference_upload(project_id, file.filename or "upload", await file.read(), UPLOADS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project.source_file = reference.source_file
    project.source_image = reference.source_image
    project.source_kind = reference.source_kind
    return save_project(project)


@app.get("/api/projects/{project_id}/analyze", response_model=AnalyzeResponse)
def api_analyze(project_id: str) -> AnalyzeResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        return analyze_reference(project, UPLOADS_DIR, output_dir=project_dir(project_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/ocr", response_model=OcrResponse)
def api_ocr(project_id: str, request: OcrRequest | None = None) -> OcrResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        ocr_request = request or OcrRequest()
        result = run_project_ocr(
            project,
            UPLOADS_DIR,
            language_hint=ocr_request.language,
            engine_hint=ocr_request.engine,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project.ocr_regions = result.regions
    project = save_project_metadata(project)
    return OcrResponse(project=project, regions=result.regions, warnings=result.warnings)


@app.post("/api/projects/{project_id}/ocr/tables", response_model=TableOcrResponse)
def api_table_ocr(project_id: str, request: OcrRequest | None = None) -> TableOcrResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        ocr_request = request or OcrRequest()
        result = extract_table_ocr_from_reference(
            project,
            UPLOADS_DIR,
            language_hint=ocr_request.language,
            engine_hint=ocr_request.engine,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project.table_ocr_cells = result.cells
    project = save_project_metadata(project)
    return TableOcrResponse(project=project, cells=result.cells, warnings=result.warnings)


@app.post("/api/projects/{project_id}/reconstruct/tables", response_model=TableReconstructionResponse)
def api_reconstruct_tables(project_id: str) -> TableReconstructionResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        reconstruction = reconstruct_tables_from_reference(project, UPLOADS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    before_ids = [entity.id for entity in project.ir.entities]
    project.ir = reconstruction.ir
    project.diffs = [DiffItem(path="ir.entities", before=before_ids, after=[entity.id for entity in project.ir.entities])]
    project = save_project(project)
    return TableReconstructionResponse(
        project=project,
        layout_passed=reconstruction.layout_passed,
        warnings=reconstruction.warnings,
        regions=[
            ReconstructionRegion(target=region.target, x1=region.bbox[0], y1=region.bbox[1], x2=region.bbox[2], y2=region.bbox[3])
            for region in reconstruction.regions
        ],
    )


@app.post("/api/projects/{project_id}/reconstruct/section", response_model=SectionReconstructionResponse)
def api_reconstruct_section(project_id: str) -> SectionReconstructionResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        reconstruction = reconstruct_section_from_reference(project, UPLOADS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    before_ids = [entity.id for entity in project.ir.entities if entity.group == "section_view"]
    project.ir = reconstruction.ir
    after_ids = [entity.id for entity in project.ir.entities if entity.group == "section_view"]
    project.diffs = [DiffItem(path="ir.section_view", before=before_ids, after=after_ids)]
    project = save_project(project)
    region = reconstruction.region
    return SectionReconstructionResponse(
        project=project,
        line_count=reconstruction.line_count,
        hatch_count=reconstruction.hatch_count,
        warnings=reconstruction.warnings,
        region=ReconstructionRegion(
            target="section_view",
            x1=region.bbox[0],
            y1=region.bbox[1],
            x2=region.bbox[2],
            y2=region.bbox[3],
        ),
    )


@app.post("/api/projects/{project_id}/reconstruct/scan", response_model=ScanCadReconstructionResponse)
def api_reconstruct_scan(project_id: str) -> ScanCadReconstructionResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        reconstruction = reconstruct_scan_cad_from_reference(project, UPLOADS_DIR, output_dir=project_dir(project.project_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    before_ids = [entity.id for entity in project.ir.entities]
    project.ir = reconstruction.ir
    project.diffs = [
        DiffItem(
            path="ir.entities",
            before=ids_diff_payload(before_ids),
            after=ids_diff_payload([entity.id for entity in project.ir.entities]),
        )
    ]
    project = save_project(project)
    return ScanCadReconstructionResponse(
        project=project,
        entity_count=reconstruction.entity_count,
        trace_count=reconstruction.trace_count,
        structured_counts=reconstruction.structured_counts,
        warnings=reconstruction.warnings,
    )


@app.post("/api/projects/{project_id}/promote/scan", response_model=ScanPromotionResponse)
def api_promote_scan(project_id: str) -> ScanPromotionResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        promotion = promote_scan_primitives(project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    before_ids = [entity.id for entity in project.ir.entities if entity.group == "promoted_geometry"]
    project.ir = promotion.ir
    after_ids = [entity.id for entity in project.ir.entities if entity.group == "promoted_geometry"]
    project.diffs = [
        DiffItem(
            path="ir.promoted_geometry",
            before=ids_diff_payload(before_ids),
            after=ids_diff_payload(after_ids),
        )
    ]
    project = save_project(project)
    return ScanPromotionResponse(
        project=project,
        promoted_counts=promotion.promoted_counts,
        source_count=promotion.source_count,
        warnings=promotion.warnings,
    )


@app.post("/api/projects/{project_id}/semantics/dimensions", response_model=DimensionSemanticsResponse)
def api_dimension_semantics(project_id: str) -> DimensionSemanticsResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    semantics = detect_dimension_bindings(project)
    project.dimension_bindings = semantics.bindings
    arrow_render = render_dimension_binding_arrowheads(project.ir, semantics.bindings)
    project.ir = arrow_render.ir
    project.mechanical_ir = build_mechanical_drawing_ir(
        project.ir,
        semantics.bindings,
        arrow_render.mechanical_dimensions,
        [*semantics.warnings, *arrow_render.warnings],
    )
    project.mechanical_dimensions = project.mechanical_ir.dimensions
    project = save_project(project)
    return DimensionSemanticsResponse(
        project=project,
        bindings=semantics.bindings,
        warnings=project.mechanical_ir.warnings,
    )


@app.post("/api/projects/{project_id}/pipeline/cad", response_model=CadPipelineResponse)
def api_cad_pipeline(project_id: str, request: CadPipelineRequest | None = None) -> CadPipelineResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    if not project.source_image and not project.source_file:
        raise HTTPException(status_code=400, detail="Upload a PDF or image before running the CAD pipeline.")

    pipeline_request = request or CadPipelineRequest()
    steps: list[CadPipelineStep] = []
    warnings: list[str] = []

    if project.source_kind == "vector_pdf":
        try:
            project.ir = reconstruct_vector_from_reference(project, UPLOADS_DIR)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        steps.append(_pipeline_step("vector_extract", "ok", f"{len(project.ir.entities)} entities"))
    else:
        try:
            analysis = analyze_reference(project, UPLOADS_DIR, output_dir=project_dir(project_id))
            steps.append(_pipeline_step("analyze", "ok", f"{len(analysis.boxes)} regions"))
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"Analyze skipped: {exc}")
            steps.append(_pipeline_step("analyze", "warning", str(exc)))

        try:
            reconstruction = reconstruct_scan_cad_from_reference(project, UPLOADS_DIR, output_dir=project_dir(project_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        project.ir = reconstruction.ir
        warnings.extend(reconstruction.warnings)
        structured = ", ".join(f"{name}={count}" for name, count in reconstruction.structured_counts.items())
        steps.append(
            _pipeline_step(
                "scan_reconstruct",
                "ok",
                f"{reconstruction.entity_count} entities, trace={reconstruction.trace_count}; {structured}",
            )
        )

        try:
            promotion = promote_scan_primitives(project)
        except ValueError as exc:
            warnings.append(f"Promote skipped: {exc}")
            steps.append(_pipeline_step("promote", "warning", str(exc)))
        else:
            project.ir = promotion.ir
            warnings.extend(promotion.warnings)
            promoted = ", ".join(f"{name}={count}" for name, count in promotion.promoted_counts.items())
            steps.append(_pipeline_step("promote", "ok", promoted))

        try:
            arrow_detection = detect_arrowheads_from_reference(project, UPLOADS_DIR)
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"Arrow template detection skipped: {exc}")
            steps.append(_pipeline_step("arrow_template", "warning", str(exc)))
        else:
            project.ir = arrow_detection.ir
            warnings.extend(arrow_detection.warnings)
            steps.append(_pipeline_step("arrow_template", "ok", f"{arrow_detection.detected_count} arrowheads"))

    try:
        ocr_result = run_project_ocr(
            project,
            UPLOADS_DIR,
            language_hint=pipeline_request.language,
            engine_hint=pipeline_request.engine,
        )
    except (FileNotFoundError, ValueError) as exc:
        warnings.append(f"OCR skipped: {exc}")
        steps.append(_pipeline_step("ocr", "warning", str(exc)))
    else:
        project.ocr_regions = ocr_result.regions
        warnings.extend(ocr_result.warnings)
        text_count = sum(1 for region in ocr_result.regions if region.text.strip())
        steps.append(_pipeline_step("ocr", "ok", f"{len(ocr_result.regions)} regions, {text_count} with text"))

    if pipeline_request.include_table_ocr:
        table_cells = []
        try:
            table_result = extract_table_ocr_from_reference(
                project,
                UPLOADS_DIR,
                language_hint=pipeline_request.language,
                engine_hint=pipeline_request.engine,
            )
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"Table OCR skipped: {exc}")
            steps.append(_pipeline_step("table_ocr", "warning", str(exc)))
        else:
            project.table_ocr_cells = table_result.cells
            warnings.extend(table_result.warnings)
            text_count = sum(1 for cell in table_result.cells if cell.text.strip())
            steps.append(_pipeline_step("table_ocr", "ok", f"{len(table_result.cells)} cells, {text_count} with text"))
            table_cells = table_result.cells

        title_block = extract_title_block_cells(project, UPLOADS_DIR, table_cells)
        project.title_block_cells = title_block.cells
        warnings.extend(title_block.warnings)
        title_status = "ok" if title_block.cells else "warning"
        steps.append(_pipeline_step("title_block", title_status, f"{title_block.provider}: {len(title_block.cells)} cells"))
        title_render = render_title_block_cells_into_ir(project.ir, title_block.cells)
        project.ir = title_render.ir
        warnings.extend(title_render.warnings)
        render_status = "ok" if title_render.grid_count or title_render.text_count else "skipped"
        steps.append(
            _pipeline_step(
                "title_block_render",
                render_status,
                f"{title_render.grid_count} grid lines, {title_render.text_count} text entities",
            )
        )

        parameter_table_cells = [cell for cell in table_cells if cell.target == "parameter_table"]
        table_text = render_table_ocr_cells_into_ir(project.ir, parameter_table_cells)
        project.ir = table_text.ir
        warnings.extend(table_text.warnings)
        steps.append(_pipeline_step("table_text", "ok", f"{table_text.text_count} parameter-table CAD text entities"))

    semantics = detect_dimension_bindings(project)
    project.dimension_bindings = semantics.bindings
    warnings.extend(semantics.warnings)
    steps.append(_pipeline_step("dimension_semantics", "ok", f"{len(semantics.bindings)} bindings"))
    arrow_render = render_dimension_binding_arrowheads(project.ir, semantics.bindings)
    project.ir = arrow_render.ir
    project.mechanical_ir = build_mechanical_drawing_ir(
        project.ir,
        semantics.bindings,
        arrow_render.mechanical_dimensions,
        [*semantics.warnings, *arrow_render.warnings],
    )
    project.mechanical_dimensions = project.mechanical_ir.dimensions
    warnings.extend(arrow_render.warnings)
    arrow_render_status = "ok" if arrow_render.arrow_line_count else "skipped"
    steps.append(
        _pipeline_step(
            "dimension_arrow_render",
            arrow_render_status,
            (
                f"{arrow_render.arrow_line_count} solid arrowheads; "
                f"{len(project.mechanical_ir.dimensions)} Dimension objects, "
                f"{sum(dimension.export_ready for dimension in project.mechanical_ir.dimensions)} native-DXF ready, "
                f"{len(project.mechanical_ir.unresolved_binding_ids)} unresolved"
            ),
        )
    )

    if project.source_kind == "vector_pdf" and project.source_file:
        project = save_project(project)
        try:
            source_path = _upload_url_to_path(project.source_file, UPLOADS_DIR)
            assets = export_vector_pdf_assets(source_path, project_dir(project.project_id))
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"Vector asset export skipped: {exc}")
            steps.append(_pipeline_step("vector_assets", "warning", str(exc)))
        else:
            warnings.extend(assets.warnings)
            steps.append(_pipeline_step("vector_assets", "ok", f"preview={assets.preview_source}; dxf={assets.dxf_source}"))
        project = save_project_metadata(project)
    else:
        project = save_project(project)

    return CadPipelineResponse(project=project, steps=steps, warnings=warnings)


@app.post("/api/projects/{project_id}/benchmark/vectorizers", response_model=VectorizerBenchmarkResponse)
def api_benchmark_vectorizers(project_id: str) -> VectorizerBenchmarkResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        benchmark = run_project_vectorizer_benchmark(project, UPLOADS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VectorizerBenchmarkResponse(
        project_id=benchmark.project_id,
        prepared_image_url=benchmark.prepared_image_url,
        results=[
            VectorizerToolResult(
                name=result.name,
                status=result.status,
                elapsed_sec=result.elapsed_sec,
                svg_url=result.svg_url,
                dxf_url=result.dxf_url,
                preview_url=result.preview_url,
                svg_path_count=result.svg_path_count,
                dxf_entity_count=result.dxf_entity_count,
                output_bytes=result.output_bytes,
                command=result.command,
                detail=result.detail,
            )
            for result in benchmark.results
        ],
    )


@app.post("/api/projects/{project_id}/reconstruct/vector", response_model=VectorReconstructionResponse)
def api_reconstruct_vector(project_id: str) -> VectorReconstructionResponse:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    try:
        ir = reconstruct_vector_from_reference(project, UPLOADS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    before_ids = [entity.id for entity in project.ir.entities]
    project.ir = ir
    project.diffs = [
        DiffItem(
            path="ir.entities",
            before=ids_diff_payload(before_ids),
            after=ids_diff_payload([entity.id for entity in ir.entities]),
        )
    ]
    project = save_project(project)
    source_path = UPLOADS_DIR / project.source_file.removeprefix("/api/uploads/")
    assets = export_vector_pdf_assets(source_path, project_dir(project.project_id))
    project = save_project_metadata(project)
    return VectorReconstructionResponse(
        project=project,
        entity_count=len(ir.entities),
        preview_source=assets.preview_source,
        dxf_source=assets.dxf_source,
        warnings=assets.warnings,
    )


@app.get("/api/projects/{project_id}/exports", response_model=ExportPaths)
def api_exports(project_id: str) -> ExportPaths:
    project_path = PROJECTS_DIR / project_id
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return ExportPaths(
        dxf_url=f"/api/projects/{project_id}/files/output.dxf",
        svg_url=f"/api/projects/{project_id}/files/preview.svg",
    )


@app.get("/api/projects/{project_id}/files/{filename}")
def api_project_file(project_id: str, filename: str):
    path = PROJECTS_DIR / project_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if filename.endswith(".svg"):
        media_type = "image/svg+xml"
    elif filename.endswith(".png"):
        media_type = "image/png"
    else:
        media_type = "application/dxf"
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/uploads/{filename}")
def api_uploaded_file(filename: str):
    path = UPLOADS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    return FileResponse(path)

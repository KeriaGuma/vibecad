from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class Layer(BaseModel):
    name: str
    color: str = "white"
    lineweight: float = 0.25
    linetype: str = "CONTINUOUS"
    locked: bool = False
    editable: bool = True


class BaseEntity(BaseModel):
    id: str
    type: str
    layer: str = "0"
    label: str | None = None
    group: str | None = None
    tags: list[str] = Field(default_factory=list)
    stroke_width: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LineEntity(BaseEntity):
    type: Literal["line"] = "line"
    x1: float
    y1: float
    x2: float
    y2: float


class PolylineEntity(BaseEntity):
    type: Literal["polyline"] = "polyline"
    points: list[list[float]]
    closed: bool = False


class CircleEntity(BaseEntity):
    type: Literal["circle"] = "circle"
    cx: float
    cy: float
    r: float


class ArcEntity(BaseEntity):
    type: Literal["arc"] = "arc"
    cx: float
    cy: float
    r: float
    start_angle: float
    end_angle: float


class RectangleEntity(BaseEntity):
    type: Literal["rectangle"] = "rectangle"
    x: float
    y: float
    width: float
    height: float


class TextEntity(BaseEntity):
    type: Literal["text"] = "text"
    x: float
    y: float
    text: str
    height: float = 3.5
    rotation: float = 0


Entity = Annotated[
    LineEntity | PolylineEntity | CircleEntity | ArcEntity | RectangleEntity | TextEntity,
    Field(discriminator="type"),
]


class DrawingIR(BaseModel):
    units: Literal["mm", "inch"] = "mm"
    layers: list[Layer] = Field(default_factory=lambda: [Layer(name="0")])
    entities: list[Entity] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DiffItem(BaseModel):
    path: str
    before: Any = None
    after: Any = None


class Operation(BaseModel):
    operation: Literal[
        "create_plate",
        "create_spur_gear_drawing",
        "add_entity",
        "modify_entity",
        "delete_entity",
        "move_entity",
        "set_layer",
    ]
    entity_id: str | None = None
    entity: Entity | None = None
    changes: dict[str, Any] = Field(default_factory=dict)
    dx: float = 0
    dy: float = 0
    layer: str | None = None
    reason: str = ""


class ParsedDimensionValue(BaseModel):
    kind: Literal["linear", "diameter", "radius", "roughness", "tolerance", "unknown"] = "unknown"
    raw_text: str
    nominal: float | None = None
    upper_tol: float | None = None
    lower_tol: float | None = None
    unit: str = "mm"


class DimensionBinding(BaseModel):
    id: str
    dimension_line_id: str
    arrow_ids: list[str] = Field(default_factory=list)
    text_id: str | None = None
    text: str = ""
    parsed: ParsedDimensionValue
    confidence: float = Field(ge=0, le=1)
    kind: Literal["linear", "diameter", "radius", "roughness", "tolerance", "unknown"] = "unknown"
    line_x1: float
    line_y1: float
    line_x2: float
    line_y2: float
    text_x: float | None = None
    text_y: float | None = None
    binding_method: str = "graph_text_arrow_line"
    graph_path: list[str] = Field(default_factory=list)
    graph_score: float | None = None
    source: str = "dimension_semantics_v0"


class MechanicalArrowhead(BaseModel):
    candidate_id: str
    source_entity_id: str
    render_entity_id: str
    tip_x: float
    tip_y: float
    direction_x: float
    direction_y: float
    score: float
    endpoint: str
    endpoint_distance: float


class MechanicalDimensionObject(BaseModel):
    id: str
    binding_id: str
    kind: Literal["linear", "diameter", "radius", "roughness", "tolerance", "unknown"] = "unknown"
    text: str = ""
    parsed: ParsedDimensionValue
    confidence: float = Field(ge=0, le=1)
    dimension_line_id: str
    text_id: str | None = None
    arrowheads: list[MechanicalArrowhead] = Field(default_factory=list)
    extension_line_ids: list[str] = Field(default_factory=list)
    measured_geometry_ids: list[str] = Field(default_factory=list)
    target_geometry_ids: list[str] = Field(default_factory=list)
    measurement_points: list[list[float]] = Field(default_factory=list)
    dimension_line_point: list[float] | None = None
    relation_confidence: dict[str, float] = Field(default_factory=dict)
    orientation: Literal["horizontal", "vertical", "aligned", "radial"] = "aligned"
    dxf_dimension_type: Literal["linear", "aligned", "diameter", "radius"] | None = None
    export_ready: bool = False
    edit_mode: Literal["annotation_override", "driving"] = "annotation_override"
    measured_value: float | None = None
    last_edit_source: Literal["deterministic", "deepseek"] | None = None
    validation_status: Literal["passed", "failed"] | None = None
    status: Literal["complete", "partial", "unresolved"] = "partial"
    issues: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source: str = "mechanical_semantics_v0"


class MechanicalDrawingIR(BaseModel):
    """Editable mechanical semantics linked back to drawing entity ids."""

    schema_version: Literal["1.0"] = "1.0"
    units: Literal["mm", "inch"] = "mm"
    dimensions: list[MechanicalDimensionObject] = Field(default_factory=list)
    entity_roles: dict[str, list[str]] = Field(default_factory=dict)
    unresolved_binding_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: str = "mechanical_ir_v1"


class DimensionGroundTruth(BaseModel):
    id: str
    label: str
    expected_text: str
    kind: Literal["linear", "diameter", "radius", "roughness", "tolerance", "unknown"]
    nominal: float | None = None
    unit: Literal["mm", "inch"] = "mm"
    matched_dimension_id: str | None = None
    required_relations: list[
        Literal[
            "text",
            "dimension_line",
            "arrowheads",
            "extension_lines",
            "measured_geometry",
            "definition_points",
        ]
    ] = Field(
        default_factory=lambda: [
            "text",
            "dimension_line",
            "arrowheads",
            "extension_lines",
            "measured_geometry",
            "definition_points",
        ]
    )
    notes: str = ""
    source: Literal["seed", "manual"] = "manual"


class DimensionCorrection(BaseModel):
    ground_truth_id: str
    dimension_id: str
    text_id: str | None = None
    dimension_line_id: str
    arrow_entity_ids: list[str] = Field(default_factory=list)
    extension_line_ids: list[str] = Field(default_factory=list)
    measured_geometry_ids: list[str] = Field(default_factory=list)
    updated_at: datetime
    source: Literal["manual_override", "auto_repair"] = "manual_override"


class SemanticRepairStep(BaseModel):
    index: int = Field(ge=0)
    ground_truth_id: str
    label: str
    status: Literal["accepted", "rejected", "skipped", "error"]
    tool_calls: list[str] = Field(default_factory=list)
    score_before: float = Field(ge=0, le=1)
    score_after: float = Field(ge=0, le=1)
    overall_before: float = Field(ge=0, le=1)
    overall_after: float = Field(ge=0, le=1)
    missing_before: list[str] = Field(default_factory=list)
    missing_after: list[str] = Field(default_factory=list)
    selected_entities: dict[str, list[str]] = Field(default_factory=dict)
    detail: str = ""


class SemanticRepairRun(BaseModel):
    id: str
    created_at: datetime
    planner_source: Literal["deepseek", "deterministic", "deterministic_fallback"]
    planner_model: str | None = None
    planner_reason: str = ""
    budget: int = Field(ge=1)
    llm_calls: int = Field(default=0, ge=0)
    before_score: float = Field(ge=0, le=1)
    after_score: float = Field(ge=0, le=1)
    accepted_steps: int = Field(ge=0)
    rejected_steps: int = Field(ge=0)
    stopped_reason: str
    snapshot_file: str | None = None
    steps: list[SemanticRepairStep] = Field(default_factory=list)
    rolled_back_at: datetime | None = None


AgentToolName = Literal[
    "inspect_drawing",
    "evaluate_drawing",
    "evaluate_dimensions",
    "repair_dimensions",
    "drive_dimension",
    "edit_cad",
    "export_dxf",
]


class AgentToolArguments(BaseModel):
    message: str | None = None
    dimension_id: str | None = None
    target_value: float | None = None
    anchor: Literal["start", "end"] = "start"
    max_steps: int = Field(default=3, ge=1, le=10)
    min_gain: float = Field(default=0.01, ge=0, le=0.5)


class AgentPlannedStep(BaseModel):
    call_id: str
    tool: AgentToolName
    arguments: AgentToolArguments = Field(default_factory=AgentToolArguments)
    reason: str = ""


class AgentTaskStepTrace(BaseModel):
    index: int = Field(ge=0)
    attempt: int = Field(default=1, ge=1)
    call_id: str
    tool: AgentToolName
    status: Literal["accepted", "skipped", "error", "rolled_back"]
    arguments: AgentToolArguments = Field(default_factory=AgentToolArguments)
    reason: str = ""
    observation: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    dimension_score_before: float | None = Field(default=None, ge=0, le=1)
    dimension_score_after: float | None = Field(default=None, ge=0, le=1)
    mutating: bool = False
    reversible: bool = False
    started_at: datetime
    completed_at: datetime


class AgentTaskClarification(BaseModel):
    """One focused question emitted when an edit target is not safely resolvable."""

    question: str
    reason: str
    candidates: list[str] = Field(default_factory=list)


class AgentTaskRun(BaseModel):
    id: str
    goal: str
    status: Literal["completed", "partial", "failed", "needs_clarification", "rolled_back"]
    planner_source: Literal["deepseek", "deterministic", "deterministic_fallback"]
    planner_model: str | None = None
    planner_reason: str = ""
    initial_plan: list[AgentPlannedStep] = Field(default_factory=list)
    steps: list[AgentTaskStepTrace] = Field(default_factory=list)
    llm_calls: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)
    policy_injected_steps: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(ge=1)
    before_dimension_score: float | None = Field(default=None, ge=0, le=1)
    after_dimension_score: float | None = Field(default=None, ge=0, le=1)
    summary: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    clarification: AgentTaskClarification | None = None
    snapshot_file: str | None = None
    created_at: datetime
    completed_at: datetime
    rolled_back_at: datetime | None = None


class AgentToolDefinition(BaseModel):
    name: AgentToolName
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    mutating: bool = False
    reversible: bool = False
    validator: str


class AgentEvalAssertion(BaseModel):
    kind: Literal[
        "entity_field",
        "entity_count",
        "dimension_value",
        "dimension_complete_min",
        "ir_unchanged",
        "artifact_present",
        "no_mutating_calls",
        "rollback_observed",
    ]
    entity_id: str | None = None
    field: str | None = None
    expected: Any = None
    artifact: str | None = None


class AgentEvalCase(BaseModel):
    id: str
    category: Literal["planning", "editing", "semantic", "safety", "recovery", "export"]
    description: str
    goal: str
    fixture: Literal["baseline_plate", "complete_dimension", "repairable_dimension", "locked_reference"]
    expected_status: Literal["completed", "partial", "failed", "needs_clarification"]
    expected_tools: list[AgentToolName] = Field(default_factory=list)
    forbidden_tools: list[AgentToolName] = Field(default_factory=list)
    expected_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    assertions: list[AgentEvalAssertion] = Field(default_factory=list)


class AgentEvalCaseResult(BaseModel):
    case_id: str
    category: str
    goal: str
    passed: bool
    task_success: bool = False
    score: float = Field(ge=0, le=1)
    status_match: bool
    tool_precision: float = Field(ge=0, le=1)
    tool_recall: float = Field(ge=0, le=1)
    tool_order_match: bool
    argument_accuracy: float = Field(ge=0, le=1)
    assertion_accuracy: float = Field(ge=0, le=1)
    safety_passed: bool
    expected_tools: list[AgentToolName] = Field(default_factory=list)
    actual_tools: list[AgentToolName] = Field(default_factory=list)
    failed_assertions: list[str] = Field(default_factory=list)
    invalid_action_count: int = Field(ge=0)
    rollback_observed: bool = False
    replan_count: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    policy_injected_steps: int = Field(default=0, ge=0)
    duration_ms: float = Field(ge=0)
    run_status: str
    planner_source: str


class AgentEvalMetrics(BaseModel):
    task_success_rate: float = Field(ge=0, le=1)
    average_score: float = Field(ge=0, le=1)
    tool_selection_precision: float = Field(ge=0, le=1)
    tool_selection_recall: float = Field(ge=0, le=1)
    tool_order_accuracy: float = Field(ge=0, le=1)
    argument_accuracy: float = Field(ge=0, le=1)
    assertion_accuracy: float = Field(ge=0, le=1)
    safety_pass_rate: float = Field(ge=0, le=1)
    invalid_action_rate: float = Field(ge=0, le=1)
    rollback_success_rate: float = Field(ge=0, le=1)
    replan_success_rate: float = Field(ge=0, le=1)
    average_tool_calls: float = Field(ge=0)
    average_llm_calls: float = Field(ge=0)
    average_replans: float = Field(ge=0)
    average_duration_ms: float = Field(ge=0)
    average_policy_injected_steps: float = Field(default=0, ge=0)


class AgentEvalReport(BaseModel):
    id: str
    dataset_version: str
    mode: Literal["deterministic", "deepseek"]
    case_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    metrics: AgentEvalMetrics
    cases: list[AgentEvalCaseResult] = Field(default_factory=list)
    created_at: datetime
    completed_at: datetime


class MechanicalOperation(BaseModel):
    """A constrained, locally validated operation proposed by an agent planner."""

    operation: Literal["drive_dimension"] = "drive_dimension"
    dimension_id: str
    target_value: float = Field(gt=0)
    anchor: Literal["start", "end"] = "start"
    planner_source: Literal["deterministic", "deepseek"] = "deterministic"
    confidence: float = Field(default=1.0, ge=0, le=1)
    reason: str = ""


class MechanicalValidation(BaseModel):
    passed: bool
    dimension_id: str
    target_value: float
    measured_value: float | None = None
    tolerance: float = 0.01
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class MechanicalTransaction(BaseModel):
    """Snapshot-backed edit transaction used for safe rollback."""

    id: str
    command: str
    planner_source: str
    operations: list[Operation] = Field(default_factory=list)
    diffs: list[DiffItem] = Field(default_factory=list)
    validation: MechanicalValidation
    before_ir: DrawingIR
    before_dimension_bindings: list[DimensionBinding] = Field(default_factory=list)
    before_mechanical_ir: MechanicalDrawingIR = Field(default_factory=MechanicalDrawingIR)
    before_history_length: int = 0


class ProjectState(BaseModel):
    project_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    source_file: str | None = None
    source_image: str | None = None
    source_kind: str | None = None
    ir: DrawingIR
    history: list[Operation] = Field(default_factory=list)
    diffs: list[DiffItem] = Field(default_factory=list)
    ocr_regions: list["OcrRegion"] = Field(default_factory=list)
    table_ocr_cells: list["TableCellOcr"] = Field(default_factory=list)
    title_block_cells: list["TitleBlockCell"] = Field(default_factory=list)
    dimension_bindings: list[DimensionBinding] = Field(default_factory=list)
    mechanical_dimensions: list[MechanicalDimensionObject] = Field(default_factory=list)
    mechanical_ir: MechanicalDrawingIR = Field(default_factory=MechanicalDrawingIR)
    mechanical_transactions: list[MechanicalTransaction] = Field(default_factory=list)
    dimension_ground_truth: list[DimensionGroundTruth] = Field(default_factory=list)
    dimension_corrections: list[DimensionCorrection] = Field(default_factory=list)
    semantic_repair_runs: list[SemanticRepairRun] = Field(default_factory=list)
    agent_task_runs: list[AgentTaskRun] = Field(default_factory=list)
    agent_eval_reports: list[AgentEvalReport] = Field(default_factory=list)


class CreateProjectRequest(BaseModel):
    name: str = "Untitled drawing"
    prompt: str = ""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    project: ProjectState
    reply: str
    operations: list[Operation]
    diffs: list[DiffItem]


class DimensionBenchmarkSeedRequest(BaseModel):
    targets: list[DimensionGroundTruth] = Field(default_factory=list)
    replace: bool = False


class DimensionCorrectionRequest(BaseModel):
    ground_truth_id: str
    dimension_id: str | None = None
    text_id: str | None = None
    dimension_line_id: str
    arrow_entity_ids: list[str] = Field(default_factory=list)
    extension_line_ids: list[str] = Field(default_factory=list)
    measured_geometry_ids: list[str] = Field(default_factory=list)


class DimensionTargetEval(BaseModel):
    ground_truth: DimensionGroundTruth
    matched_dimension_id: str | None = None
    matched_text: str = ""
    score: float = Field(ge=0, le=1)
    passed: bool = False
    corrected: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    missing_relations: list[str] = Field(default_factory=list)


class DimensionBenchmarkReport(BaseModel):
    project_id: str
    target_count: int
    matched_count: int
    complete_count: int
    overall_score: float = Field(ge=0, le=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    targets: list[DimensionTargetEval] = Field(default_factory=list)


class DimensionBenchmarkResponse(BaseModel):
    project: ProjectState
    report: DimensionBenchmarkReport


class SemanticRepairRequest(BaseModel):
    use_llm: bool = True
    max_steps: int = Field(default=3, ge=1, le=10)
    min_gain: float = Field(default=0.01, ge=0, le=0.5)


class SemanticRepairResponse(BaseModel):
    project: ProjectState
    report: DimensionBenchmarkReport
    run: SemanticRepairRun


class AgentTaskRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    use_llm: bool = True
    max_tool_calls: int = Field(default=8, ge=1, le=20)
    max_replans: int = Field(default=1, ge=0, le=3)


class AgentTaskResponse(BaseModel):
    project: ProjectState
    run: AgentTaskRun


class AgentToolCatalogResponse(BaseModel):
    tools: list[AgentToolDefinition]


class AgentEvalRequest(BaseModel):
    mode: Literal["deterministic", "deepseek"] = "deterministic"
    case_ids: list[str] = Field(default_factory=list)
    max_cases: int = Field(default=13, ge=1, le=30)


class AgentEvalDatasetResponse(BaseModel):
    version: str
    cases: list[AgentEvalCase]


class AgentEvalResponse(BaseModel):
    project: ProjectState
    report: AgentEvalReport


class ExportPaths(BaseModel):
    dxf_url: str
    svg_url: str


class RegionBox(BaseModel):
    target: Literal["title_block", "parameter_table", "section_view", "circular_view", "dimensions"]
    label: str
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    confidence: float = Field(ge=0, le=1)
    source: str = "layout_heuristic_v0"


class AnalyzeResponse(BaseModel):
    project_id: str
    source_image: str
    image_width: int | None = None
    image_height: int | None = None
    overlay_image: str | None = None
    preprocessed_image: str | None = None
    frame: list[float] | None = None
    deskew_angle: float = 0.0
    boxes: list[RegionBox]


class OcrRegion(BaseModel):
    target: Literal["title_block", "parameter_table", "section_view", "circular_view", "dimensions"]
    label: str
    text: str = ""
    confidence: float = 0.0
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    engine: str = "tesseract"
    language: str = ""
    source: str = "tesseract_cli"


class OcrRequest(BaseModel):
    language: Literal["auto", "zh", "en"] = "auto"
    engine: Literal["auto", "edocr2", "paddle", "tesseract"] = "auto"


class CadPipelineRequest(BaseModel):
    language: Literal["auto", "zh", "en"] = "auto"
    engine: Literal["auto", "edocr2", "paddle", "tesseract"] = "auto"
    include_table_ocr: bool = True


class CadPipelineStep(BaseModel):
    name: str
    status: Literal["ok", "skipped", "warning"]
    detail: str = ""


class TableCellOcr(BaseModel):
    target: Literal["title_block", "parameter_table"]
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str = ""
    confidence: float = Field(ge=0, le=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    engine: str = "paddleocr"
    language: str = "zh"
    source: str = "table_cell_ocr_v0"


class TitleBlockCell(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    text: str = ""
    confidence: float = Field(ge=0, le=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    provider: str = "current_grid"
    source: str = "title_block_provider"


class ReconstructionRegion(BaseModel):
    target: Literal["title_block", "parameter_table", "section_view"]
    x1: float
    y1: float
    x2: float
    y2: float


class TableReconstructionResponse(BaseModel):
    project: ProjectState
    layout_passed: bool
    warnings: list[str] = Field(default_factory=list)
    regions: list[ReconstructionRegion]


class SectionReconstructionResponse(BaseModel):
    project: ProjectState
    line_count: int
    hatch_count: int
    warnings: list[str] = Field(default_factory=list)
    region: ReconstructionRegion


class ScanCadReconstructionResponse(BaseModel):
    project: ProjectState
    entity_count: int
    trace_count: int
    structured_counts: dict[str, int] = Field(default_factory=dict)
    source: str = "scan_cv_trace_v0"
    warnings: list[str] = Field(default_factory=list)


class ScanPromotionResponse(BaseModel):
    project: ProjectState
    promoted_counts: dict[str, int] = Field(default_factory=dict)
    source_count: int
    warnings: list[str] = Field(default_factory=list)


class VectorizerToolResult(BaseModel):
    name: str
    status: str
    elapsed_sec: float = 0.0
    svg_url: str | None = None
    dxf_url: str | None = None
    preview_url: str | None = None
    svg_path_count: int = 0
    dxf_entity_count: int = 0
    output_bytes: int = 0
    command: list[str] = Field(default_factory=list)
    detail: str = ""


class VectorizerBenchmarkResponse(BaseModel):
    project_id: str
    prepared_image_url: str
    results: list[VectorizerToolResult]


class VectorReconstructionResponse(BaseModel):
    project: ProjectState
    entity_count: int
    source: str = "vector_pdf_extract"
    preview_source: str = "ir_svg"
    dxf_source: str = "ir_fallback"
    warnings: list[str] = Field(default_factory=list)


class OcrResponse(BaseModel):
    project: ProjectState
    regions: list[OcrRegion]
    warnings: list[str] = Field(default_factory=list)


class TableOcrResponse(BaseModel):
    project: ProjectState
    cells: list[TableCellOcr]
    warnings: list[str] = Field(default_factory=list)


class DimensionSemanticsResponse(BaseModel):
    project: ProjectState
    bindings: list[DimensionBinding]
    warnings: list[str] = Field(default_factory=list)


class CadPipelineResponse(BaseModel):
    project: ProjectState
    steps: list[CadPipelineStep]
    warnings: list[str] = Field(default_factory=list)


def default_ir() -> DrawingIR:
    return DrawingIR(
        layers=[
            Layer(name="outline", color="white"),
            Layer(name="holes", color="red"),
            Layer(name="dimensions", color="green"),
            Layer(name="notes", color="cyan"),
        ],
        entities=[
            RectangleEntity(
                id="plate_1",
                layer="outline",
                x=0,
                y=0,
                width=100,
                height=60,
                label="base plate",
            ),
            CircleEntity(id="hole_1", layer="holes", cx=25, cy=30, r=4, label="left hole"),
            CircleEntity(id="hole_2", layer="holes", cx=75, cy=30, r=4, label="right hole"),
            TextEntity(id="note_1", layer="notes", x=0, y=68, text="Vibe CAD MVP", height=4),
        ],
        notes=["Baseline sample: 100 x 60 mm plate with two diameter 8 holes."],
    )

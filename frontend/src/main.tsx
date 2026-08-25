import React, { FormEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  CheckCircle2,
  Download,
  FileUp,
  History,
  Layers,
  Lock,
  MessageSquare,
  MousePointer2,
  Play,
  Plus,
  RefreshCcw,
  Ruler,
  Save,
  ScanLine,
  Search,
  Wand2,
  XCircle,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const ENTITY_LIST_LIMIT = 200;
const INSPECTOR_ENTITY_LIMIT = 80;
const INSPECTOR_TEXT_LIMIT = 16000;

type Layer = {
  name: string;
  color: string;
  lineweight?: number;
  linetype?: string;
  locked?: boolean;
  editable?: boolean;
};

type Entity = {
  id: string;
  type: string;
  layer: string;
  label?: string | null;
  [key: string]: unknown;
};

type DrawingIR = {
  units: "mm" | "inch";
  layers: Layer[];
  entities: Entity[];
  notes: string[];
};

type DiffItem = {
  path: string;
  before: unknown;
  after: unknown;
};

type Operation = {
  operation: string;
  entity_id?: string | null;
  reason?: string;
  [key: string]: unknown;
};

type SourceKind = "vector_pdf" | "scanned_pdf" | "image";

type Project = {
  project_id: string;
  name: string;
  source_file?: string | null;
  source_image?: string | null;
  source_kind?: SourceKind | null;
  ir: DrawingIR;
  history: Operation[];
  diffs: DiffItem[];
  ocr_regions: OcrRegion[];
  table_ocr_cells: TableCellOcr[];
  title_block_cells: TitleBlockCell[];
  dimension_bindings: DimensionBinding[];
  mechanical_dimensions: MechanicalDimension[];
  mechanical_ir: MechanicalDrawingIR;
  mechanical_transactions?: MechanicalTransaction[];
  dimension_ground_truth: DimensionGroundTruth[];
  dimension_corrections: DimensionCorrection[];
  semantic_repair_runs?: SemanticRepairRun[];
  agent_task_runs?: AgentTaskRun[];
  agent_eval_reports?: AgentEvalReport[];
  updated_at: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

type TargetEval = {
  name: string;
  score: number;
  passed: boolean;
  missing: string[];
};

type StructureEvalReport = {
  overall_score: number;
  passed: boolean;
  targets: TargetEval[];
};

type RegionTarget = "title_block" | "parameter_table" | "section_view" | "circular_view" | "dimensions";
type OcrLanguage = "auto" | "zh" | "en";

type RegionBox = {
  target: RegionTarget;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
};

type OcrRegion = RegionBox & {
  text: string;
  engine: string;
  language: string;
  source: string;
};

type TableCellOcr = {
  target: "title_block" | "parameter_table";
  row: number;
  col: number;
  text: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  engine: string;
  language: string;
  source: string;
};

type TitleBlockCell = {
  row: number;
  col: number;
  row_span: number;
  col_span: number;
  text: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  provider: string;
  source: string;
};

type ParsedDimensionValue = {
  kind: "linear" | "diameter" | "radius" | "roughness" | "tolerance" | "unknown";
  raw_text: string;
  nominal?: number | null;
  upper_tol?: number | null;
  lower_tol?: number | null;
  unit: string;
};

type DimensionBinding = {
  id: string;
  dimension_line_id: string;
  arrow_ids: string[];
  text_id?: string | null;
  text: string;
  parsed: ParsedDimensionValue;
  confidence: number;
  kind: ParsedDimensionValue["kind"];
  line_x1: number;
  line_y1: number;
  line_x2: number;
  line_y2: number;
  text_x?: number | null;
  text_y?: number | null;
  binding_method?: string;
  graph_path?: string[];
  graph_score?: number | null;
  source: string;
};

type MechanicalArrowhead = {
  candidate_id: string;
  source_entity_id: string;
  render_entity_id: string;
  tip_x: number;
  tip_y: number;
  direction_x: number;
  direction_y: number;
  score: number;
  endpoint: string;
  endpoint_distance: number;
};

type MechanicalDimension = {
  id: string;
  binding_id: string;
  kind: ParsedDimensionValue["kind"];
  text: string;
  parsed: ParsedDimensionValue;
  confidence: number;
  dimension_line_id: string;
  text_id?: string | null;
  arrowheads: MechanicalArrowhead[];
  extension_line_ids: string[];
  measured_geometry_ids: string[];
  target_geometry_ids: string[];
  measurement_points: number[][];
  dimension_line_point?: number[] | null;
  relation_confidence: Record<string, number>;
  orientation: "horizontal" | "vertical" | "aligned" | "radial";
  dxf_dimension_type?: "linear" | "aligned" | "diameter" | "radius" | null;
  export_ready: boolean;
  edit_mode: "annotation_override" | "driving";
  measured_value?: number | null;
  last_edit_source?: "deterministic" | "deepseek" | null;
  validation_status?: "passed" | "failed" | null;
  status: "complete" | "partial" | "unresolved";
  issues: string[];
  evidence: string[];
  source: string;
};

type MechanicalTransaction = {
  id: string;
  command: string;
  planner_source: string;
  validation: {
    passed: boolean;
    target_value: number;
    measured_value?: number | null;
  };
};

type MechanicalDrawingIR = {
  schema_version: "1.0";
  units: "mm" | "inch";
  dimensions: MechanicalDimension[];
  entity_roles: Record<string, string[]>;
  unresolved_binding_ids: string[];
  warnings: string[];
  source: string;
};

type DimensionGroundTruth = {
  id: string;
  label: string;
  expected_text: string;
  kind: ParsedDimensionValue["kind"];
  nominal?: number | null;
  unit: "mm" | "inch";
  matched_dimension_id?: string | null;
  required_relations: string[];
  notes: string;
  source: "seed" | "manual";
};

type DimensionCorrection = {
  ground_truth_id: string;
  dimension_id: string;
  text_id?: string | null;
  dimension_line_id: string;
  arrow_entity_ids: string[];
  extension_line_ids: string[];
  measured_geometry_ids: string[];
  updated_at: string;
};

type DimensionTargetEval = {
  ground_truth: DimensionGroundTruth;
  matched_dimension_id?: string | null;
  matched_text: string;
  score: number;
  passed: boolean;
  corrected: boolean;
  metrics: Record<string, number>;
  missing_relations: string[];
};

type DimensionBenchmarkReport = {
  project_id: string;
  target_count: number;
  matched_count: number;
  complete_count: number;
  overall_score: number;
  metrics: Record<string, number>;
  targets: DimensionTargetEval[];
};

type DimensionBenchmarkResponse = {
  project: Project;
  report: DimensionBenchmarkReport;
};

type SemanticRepairStep = {
  index: number;
  ground_truth_id: string;
  label: string;
  status: "accepted" | "rejected" | "skipped" | "error";
  tool_calls: string[];
  score_before: number;
  score_after: number;
  overall_before: number;
  overall_after: number;
  missing_before: string[];
  missing_after: string[];
  selected_entities: Record<string, string[]>;
  detail: string;
};

type SemanticRepairRun = {
  id: string;
  created_at: string;
  planner_source: "deepseek" | "deterministic" | "deterministic_fallback";
  planner_model?: string | null;
  planner_reason: string;
  budget: number;
  llm_calls: number;
  before_score: number;
  after_score: number;
  accepted_steps: number;
  rejected_steps: number;
  stopped_reason: string;
  snapshot_file?: string | null;
  steps: SemanticRepairStep[];
  rolled_back_at?: string | null;
};

type SemanticRepairResponse = {
  project: Project;
  report: DimensionBenchmarkReport;
  run: SemanticRepairRun;
};

type AgentToolName =
  | "inspect_drawing"
  | "evaluate_drawing"
  | "evaluate_dimensions"
  | "repair_dimensions"
  | "drive_dimension"
  | "edit_cad"
  | "export_dxf";

type AgentToolArguments = {
  message?: string | null;
  dimension_id?: string | null;
  target_value?: number | null;
  anchor: "start" | "end";
  max_steps: number;
  min_gain: number;
};

type AgentPlannedStep = {
  call_id: string;
  tool: AgentToolName;
  arguments: AgentToolArguments;
  reason: string;
};

type AgentTaskStepTrace = {
  index: number;
  attempt: number;
  call_id: string;
  tool: AgentToolName;
  status: "accepted" | "skipped" | "error" | "rolled_back";
  arguments: AgentToolArguments;
  reason: string;
  observation: string;
  output: Record<string, unknown>;
  validation: Record<string, unknown>;
  dimension_score_before?: number | null;
  dimension_score_after?: number | null;
  mutating: boolean;
  reversible: boolean;
  started_at: string;
  completed_at: string;
};

type AgentTaskRun = {
  id: string;
  goal: string;
  status: "completed" | "partial" | "failed" | "rolled_back";
  planner_source: "deepseek" | "deterministic" | "deterministic_fallback";
  planner_model?: string | null;
  planner_reason: string;
  initial_plan: AgentPlannedStep[];
  steps: AgentTaskStepTrace[];
  llm_calls: number;
  replan_count: number;
  policy_injected_steps: number;
  max_tool_calls: number;
  before_dimension_score?: number | null;
  after_dimension_score?: number | null;
  summary: string;
  artifacts: Record<string, string>;
  snapshot_file?: string | null;
  created_at: string;
  completed_at: string;
  rolled_back_at?: string | null;
};

type AgentTaskResponse = {
  project: Project;
  run: AgentTaskRun;
};

type AgentEvalCaseResult = {
  case_id: string;
  category: string;
  goal: string;
  passed: boolean;
  task_success: boolean;
  score: number;
  status_match: boolean;
  tool_precision: number;
  tool_recall: number;
  tool_order_match: boolean;
  argument_accuracy: number;
  assertion_accuracy: number;
  safety_passed: boolean;
  expected_tools: AgentToolName[];
  actual_tools: AgentToolName[];
  failed_assertions: string[];
  invalid_action_count: number;
  rollback_observed: boolean;
  replan_count: number;
  llm_calls: number;
  policy_injected_steps: number;
  duration_ms: number;
  run_status: string;
  planner_source: string;
};

type AgentEvalMetrics = {
  task_success_rate: number;
  average_score: number;
  tool_selection_precision: number;
  tool_selection_recall: number;
  tool_order_accuracy: number;
  argument_accuracy: number;
  assertion_accuracy: number;
  safety_pass_rate: number;
  invalid_action_rate: number;
  rollback_success_rate: number;
  replan_success_rate: number;
  average_tool_calls: number;
  average_llm_calls: number;
  average_replans: number;
  average_duration_ms: number;
  average_policy_injected_steps: number;
};

type AgentEvalReport = {
  id: string;
  dataset_version: string;
  mode: "deterministic" | "deepseek";
  case_count: number;
  passed_count: number;
  metrics: AgentEvalMetrics;
  cases: AgentEvalCaseResult[];
  created_at: string;
  completed_at: string;
};

type AgentEvalResponse = {
  project: Project;
  report: AgentEvalReport;
};

type DimensionCorrectionRole =
  | "text"
  | "dimension_line"
  | "arrowheads"
  | "extension_lines"
  | "measured_geometry";

type DimensionCorrectionDraft = {
  dimensionId?: string | null;
  textId?: string | null;
  dimensionLineId: string;
  arrowEntityIds: string[];
  extensionLineIds: string[];
  measuredGeometryIds: string[];
};

type AnalyzeReport = {
  project_id: string;
  source_image: string;
  image_width?: number | null;
  image_height?: number | null;
  overlay_image?: string | null;
  preprocessed_image?: string | null;
  frame?: number[] | null;
  deskew_angle: number;
  boxes: RegionBox[];
};

type ReconstructionRegion = {
  target: "title_block" | "parameter_table" | "section_view";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type TableReconstructionResponse = {
  project: Project;
  layout_passed: boolean;
  warnings: string[];
  regions: ReconstructionRegion[];
};

type SectionReconstructionResponse = {
  project: Project;
  line_count: number;
  hatch_count: number;
  warnings: string[];
  region: ReconstructionRegion;
};

type ScanCadReconstructionResponse = {
  project: Project;
  entity_count: number;
  trace_count: number;
  structured_counts: Record<string, number>;
  source: string;
  warnings: string[];
};

type ScanPromotionResponse = {
  project: Project;
  promoted_counts: Record<string, number>;
  source_count: number;
  warnings: string[];
};

type VectorizerToolResult = {
  name: string;
  status: string;
  elapsed_sec: number;
  svg_url?: string | null;
  dxf_url?: string | null;
  preview_url?: string | null;
  svg_path_count: number;
  dxf_entity_count: number;
  output_bytes: number;
  detail: string;
};

type VectorizerBenchmarkResponse = {
  project_id: string;
  prepared_image_url: string;
  results: VectorizerToolResult[];
};

type VectorReconstructionResponse = {
  project: Project;
  entity_count: number;
  source: string;
  preview_source: string;
  dxf_source: string;
  warnings: string[];
};

type OcrResponse = {
  project: Project;
  regions: OcrRegion[];
  warnings: string[];
};

type TableOcrResponse = {
  project: Project;
  cells: TableCellOcr[];
  warnings: string[];
};

type DimensionSemanticsResponse = {
  project: Project;
  bindings: DimensionBinding[];
  warnings: string[];
};

type CadPipelineStep = {
  name: string;
  status: "ok" | "skipped" | "warning";
  detail: string;
};

type CadPipelineResponse = {
  project: Project;
  steps: CadPipelineStep[];
  warnings: string[];
};

const EVAL_LABELS: Record<string, string> = {
  title_block: "标题栏",
  parameter_table: "参数表",
  section_view: "剖视图",
  circular_view: "圆视图",
  dimensions: "尺寸标注",
  scan_trace: "Trace 层",
  scan_visual_match: "视觉相似",
  scan_tables: "表格清理",
  scan_lineweights: "线宽",
  scan_primitive_quality: "Primitive",
  scan_noise: "噪声",
};

const DIMENSION_ROLE_LABELS: Record<DimensionCorrectionRole, string> = {
  text: "文字",
  dimension_line: "尺寸线",
  arrowheads: "箭头",
  extension_lines: "界线",
  measured_geometry: "被测轮廓",
};

const CAD_LAYER_ORDER = [
  "REFERENCE_TRACE",
  "OUTLINE",
  "DIMENSION",
  "CENTER",
  "HATCH",
  "TEXT",
  "TITLE_BLOCK",
];

const LEGACY_LAYER_ALIASES: Record<string, string> = {
  reference_trace: "REFERENCE_TRACE",
  editable_linework: "OUTLINE",
  promoted_geometry: "OUTLINE",
  geometry: "OUTLINE",
  outline: "OUTLINE",
  holes: "OUTLINE",
  dimensions: "DIMENSION",
  centerline: "CENTER",
  hatch: "HATCH",
  notes: "TEXT",
  text: "TEXT",
  sheet: "TITLE_BLOCK",
  table: "TITLE_BLOCK",
  title_block: "TITLE_BLOCK",
};

const SVG_LAYER_STROKE_WIDTH: Record<string, number> = {
  REFERENCE_TRACE: 0.13,
  OUTLINE: 0.5,
  DIMENSION: 0.25,
  CENTER: 0.18,
  HATCH: 0.18,
  TEXT: 0.18,
  TITLE_BLOCK: 0.25,
};

const SVG_LAYER_STROKE_OPACITY: Record<string, number> = {
  REFERENCE_TRACE: 0.46,
  OUTLINE: 1,
  DIMENSION: 1,
  CENTER: 0.82,
  HATCH: 0.72,
  TEXT: 1,
  TITLE_BLOCK: 0.92,
};

const LAYER_LABELS: Record<string, string> = {
  REFERENCE_TRACE: "REFERENCE_TRACE",
  OUTLINE: "OUTLINE",
  DIMENSION: "DIMENSION",
  CENTER: "CENTER",
  HATCH: "HATCH",
  TEXT: "TEXT",
  TITLE_BLOCK: "TITLE_BLOCK",
};

type Point2 = [number, number];

type DrawingBounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

type LayerSummary = {
  name: string;
  color: string;
  count: number;
  locked: boolean;
  editable: boolean;
};

function targetLabel(name: string): string {
  return EVAL_LABELS[name] ?? name;
}

function layerLabel(name: string): string {
  return LAYER_LABELS[name] ?? name;
}

function canonicalLayerName(name: string): string {
  if (CAD_LAYER_ORDER.includes(name)) return name;
  return LEGACY_LAYER_ALIASES[name.toLowerCase()] ?? "OUTLINE";
}

function entityLayerName(entity: Entity): string {
  return canonicalLayerName(entity.layer);
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    let messageText = body || `${res.status} ${res.statusText}`;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      messageText = parsed.detail || messageText;
    } catch {
      // Keep the raw body for non-JSON errors.
    }
    throw new Error(messageText);
  }
  return res.json() as Promise<T>;
}

function entitySummary(entity: Entity): string {
  if (entity.type === "rectangle") {
    return `${entity.id}: rectangle ${entity.width} x ${entity.height}`;
  }
  if (entity.type === "circle") {
    return `${entity.id}: circle Ø${Number(entity.r) * 2} at (${entity.cx}, ${entity.cy})`;
  }
  if (entity.type === "line") {
    return `${entity.id}: line (${entity.x1}, ${entity.y1}) -> (${entity.x2}, ${entity.y2})`;
  }
  if (entity.type === "polyline") {
    const points = Array.isArray(entity.points) ? entity.points.length : 0;
    return `${entity.id}: polyline ${points} pts`;
  }
  if (entity.type === "arc") {
    return `${entity.id}: arc r${entity.r} at (${entity.cx}, ${entity.cy})`;
  }
  if (entity.type === "text") {
    return `${entity.id}: text "${entity.text}"`;
  }
  return `${entity.id}: ${entity.type}`;
}

function entityNumber(entity: Entity, key: string, fallback = 0): number {
  const value = entity[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }
  return fallback;
}

function entityText(entity: Entity): string {
  const value = entity.text;
  return typeof value === "string" ? value : "";
}

function entityPoints(entity: Entity): Point2[] {
  const value = entity.points;
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((point) => {
    if (!Array.isArray(point) || point.length < 2) {
      return [];
    }
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return [];
    }
    return [[x, y] as Point2];
  });
}

function entityTags(entity: Entity): string[] {
  return Array.isArray(entity.tags) ? entity.tags.filter((tag): tag is string => typeof tag === "string") : [];
}

function entityMetadata(entity: Entity): Record<string, unknown> {
  const metadata = entity.metadata;
  if (metadata && typeof metadata === "object" && !Array.isArray(metadata)) {
    return metadata as Record<string, unknown>;
  }
  return {};
}

function entityHasSolidFill(entity: Entity): boolean {
  return entityMetadata(entity).fill === true || entityTags(entity).includes("solid_fill");
}

function extendBounds(bounds: DrawingBounds, x: number, y: number) {
  bounds.minX = Math.min(bounds.minX, x);
  bounds.minY = Math.min(bounds.minY, y);
  bounds.maxX = Math.max(bounds.maxX, x);
  bounds.maxY = Math.max(bounds.maxY, y);
}

function computeBounds(entities: Entity[]): DrawingBounds {
  const bounds: DrawingBounds = {
    minX: Number.POSITIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
  };

  for (const entity of entities) {
    if (entity.type === "line") {
      extendBounds(bounds, entityNumber(entity, "x1"), entityNumber(entity, "y1"));
      extendBounds(bounds, entityNumber(entity, "x2"), entityNumber(entity, "y2"));
    } else if (entity.type === "polyline") {
      for (const [x, y] of entityPoints(entity)) {
        extendBounds(bounds, x, y);
      }
    } else if (entity.type === "circle" || entity.type === "arc") {
      const cx = entityNumber(entity, "cx");
      const cy = entityNumber(entity, "cy");
      const r = Math.max(0, entityNumber(entity, "r"));
      extendBounds(bounds, cx - r, cy - r);
      extendBounds(bounds, cx + r, cy + r);
    } else if (entity.type === "rectangle") {
      const x = entityNumber(entity, "x");
      const y = entityNumber(entity, "y");
      extendBounds(bounds, x, y);
      extendBounds(bounds, x + entityNumber(entity, "width"), y + entityNumber(entity, "height"));
    } else if (entity.type === "text") {
      const x = entityNumber(entity, "x");
      const y = entityNumber(entity, "y");
      const height = Math.max(1, entityNumber(entity, "height", 3.5));
      extendBounds(bounds, x, y - height);
      extendBounds(bounds, x + Math.max(height, entityText(entity).length * height * 0.62), y + height);
    }
  }

  if (!Number.isFinite(bounds.minX) || !Number.isFinite(bounds.minY)) {
    return { minX: 0, minY: 0, maxX: 100, maxY: 100 };
  }
  return bounds;
}

function strokeColor(_ir: DrawingIR, entity: Entity): string {
  return entityLayerName(entity) === "REFERENCE_TRACE" ? "#647084" : "#111827";
}

function strokeWidth(entity: Entity): number {
  const layerDefault = SVG_LAYER_STROKE_WIDTH[entityLayerName(entity)] ?? 0.25;
  const explicit = entityNumber(entity, "stroke_width", Number.NaN);
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.max(explicit, layerDefault);
  }
  return layerDefault;
}

function strokeOpacity(entity: Entity): number {
  return SVG_LAYER_STROKE_OPACITY[entityLayerName(entity)] ?? 1;
}

function layerSummaries(ir: DrawingIR): LayerSummary[] {
  const counts = new Map<string, number>();
  for (const entity of ir.entities) {
    const layer = entityLayerName(entity);
    counts.set(layer, (counts.get(layer) ?? 0) + 1);
  }
  const configuredLayers = new Map(ir.layers.map((layer) => [canonicalLayerName(layer.name), layer]));
  return CAD_LAYER_ORDER
    .map((name) => ({
      name,
      color: name === "REFERENCE_TRACE" ? "gray" : "white",
      count: counts.get(name) ?? 0,
      locked: configuredLayers.get(name)?.locked ?? name === "REFERENCE_TRACE",
      editable: configuredLayers.get(name)?.editable ?? name !== "REFERENCE_TRACE",
    }))
    .filter((layer) => layer.count > 0);
}

function entityArcPath(entity: Entity): string {
  const cx = entityNumber(entity, "cx");
  const cy = entityNumber(entity, "cy");
  const r = entityNumber(entity, "r");
  const startAngle = (entityNumber(entity, "start_angle") * Math.PI) / 180;
  const endAngle = (entityNumber(entity, "end_angle") * Math.PI) / 180;
  const x1 = cx + Math.cos(startAngle) * r;
  const y1 = cy + Math.sin(startAngle) * r;
  const x2 = cx + Math.cos(endAngle) * r;
  const y2 = cy + Math.sin(endAngle) * r;
  const delta = ((entityNumber(entity, "end_angle") - entityNumber(entity, "start_angle")) % 360 + 360) % 360;
  const largeArc = delta > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}

function renderCadEntity(
  ir: DrawingIR,
  entity: Entity,
  idx: number,
  highlightedEntityIds: Set<string>,
  onEntityClick?: (entityId: string) => void,
): React.ReactNode {
  const highlighted = highlightedEntityIds.has(entity.id);
  const baseStroke = strokeColor(ir, entity);
  const drawStroke = highlighted ? "#f97316" : baseStroke;
  const drawWidth = highlighted ? Math.max(strokeWidth(entity) * 1.8, 0.75) : strokeWidth(entity);
  const common = {
    stroke: drawStroke,
    strokeWidth: drawWidth,
    strokeOpacity: highlighted ? 1 : strokeOpacity(entity),
    strokeDasharray: entityLayerName(entity) === "CENTER" ? "2.5 0.8 0.4 0.8" : undefined,
    className: highlighted ? "cad-highlight" : undefined,
    "data-id": entity.id,
    "data-layer": entityLayerName(entity),
    "data-group": typeof entity.group === "string" ? entity.group : undefined,
    onClick: onEntityClick ? () => onEntityClick(entity.id) : undefined,
    style: onEntityClick ? { cursor: "crosshair" } : undefined,
  };
  const key = `${entity.id}-${idx}`;

  if (entity.type === "line") {
    return (
      <line
        key={key}
        x1={entityNumber(entity, "x1")}
        y1={entityNumber(entity, "y1")}
        x2={entityNumber(entity, "x2")}
        y2={entityNumber(entity, "y2")}
        {...common}
      />
    );
  }
  if (entity.type === "polyline") {
    const points = entityPoints(entity).map(([x, y]) => `${x},${y}`).join(" ");
    if (!points) return null;
    return entity.closed ? (
      <polygon key={key} points={points} fill={entityHasSolidFill(entity) ? drawStroke : "none"} {...common} />
    ) : (
      <polyline key={key} points={points} {...common} />
    );
  }
  if (entity.type === "circle") {
    return (
      <circle
        key={key}
        cx={entityNumber(entity, "cx")}
        cy={entityNumber(entity, "cy")}
        r={entityNumber(entity, "r")}
        {...common}
      />
    );
  }
  if (entity.type === "arc") {
    return <path key={key} d={entityArcPath(entity)} {...common} />;
  }
  if (entity.type === "rectangle") {
    return (
      <rect
        key={key}
        x={entityNumber(entity, "x")}
        y={entityNumber(entity, "y")}
        width={entityNumber(entity, "width")}
        height={entityNumber(entity, "height")}
        {...common}
      />
    );
  }
  return null;
}

function renderCadText(
  ir: DrawingIR,
  entity: Entity,
  idx: number,
  highlightedEntityIds: Set<string>,
  onEntityClick?: (entityId: string) => void,
): React.ReactNode {
  const text = entityText(entity);
  if (!text) return null;
  const highlighted = highlightedEntityIds.has(entity.id);
  const x = entityNumber(entity, "x");
  const y = -entityNumber(entity, "y");
  const rotation = entityNumber(entity, "rotation");
  const transform = rotation ? `rotate(${-rotation} ${x} ${y})` : undefined;
  return (
    <text
      key={`${entity.id}-${idx}`}
      x={x}
      y={y}
      fill={highlighted ? "#f97316" : strokeColor(ir, entity)}
      fontSize={entityNumber(entity, "height", 3.5)}
      fontWeight={highlighted ? 700 : 400}
      fontFamily="Songti SC, STSong, SimSun, Noto Serif CJK SC, serif"
      transform={transform}
      className={highlighted ? "cad-highlight" : undefined}
      data-id={entity.id}
      data-layer={entityLayerName(entity)}
      data-group={typeof entity.group === "string" ? entity.group : undefined}
      onClick={onEntityClick ? () => onEntityClick(entity.id) : undefined}
      style={onEntityClick ? { cursor: "crosshair" } : undefined}
    >
      {text}
    </text>
  );
}

function CadPreview({
  ir,
  hiddenLayers,
  highlightedEntityIds = new Set<string>(),
  onEntityClick,
}: {
  ir: DrawingIR;
  hiddenLayers: Set<string>;
  highlightedEntityIds?: Set<string>;
  onEntityClick?: (entityId: string) => void;
}) {
  const bounds = computeBounds(ir.entities);
  const rawWidth = Math.max(bounds.maxX - bounds.minX, 1);
  const rawHeight = Math.max(bounds.maxY - bounds.minY, 1);
  const margin = Math.max(rawWidth, rawHeight) * 0.08;
  const minX = bounds.minX - margin;
  const maxY = bounds.maxY + margin;
  const width = rawWidth + margin * 2;
  const height = rawHeight + margin * 2;
  const visible = ir.entities.filter((entity) => !hiddenLayers.has(entityLayerName(entity)));
  const geometry = visible.filter((entity) => entity.type !== "text");
  const text = visible.filter((entity) => entity.type === "text");

  return (
    <svg
      className="cad-preview-svg"
      viewBox={`${minX} ${-maxY} ${width} ${height}`}
      role="img"
      aria-label="CAD IR preview"
    >
      <g transform="scale(1,-1)" fill="none" strokeLinecap="round" strokeLinejoin="round">
        {geometry.map((entity, idx) => renderCadEntity(ir, entity, idx, highlightedEntityIds, onEntityClick))}
      </g>
      {text.map((entity, idx) => renderCadText(ir, entity, idx, highlightedEntityIds, onEntityClick))}
    </svg>
  );
}

function compactIrForInspector(ir: DrawingIR) {
  if (ir.entities.length <= INSPECTOR_ENTITY_LIMIT) {
    return ir;
  }
  return {
    ...ir,
    entity_count: ir.entities.length,
    entities_preview: ir.entities.slice(0, INSPECTOR_ENTITY_LIMIT),
    entities_preview_truncated: ir.entities.length - INSPECTOR_ENTITY_LIMIT,
    entities: `[${ir.entities.length} entities hidden in inspector preview]`,
  };
}

function inspectorText(value: unknown): string {
  const text = JSON.stringify(value, null, 2);
  if (text.length <= INSPECTOR_TEXT_LIMIT) {
    return text;
  }
  return `${text.slice(0, INSPECTOR_TEXT_LIMIT)}\n... truncated ${text.length - INSPECTOR_TEXT_LIMIT} chars for UI preview`;
}

function dimensionValueSummary(binding: DimensionBinding): string {
  const parts: string[] = [binding.kind];
  if (binding.parsed.nominal !== null && binding.parsed.nominal !== undefined) {
    parts.push(`${binding.parsed.nominal}${binding.parsed.unit === "mm" ? "" : ` ${binding.parsed.unit}`}`);
  }
  const upper = binding.parsed.upper_tol;
  const lower = binding.parsed.lower_tol;
  if (upper !== null && upper !== undefined && lower !== null && lower !== undefined) {
    parts.push(`tol +${upper} / ${lower}`);
  } else if (upper !== null && upper !== undefined) {
    parts.push(`tol +${upper}`);
  } else if (lower !== null && lower !== undefined) {
    parts.push(`tol ${lower}`);
  }
  return parts.join(" · ");
}

function mechanicalDimensionSummary(dimension: MechanicalDimension): string {
  const parts: string[] = [dimension.kind];
  if (dimension.parsed.nominal !== null && dimension.parsed.nominal !== undefined) {
    parts.push(`${dimension.parsed.nominal}${dimension.parsed.unit === "mm" ? "" : ` ${dimension.parsed.unit}`}`);
  }
  const upper = dimension.parsed.upper_tol;
  const lower = dimension.parsed.lower_tol;
  if (upper !== null && upper !== undefined && lower !== null && lower !== undefined) {
    parts.push(`tol +${upper} / ${lower}`);
  } else if (upper !== null && upper !== undefined) {
    parts.push(`tol +${upper}`);
  } else if (lower !== null && lower !== undefined) {
    parts.push(`tol ${lower}`);
  }
  return parts.join(" · ");
}

function mechanicalDimensionEntityIds(dimension: MechanicalDimension | null | undefined): Set<string> {
  const ids = new Set<string>();
  if (!dimension) return ids;
  ids.add(dimension.dimension_line_id);
  if (dimension.text_id) ids.add(dimension.text_id);
  for (const arrow of dimension.arrowheads) {
    ids.add(arrow.render_entity_id);
    ids.add(arrow.source_entity_id);
  }
  for (const id of dimension.extension_line_ids ?? []) {
    ids.add(id);
  }
  for (const id of dimension.measured_geometry_ids ?? []) {
    ids.add(id);
  }
  for (const id of dimension.target_geometry_ids ?? []) {
    ids.add(id);
  }
  return ids;
}

function correctionDraftEntityIds(draft: DimensionCorrectionDraft | null): Set<string> {
  const ids = new Set<string>();
  if (!draft) return ids;
  if (draft.textId) ids.add(draft.textId);
  if (draft.dimensionLineId) ids.add(draft.dimensionLineId);
  for (const id of draft.arrowEntityIds) ids.add(id);
  for (const id of draft.extensionLineIds) ids.add(id);
  for (const id of draft.measuredGeometryIds) ids.add(id);
  return ids;
}

function draftForDimension(
  dimension: MechanicalDimension | null,
  correction: DimensionCorrection | undefined,
): DimensionCorrectionDraft {
  return {
    dimensionId: correction?.dimension_id ?? dimension?.id ?? null,
    textId: correction?.text_id ?? dimension?.text_id ?? null,
    dimensionLineId: correction?.dimension_line_id ?? dimension?.dimension_line_id ?? "",
    arrowEntityIds:
      correction?.arrow_entity_ids ?? dimension?.arrowheads.map((arrow) => arrow.render_entity_id) ?? [],
    extensionLineIds: correction?.extension_line_ids ?? dimension?.extension_line_ids ?? [],
    measuredGeometryIds:
      correction?.measured_geometry_ids ?? dimension?.measured_geometry_ids ?? dimension?.target_geometry_ids ?? [],
  };
}

function toggleEntityId(ids: string[], entityId: string): string[] {
  return ids.includes(entityId) ? ids.filter((id) => id !== entityId) : [...ids, entityId];
}

function correctionIdsForRole(draft: DimensionCorrectionDraft, role: DimensionCorrectionRole): string[] {
  if (role === "text") return draft.textId ? [draft.textId] : [];
  if (role === "dimension_line") return draft.dimensionLineId ? [draft.dimensionLineId] : [];
  if (role === "arrowheads") return draft.arrowEntityIds;
  if (role === "extension_lines") return draft.extensionLineIds;
  return draft.measuredGeometryIds;
}

function SemanticRelationRow({
  label,
  ids,
  dimension,
}: {
  label: string;
  ids: string[];
  dimension: MechanicalDimension;
}) {
  const confidences = ids
    .map((id) => dimension.relation_confidence?.[id])
    .filter((value): value is number => Number.isFinite(value));
  const confidence = confidences.length ? Math.round(Math.min(...confidences) * 100) : null;
  return (
    <div className={`semantic-relation ${ids.length ? "bound" : "missing"}`}>
      <div>
        <strong>{label}</strong>
        <small>{ids.length ? `${ids.length} bound` : "missing"}</small>
      </div>
      <code>{ids.join("\n") || "--"}</code>
      {confidence !== null ? <span>{confidence}%</span> : null}
    </div>
  );
}

function graphPathSummary(binding: DimensionBinding): string {
  const path = binding.graph_path ?? [];
  if (!path.length) {
    return "graph path: --";
  }
  return path.join(" -> ");
}

function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [evalReport, setEvalReport] = useState<StructureEvalReport | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeReport | null>(null);
  const [message, setMessage] = useState("把左边孔直径改成 10");
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [reconstructing, setReconstructing] = useState(false);
  const [ocring, setOcring] = useState(false);
  const [semanticing, setSemanticing] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [ocrLanguage, setOcrLanguage] = useState<OcrLanguage>("auto");
  const [layoutNotice, setLayoutNotice] = useState("");
  const [error, setError] = useState("");
  const [tab, setTab] = useState<"ir" | "diff" | "history" | "ocr" | "cells" | "dims">("ir");
  const [assetRevision, setAssetRevision] = useState(0);
  const [hiddenPreviewLayers, setHiddenPreviewLayers] = useState<string[]>([]);
  const [selectedMechanicalDimensionId, setSelectedMechanicalDimensionId] = useState<string | null>(null);
  const [dimensionBenchmark, setDimensionBenchmark] = useState<DimensionBenchmarkReport | null>(null);
  const [selectedGroundTruthId, setSelectedGroundTruthId] = useState<string | null>(null);
  const [correctionRole, setCorrectionRole] = useState<DimensionCorrectionRole>("dimension_line");
  const [correctionDraft, setCorrectionDraft] = useState<DimensionCorrectionDraft | null>(null);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [agentEvalMode, setAgentEvalMode] = useState<"deterministic" | "deepseek">("deterministic");
  const [agentEvalBusy, setAgentEvalBusy] = useState(false);
  const cacheKey = project ? `${encodeURIComponent(project.updated_at)}-${assetRevision}` : "";

  function loadProject(next: Project) {
    setProject(next);
    setAssetRevision(Date.now());
  }

  useEffect(() => {
    const projectId = new URLSearchParams(window.location.search).get("project");
    if (projectId) {
      void loadExistingProject(projectId);
      return;
    }
    void createProject();
  }, []);

  useEffect(() => {
    if (!project) {
      setHiddenPreviewLayers([]);
      setSelectedMechanicalDimensionId(null);
      setDimensionBenchmark(null);
      setSelectedGroundTruthId(null);
      setCorrectionDraft(null);
      return;
    }
    const existingLayers = new Set(project.ir.entities.map((entity) => entity.layer));
    setHiddenPreviewLayers((items) => items.filter((layer) => existingLayers.has(layer)));
    setSelectedMechanicalDimensionId((id) => {
      const dimensions = project.mechanical_ir?.dimensions?.length
        ? project.mechanical_ir.dimensions
        : project.mechanical_dimensions ?? [];
      if (id && dimensions.some((dimension) => dimension.id === id)) {
        return id;
      }
      return dimensions[0]?.id ?? null;
    });
  }, [project?.project_id, project?.updated_at]);

  useEffect(() => {
    if (!project) {
      setEvalReport(null);
      return;
    }
    let cancelled = false;
    const scanEval = project.source_kind === "scanned_pdf" || project.source_kind === "image";
    const path = scanEval ? `/api/projects/${project.project_id}/eval/scan` : `/api/projects/${project.project_id}/eval`;
    void api<StructureEvalReport>(path)
      .then((report) => {
        if (!cancelled) setEvalReport(report);
      })
      .catch(() => {
        if (!cancelled) setEvalReport(null);
      });
    void api<DimensionBenchmarkResponse>(`/api/projects/${project.project_id}/eval/dimensions`)
      .then((response) => {
        if (!cancelled) setDimensionBenchmark(response.report);
      })
      .catch(() => {
        if (!cancelled) setDimensionBenchmark(null);
      });
    return () => {
      cancelled = true;
    };
  }, [project?.project_id, project?.source_kind, project?.updated_at]);

  const visibleEntities = project?.ir.entities.slice(0, ENTITY_LIST_LIMIT) ?? [];
  const hiddenEntityCount = project ? Math.max(0, project.ir.entities.length - ENTITY_LIST_LIMIT) : 0;
  const previewLayers = useMemo(() => (project ? layerSummaries(project.ir) : []), [project?.ir]);
  const hiddenPreviewLayerSet = useMemo(() => new Set(hiddenPreviewLayers), [hiddenPreviewLayers]);
  const mechanicalDimensions = useMemo(() => {
    if (project?.mechanical_ir?.dimensions?.length) return project.mechanical_ir.dimensions;
    return project?.mechanical_dimensions ?? [];
  }, [project?.mechanical_ir, project?.mechanical_dimensions]);
  const selectedMechanicalDimension = useMemo(() => {
    const dimensions = mechanicalDimensions;
    if (!dimensions.length) return null;
    return dimensions.find((dimension) => dimension.id === selectedMechanicalDimensionId) ?? dimensions[0];
  }, [mechanicalDimensions, selectedMechanicalDimensionId]);
  const selectedDimensionBinding = useMemo(
    () => project?.dimension_bindings?.find((binding) => binding.id === selectedMechanicalDimension?.binding_id) ?? null,
    [project?.dimension_bindings, selectedMechanicalDimension?.binding_id],
  );
  const selectedBenchmarkTarget = useMemo(
    () => dimensionBenchmark?.targets.find((target) => target.ground_truth.id === selectedGroundTruthId) ?? null,
    [dimensionBenchmark, selectedGroundTruthId],
  );
  const latestSemanticRepairRun = useMemo(() => {
    const runs = project?.semantic_repair_runs ?? [];
    return runs.length ? runs[runs.length - 1] : null;
  }, [project?.semantic_repair_runs]);
  const latestAgentTaskRun = useMemo(() => {
    const runs = project?.agent_task_runs ?? [];
    return runs.length ? runs[runs.length - 1] : null;
  }, [project?.agent_task_runs]);
  const latestAgentEvalReport = useMemo(() => {
    const reports = project?.agent_eval_reports ?? [];
    return reports.length ? reports[reports.length - 1] : null;
  }, [project?.agent_eval_reports]);
  const hasPendingLinearDimension = useMemo(
    () => dimensionBenchmark?.targets.some(
      (target) => target.ground_truth.kind === "linear" && !target.passed,
    ) ?? false,
    [dimensionBenchmark],
  );

  useEffect(() => {
    const targets = dimensionBenchmark?.targets ?? [];
    if (!targets.length) {
      setSelectedGroundTruthId(null);
      return;
    }
    setSelectedGroundTruthId((current) => {
      if (current && targets.some((target) => target.ground_truth.id === current)) return current;
      return targets.find((target) => !target.passed)?.ground_truth.id ?? targets[0].ground_truth.id;
    });
  }, [dimensionBenchmark]);

  useEffect(() => {
    if (!project || !selectedBenchmarkTarget) {
      setCorrectionDraft(null);
      return;
    }
    const correction = project.dimension_corrections?.find(
      (item) => item.ground_truth_id === selectedBenchmarkTarget.ground_truth.id,
    );
    const dimensionId = correction?.dimension_id ?? selectedBenchmarkTarget.matched_dimension_id;
    const dimension = mechanicalDimensions.find((item) => item.id === dimensionId) ?? null;
    setCorrectionDraft(draftForDimension(dimension, correction));
    if (dimension) setSelectedMechanicalDimensionId(dimension.id);
  }, [project?.project_id, project?.updated_at, selectedBenchmarkTarget?.ground_truth.id]);

  const highlightedEntityIds = useMemo(
    () => {
      const ids = mechanicalDimensionEntityIds(tab === "dims" ? selectedMechanicalDimension : null);
      if (tab === "dims") {
        for (const id of correctionDraftEntityIds(correctionDraft)) ids.add(id);
      }
      return ids;
    },
    [correctionDraft, selectedMechanicalDimension, tab],
  );
  const hasReferencePreviewLayer = previewLayers.some((layer) => layer.name === "REFERENCE_TRACE");
  const hasEditablePreviewLayers = previewLayers.some((layer) => layer.editable);
  const visiblePreviewEntityCount = project
    ? project.ir.entities.filter((entity) => !hiddenPreviewLayerSet.has(entityLayerName(entity))).length
    : 0;

  const dxfUrl = useMemo(() => {
    if (!project) return "";
    return `${API_BASE}/api/projects/${project.project_id}/files/output.dxf?v=${cacheKey}`;
  }, [project, cacheKey]);

  function setPreviewLayerVisible(layer: string, visible: boolean) {
    setHiddenPreviewLayers((items) => {
      if (visible) {
        return items.filter((item) => item !== layer);
      }
      return items.includes(layer) ? items : [...items, layer];
    });
  }

  function showAllPreviewLayers() {
    setHiddenPreviewLayers([]);
  }

  function showReferencePreviewLayer() {
    const nextHidden = previewLayers.map((layer) => layer.name).filter((name) => name !== "REFERENCE_TRACE");
    setHiddenPreviewLayers(nextHidden);
  }

  function showEditablePreviewLayers() {
    const nextHidden = previewLayers.filter((layer) => !layer.editable).map((layer) => layer.name);
    setHiddenPreviewLayers(nextHidden);
  }

  async function createProject(prompt = "创建 100 60 8 两个孔") {
    setBusy(true);
    setError("");
    try {
      const next = await api<Project>("/api/projects", {
        method: "POST",
        body: JSON.stringify({ name: "Vibe CAD demo", prompt }),
      });
      loadProject(next);
      setAnalysis(null);
      setLayoutNotice("");
      setChat([
        {
          role: "assistant",
          text: "已创建 baseline 图纸。你可以试试：画圆柱直齿轮图、把左边孔直径改成 10、右边孔右移 12、添加孔 50 30 6。",
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function loadExistingProject(projectId: string) {
    setBusy(true);
    setError("");
    try {
      const next = await api<Project>(`/api/projects/${encodeURIComponent(projectId)}`);
      loadProject(next);
      setAnalysis(null);
      setLayoutNotice("");
      setChat([
        {
          role: "assistant",
          text: `已加载项目 ${projectId}。`,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      await createProject();
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!project || !message.trim()) return;
    const userText = message.trim();
    setBusy(true);
    setError("");
    setLayoutNotice("");
    setChat((items) => [...items, { role: "user", text: userText }]);
    setMessage("");
    try {
      const res = await api<AgentTaskResponse>(
        `/api/projects/${project.project_id}/agent/tasks`,
        {
          method: "POST",
          body: JSON.stringify({
            goal: userText,
            use_llm: true,
            max_tool_calls: 8,
            max_replans: 1,
          }),
        },
      );
      loadProject(res.project);
      setLayoutNotice(
        `Agent ${res.run.status} · ${res.run.planner_source} · `
          + `${res.run.steps.length} tool calls · ${res.run.replan_count} replans · `
          + `${res.run.policy_injected_steps} policy checks`,
      );
      setChat((items) => [...items, { role: "assistant", text: res.run.summary }]);
      setTab("history");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function rollbackLatestAgentTask() {
    if (!project || !latestAgentTaskRun || latestAgentTaskRun.rolled_back_at) return;
    setBusy(true);
    setError("");
    try {
      const response = await api<AgentTaskResponse>(
        `/api/projects/${project.project_id}/agent/tasks/${latestAgentTaskRun.id}/rollback`,
        { method: "POST" },
      );
      loadProject(response.project);
      setLayoutNotice(response.run.summary);
      setChat((items) => [...items, { role: "assistant", text: response.run.summary }]);
      setTab("history");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runAgentEval() {
    if (!project) return;
    setAgentEvalBusy(true);
    setError("");
    try {
      const response = await api<AgentEvalResponse>(
        `/api/projects/${project.project_id}/agent/evals`,
        {
          method: "POST",
          body: JSON.stringify({ mode: agentEvalMode, max_cases: 12 }),
        },
      );
      loadProject(response.project);
      setLayoutNotice(
        `Agent Eval ${response.report.dataset_version} · `
          + `${response.report.passed_count}/${response.report.case_count} passed · `
          + `${Math.round(response.report.metrics.task_success_rate * 100)}%`,
      );
      setTab("history");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAgentEvalBusy(false);
    }
  }

  async function uploadFile(file: File | null) {
    if (!project || !file) return;
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    setError("");
    try {
      const next = await api<Project>(`/api/projects/${project.project_id}/upload`, {
        method: "POST",
        body: form,
      });
      loadProject(next);
      setAnalysis(null);
      setLayoutNotice("已上传参考图。点击 Generate CAD 一键完成重建、OCR 和尺寸绑定。");
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: "已上传参考图。点击 Generate CAD，我会自动选择矢量/扫描路径并跑完整流程。",
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function runCadPipeline(projectId = project?.project_id) {
    if (!projectId) return;
    setPipelineRunning(true);
    setError("");
    setLayoutNotice("");
    try {
      const res = await api<CadPipelineResponse>(`/api/projects/${projectId}/pipeline/cad`, {
        method: "POST",
        body: JSON.stringify({ language: ocrLanguage, engine: "auto", include_table_ocr: true }),
      });
      loadProject(res.project);
      setAnalysis(null);
      const stepSummary = res.steps.map((step) => `${step.name}:${step.status}`).join(" -> ");
      setLayoutNotice(
        [
          `Pipeline：${stepSummary}`,
          res.project.dimension_bindings.length
            ? `尺寸绑定=${res.project.dimension_bindings.length}`
            : "尺寸绑定=0",
          res.project.mechanical_dimensions?.length
            ? `机械语义=${res.project.mechanical_dimensions.length}`
            : "机械语义=0",
          ...res.warnings,
        ].join(" "),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `一键流程完成：${stepSummary}。尺寸绑定 ${res.project.dimension_bindings.length} 条，机械尺寸 ${res.project.mechanical_dimensions?.length ?? 0} 条。`,
        },
      ]);
      setTab(res.project.mechanical_dimensions?.length || res.project.dimension_bindings.length ? "dims" : "diff");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPipelineRunning(false);
    }
  }

  async function reconstructVector(projectId = project?.project_id) {
    if (!projectId) return;
    setReconstructing(true);
    setError("");
    try {
      const res = await api<VectorReconstructionResponse>(`/api/projects/${projectId}/reconstruct/vector`, {
        method: "POST",
      });
      loadProject(res.project);
      setAnalysis(null);
      setLayoutNotice(
        [
          `矢量 PDF：${res.entity_count} 个图元；预览=${res.preview_source}；DXF=${res.dxf_source}。`,
          ...res.warnings,
        ].join(" "),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `检测到矢量 PDF，右侧预览=${res.preview_source}；提取到 ${res.entity_count} 个图元。DXF 来源：${res.dxf_source}。`,
        },
      ]);
      setTab("ir");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReconstructing(false);
    }
  }

  async function reconstructScan(projectId = project?.project_id) {
    if (!projectId) return;
    setReconstructing(true);
    setError("");
    try {
      const res = await api<ScanCadReconstructionResponse>(`/api/projects/${projectId}/reconstruct/scan`, {
        method: "POST",
      });
      loadProject(res.project);
      const structured = Object.entries(res.structured_counts)
        .map(([name, count]) => `${name}=${count}`)
        .join(", ");
      setLayoutNotice(
        [
          `Scan CAD：${res.entity_count} entities，trace=${res.trace_count}`,
          structured ? `structured: ${structured}` : "",
          ...res.warnings,
        ].filter(Boolean).join("；"),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `已从扫描图生成 CAD 初稿：${res.entity_count} 个图元，其中 ${res.trace_count} 条来自整页 CV 线稿。`,
        },
      ]);
      setTab("diff");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReconstructing(false);
    }
  }

  async function promoteScan(projectId = project?.project_id) {
    if (!projectId) return;
    setReconstructing(true);
    setError("");
    try {
      const res = await api<ScanPromotionResponse>(`/api/projects/${projectId}/promote/scan`, {
        method: "POST",
      });
      loadProject(res.project);
      const promoted = Object.entries(res.promoted_counts)
        .map(([name, count]) => `${name}=${count}`)
        .join(", ");
      setLayoutNotice(
        [
          `Promote：从 ${res.source_count} 条 editable_linework 中提升 ${promoted || "0 primitives"}。`,
          ...res.warnings,
        ].join(" "),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `已把高置信度线稿提升为可编辑 CAD primitives：${promoted || "0"}。`,
        },
      ]);
      setTab("diff");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReconstructing(false);
    }
  }

  async function runVectorizerBenchmark(projectId = project?.project_id) {
    if (!projectId) return;
    setReconstructing(true);
    setError("");
    try {
      const res = await api<VectorizerBenchmarkResponse>(`/api/projects/${projectId}/benchmark/vectorizers`, {
        method: "POST",
      });
      const summary = res.results
        .map((item) => `${item.name}:${item.status} paths=${item.svg_path_count} dxf=${item.dxf_entity_count}`)
        .join("；");
      setLayoutNotice(`Open-source vectorizer benchmark：${summary}`);
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `已运行开源 vectorizer baseline：${summary}`,
        },
      ]);
      setTab("history");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReconstructing(false);
    }
  }

  async function runAnalysis(projectId = project?.project_id) {
    if (!projectId) return;
    setAnalyzing(true);
    setError("");
    try {
      const report = await api<AnalyzeReport>(`/api/projects/${projectId}/analyze`);
      setAnalysis(report);
      const categoryCount = new Set(report.boxes.map((box) => box.target)).size;
      setLayoutNotice(
        [
          `Scan analyze：${report.boxes.length} boxes / ${categoryCount} categories。`,
          `deskew=${report.deskew_angle.toFixed(2)}°。`,
          report.overlay_image ? "已生成 debug overlay。" : "",
        ].filter(Boolean).join(" "),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setAnalysis(null);
    } finally {
      setAnalyzing(false);
    }
  }

  async function reconstructTables() {
    if (!project?.source_image) return;
    setReconstructing(true);
    setError("");
    try {
      const res = await api<TableReconstructionResponse>(`/api/projects/${project.project_id}/reconstruct/tables`, {
        method: "POST",
      });
      loadProject(res.project);
      setLayoutNotice(
        res.layout_passed
          ? `Tables reconstructed: ${res.regions.length} regions, no overlap.`
          : `Layout warnings: ${res.warnings.join("; ")}`,
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: res.layout_passed
            ? "已从参考图重建参数表和标题栏，布局检查通过，没有区域重叠。"
            : `已重建参数表和标题栏，但布局检查发现：${res.warnings.join("；")}`,
        },
      ]);
      setTab("diff");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReconstructing(false);
    }
  }

  async function reconstructSection() {
    if (!project?.source_image) return;
    setReconstructing(true);
    setError("");
    try {
      const res = await api<SectionReconstructionResponse>(`/api/projects/${project.project_id}/reconstruct/section`, {
        method: "POST",
      });
      loadProject(res.project);
      setLayoutNotice(
        res.warnings.length
          ? `Section CV: ${res.line_count} lines, ${res.hatch_count} hatches. Warnings: ${res.warnings.join("; ")}`
          : `Section CV: ${res.line_count} lines, ${res.hatch_count} hatches.`,
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: res.warnings.length
            ? `已用 OpenCV 重建剖视图，检测到 ${res.line_count} 条线，其中 ${res.hatch_count} 条剖面线；需要关注：${res.warnings.join("；")}`
            : `已用 OpenCV 重建剖视图，检测到 ${res.line_count} 条线，其中 ${res.hatch_count} 条剖面线。`,
        },
      ]);
      setTab("diff");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setReconstructing(false);
    }
  }

  async function runOcr(projectId = project?.project_id) {
    if (!projectId) return;
    setOcring(true);
    setError("");
    try {
      const res = await api<OcrResponse>(`/api/projects/${projectId}/ocr`, {
        method: "POST",
        body: JSON.stringify({ language: ocrLanguage, engine: "auto" }),
      });
      loadProject(res.project);
      const textCount = res.regions.filter((region) => region.text.trim()).length;
      const engines = Array.from(new Set(res.regions.map((region) => region.engine))).join("/");
      setLayoutNotice(
        [
          `OCR(${engines || "none"})：${res.regions.length} 个区域，${textCount} 个区域有文本。`,
          ...res.warnings,
        ].join(" "),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `已对 ${res.regions.length} 个语义区域运行 OCR，${textCount} 个区域识别出文本。`,
        },
      ]);
      setTab("ocr");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setOcring(false);
    }
  }

  async function runTableOcr(projectId = project?.project_id) {
    if (!projectId) return;
    setOcring(true);
    setError("");
    try {
      const res = await api<TableOcrResponse>(`/api/projects/${projectId}/ocr/tables`, {
        method: "POST",
        body: JSON.stringify({ language: ocrLanguage, engine: "auto" }),
      });
      loadProject(res.project);
      const textCount = res.cells.filter((cell) => cell.text.trim()).length;
      const engines = Array.from(new Set(res.cells.map((cell) => cell.engine))).join("/");
      setLayoutNotice(
        [
          `Table OCR(${engines || "none"})：${res.cells.length} cells，${textCount} cells 有文本。`,
          ...res.warnings,
        ].join(" "),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `已对标题栏/参数表做 cell OCR，识别到 ${res.cells.length} 个有文本的单元格。`,
        },
      ]);
      setTab("cells");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setOcring(false);
    }
  }

  async function runDimensionSemantics(projectId = project?.project_id) {
    if (!projectId) return;
    setSemanticing(true);
    setError("");
    try {
      const res = await api<DimensionSemanticsResponse>(`/api/projects/${projectId}/semantics/dimensions`, {
        method: "POST",
      });
      loadProject(res.project);
      const complete = res.bindings.filter((binding) => binding.text_id).length;
      setLayoutNotice(
        [
          `Dimension semantics：${res.bindings.length} bindings，${complete} with text。`,
          ...res.warnings,
        ].join(" "),
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `已绑定尺寸线、箭头和尺寸文字：${res.bindings.length} 条候选，其中 ${complete} 条绑定到文字。`,
        },
      ]);
      setTab("dims");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSemanticing(false);
    }
  }

  async function seedDimensionBenchmark() {
    if (!project) return;
    setBenchmarkBusy(true);
    setError("");
    try {
      const response = await api<DimensionBenchmarkResponse>(
        `/api/projects/${project.project_id}/benchmark/dimensions/seed`,
        { method: "POST", body: JSON.stringify({ replace: false }) },
      );
      loadProject(response.project);
      setDimensionBenchmark(response.report);
      setTab("dims");
      setLayoutNotice(`尺寸基准：${response.report.target_count} 项，已匹配 ${response.report.matched_count} 项。`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function runAutomaticDimensionRepair() {
    if (!project) return;
    setBenchmarkBusy(true);
    setError("");
    try {
      const response = await api<SemanticRepairResponse>(
        `/api/projects/${project.project_id}/agent/dimensions/repair`,
        {
          method: "POST",
          body: JSON.stringify({ use_llm: true, max_steps: 3, min_gain: 0.01 }),
        },
      );
      loadProject(response.project);
      setDimensionBenchmark(response.report);
      const accepted = response.run.steps.find((step) => step.status === "accepted");
      if (accepted) setSelectedGroundTruthId(accepted.ground_truth_id);
      setTab("dims");
      setLayoutNotice(
        `自动修复：接受 ${response.run.accepted_steps} 步，`
          + `${Math.round(response.run.before_score * 100)}% -> ${Math.round(response.run.after_score * 100)}%。`,
      );
      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: `尺寸语义自动修复完成：${response.run.accepted_steps} 步通过单调评估，`
            + `${response.run.rejected_steps} 步回滚。规划器：${response.run.planner_source}。`,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function rollbackAutomaticDimensionRepair() {
    if (!project || !latestSemanticRepairRun || latestSemanticRepairRun.rolled_back_at) return;
    setBenchmarkBusy(true);
    setError("");
    try {
      const response = await api<SemanticRepairResponse>(
        `/api/projects/${project.project_id}/agent/dimensions/repair/${latestSemanticRepairRun.id}/rollback`,
        { method: "POST" },
      );
      loadProject(response.project);
      setDimensionBenchmark(response.report);
      setLayoutNotice(`已回滚自动修复，尺寸基准恢复为 ${Math.round(response.report.overall_score * 100)}%。`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBenchmarkBusy(false);
    }
  }

  function assignCorrectionEntity(entityId: string) {
    if (!project || !correctionDraft || tab !== "dims") return;
    const entity = project.ir.entities.find((item) => item.id === entityId);
    if (!entity) return;
    if (entityLayerName(entity) === "REFERENCE_TRACE") {
      setError("参考 Trace 层已锁定，请选择可编辑图元。");
      return;
    }
    const valid =
      (correctionRole === "text" && entity.type === "text") ||
      ((correctionRole === "dimension_line" || correctionRole === "extension_lines") && entity.type === "line") ||
      (correctionRole === "arrowheads" && (entity.type === "line" || entity.type === "polyline")) ||
      (correctionRole === "measured_geometry" && entity.type !== "text");
    if (!valid) {
      setError(`${DIMENSION_ROLE_LABELS[correctionRole]}不能绑定 ${entity.type} 图元。`);
      return;
    }
    setError("");
    setCorrectionDraft((current) => {
      if (!current) return current;
      if (correctionRole === "text") return { ...current, textId: entityId };
      if (correctionRole === "dimension_line") return { ...current, dimensionLineId: entityId };
      if (correctionRole === "arrowheads") {
        return { ...current, arrowEntityIds: toggleEntityId(current.arrowEntityIds, entityId) };
      }
      if (correctionRole === "extension_lines") {
        return { ...current, extensionLineIds: toggleEntityId(current.extensionLineIds, entityId) };
      }
      return { ...current, measuredGeometryIds: toggleEntityId(current.measuredGeometryIds, entityId) };
    });
  }

  function clearCorrectionRole() {
    setCorrectionDraft((current) => {
      if (!current) return current;
      if (correctionRole === "text") return { ...current, textId: null };
      if (correctionRole === "dimension_line") return { ...current, dimensionLineId: "" };
      if (correctionRole === "arrowheads") return { ...current, arrowEntityIds: [] };
      if (correctionRole === "extension_lines") return { ...current, extensionLineIds: [] };
      return { ...current, measuredGeometryIds: [] };
    });
  }

  async function saveDimensionCorrection() {
    if (!project || !selectedBenchmarkTarget || !correctionDraft) return;
    if (!correctionDraft.dimensionLineId) {
      setError("尺寸线尚未绑定。");
      return;
    }
    setBenchmarkBusy(true);
    setError("");
    try {
      const response = await api<DimensionBenchmarkResponse>(
        `/api/projects/${project.project_id}/benchmark/dimensions/correction`,
        {
          method: "PUT",
          body: JSON.stringify({
            ground_truth_id: selectedBenchmarkTarget.ground_truth.id,
            dimension_id: correctionDraft.dimensionId,
            text_id: correctionDraft.textId,
            dimension_line_id: correctionDraft.dimensionLineId,
            arrow_entity_ids: correctionDraft.arrowEntityIds,
            extension_line_ids: correctionDraft.extensionLineIds,
            measured_geometry_ids: correctionDraft.measuredGeometryIds,
          }),
        },
      );
      loadProject(response.project);
      setDimensionBenchmark(response.report);
      const updated = response.report.targets.find(
        (target) => target.ground_truth.id === selectedBenchmarkTarget.ground_truth.id,
      );
      if (updated?.matched_dimension_id) setSelectedMechanicalDimensionId(updated.matched_dimension_id);
      setLayoutNotice(
        `尺寸语义：${response.report.complete_count}/${response.report.target_count} 完整，`
          + `得分 ${Math.round(response.report.overall_score * 100)}%。`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBenchmarkBusy(false);
    }
  }

  const isVector = project?.source_kind === "vector_pdf";
  const sourceUrl = project?.source_image ? `${API_BASE}${project.source_image}` : "";
  const sourceFileUrl = project?.source_file ? `${API_BASE}${project.source_file}` : "";
  const referenceDisplayUrl = isVector && sourceFileUrl ? sourceFileUrl : sourceUrl;
  const analysisOverlayUrl = analysis?.overlay_image ? `${API_BASE}${analysis.overlay_image}?v=${cacheKey}` : "";
  const rasterReferenceUrl = analysisOverlayUrl || referenceDisplayUrl;
  const sourceIsPdf = referenceDisplayUrl.toLowerCase().includes(".pdf");
  const sourceKindLabel: Record<SourceKind, string> = {
    vector_pdf: "矢量 PDF · 直提取",
    scanned_pdf: "扫描 PDF · CV 生成",
    image: "图片 · CV 生成",
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Vibe CAD MVP</h1>
          <p>Claude Code-style loop for 2D CAD: IR edits, DXF export, visual preview.</p>
        </div>
        <div className="top-actions">
          <button className="icon-button" onClick={() => createProject()} disabled={busy} title="New baseline">
            <Plus size={18} />
            New
          </button>
          <a className="icon-button" href={dxfUrl} download="output.dxf" title="Download DXF">
            <Download size={18} />
            DXF
          </a>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="workspace">
        <section className="panel reference-panel">
          <div className="panel-title split">
            <span>
              <FileUp size={17} />
              Reference PDF/Image
              {project?.source_kind && (
                <small className={`source-badge ${project.source_kind}`}>{sourceKindLabel[project.source_kind]}</small>
              )}
            </span>
            <span className="panel-tools">
              <select
                className="mini-select"
                value={ocrLanguage}
                disabled={!project?.source_image || ocring || pipelineRunning}
                onChange={(event) => setOcrLanguage(event.target.value as OcrLanguage)}
                title="OCR language"
              >
                <option value="auto">Auto</option>
                <option value="zh">中文</option>
                <option value="en">EN</option>
              </select>
              <button
                className="mini-button run-pipeline"
                disabled={!project?.source_image || pipelineRunning}
                onClick={() => runCadPipeline()}
                title="Run the full CAD pipeline"
              >
                {pipelineRunning ? <RefreshCcw size={15} className="spin" /> : <Wand2 size={15} />}
                Generate CAD
              </button>
            </span>
          </div>
          <label className="upload-box">
            <input
              type="file"
              accept=".pdf,image/*"
              onChange={(event) => uploadFile(event.target.files?.[0] ?? null)}
            />
            <span>Upload a drawing PDF or image</span>
          </label>
          <div className="reference-stage">
            {referenceDisplayUrl ? (
              sourceIsPdf ? (
                <iframe title="Reference PDF" src={referenceDisplayUrl} />
              ) : (
                <div
                  className="reference-canvas"
                  style={
                    analysis?.image_width && analysis.image_height
                      ? { aspectRatio: `${analysis.image_width} / ${analysis.image_height}` }
                      : undefined
                  }
                >
                  <img alt="Reference drawing" src={rasterReferenceUrl} />
                  {!analysisOverlayUrl && analysis?.boxes.map((box, idx) => (
                    <div
                      className={`region-box ${box.target}`}
                      key={`${box.target}-${idx}`}
                      style={{
                        left: `${box.x * 100}%`,
                        top: `${box.y * 100}%`,
                        width: `${box.width * 100}%`,
                        height: `${box.height * 100}%`,
                      }}
                      title={`${box.label} ${Math.round(box.confidence * 100)}%`}
                    >
                      <span>{box.label || targetLabel(box.target)}</span>
                    </div>
                  ))}
                </div>
              )
            ) : (
              <div className="empty-state">PDF/image upload is optional in V0. Use chat to create or edit the CAD IR.</div>
            )}
          </div>
          {layoutNotice && <div className="layout-notice">{layoutNotice}</div>}
        </section>

        <section className="panel preview-panel">
          <div className="panel-title">
            <Layers size={17} />
            DXF Preview
          </div>
          {project && (
            <div className="layer-toolbar">
              <div className="layer-actions">
                <button className="mini-button" type="button" onClick={showAllPreviewLayers}>
                  All
                </button>
                <button
                  className="mini-button"
                  type="button"
                  onClick={showReferencePreviewLayer}
                  disabled={!hasReferencePreviewLayer}
                >
                  Reference
                </button>
                <button
                  className="mini-button"
                  type="button"
                  onClick={showEditablePreviewLayers}
                  disabled={!hasEditablePreviewLayers}
                >
                  Editable
                </button>
              </div>
              <div className="layer-toggles" aria-label="Preview layers">
                {previewLayers.map((layer) => {
                  const checked = !hiddenPreviewLayerSet.has(layer.name);
                  return (
                    <label className={`layer-toggle ${checked ? "" : "off"}`} key={layer.name} title={layer.name}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(event) => setPreviewLayerVisible(layer.name, event.target.checked)}
                      />
                      <span className={`layer-swatch ${layer.locked ? "reference" : "cad-black"}`} />
                      <span className="layer-name">{layerLabel(layer.name)}</span>
                      {layer.locked ? <Lock className="layer-lock" size={11} aria-label="Locked" /> : null}
                      <small>{layer.count}</small>
                    </label>
                  );
                })}
              </div>
              <span className="layer-count">
                {visiblePreviewEntityCount}/{project.ir.entities.length}
              </span>
            </div>
          )}
          <div className="preview-stage">
            {project ? (
              visiblePreviewEntityCount > 0 ? (
                <CadPreview
                  ir={project.ir}
                  hiddenLayers={hiddenPreviewLayerSet}
                  highlightedEntityIds={highlightedEntityIds}
                  onEntityClick={selectedBenchmarkTarget && correctionDraft ? assignCorrectionEntity : undefined}
                />
              ) : (
                <div className="empty-state">All preview layers are hidden.</div>
              )
            ) : (
              <div className="empty-state">Creating project...</div>
            )}
          </div>
          <div className="structure-eval">
            <div className="eval-summary">
              <span>Structure Eval</span>
              <strong className={evalReport?.passed ? "pass" : "fail"}>
                {evalReport ? `${Math.round(evalReport.overall_score * 100)}%` : "--"}
              </strong>
            </div>
            <div className="eval-targets">
              {evalReport?.targets.map((target) => (
                <div className={`eval-target ${target.passed ? "pass" : "fail"}`} key={target.name} title={target.missing.join(", ") || "passed"}>
                  {target.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
                  <span>{targetLabel(target.name)}</span>
                  <small>{Math.round(target.score * 100)}%</small>
                </div>
              ))}
            </div>
          </div>
          <div className="entity-list">
            {visibleEntities.map((entity) => (
              <div className="entity-row" key={entity.id}>
                <span>{entitySummary(entity)}</span>
                <small>{entityLayerName(entity)}</small>
              </div>
            ))}
            {hiddenEntityCount > 0 && (
              <div className="entity-row muted">
                <span>Hidden {hiddenEntityCount} more entities in the list preview</span>
                <small>{project?.ir.entities.length} total</small>
              </div>
            )}
          </div>
        </section>

        <aside className="panel agent-panel">
          <div className="panel-title">
            <Bot size={17} />
            CAD Agent
          </div>
          <div className="chat-log">
            {chat.map((item, idx) => (
              <div className={`chat-message ${item.role}`} key={`${item.role}-${idx}`}>
                <strong>{item.role === "user" ? "You" : "Agent"}</strong>
                <span>{item.text}</span>
              </div>
            ))}
          </div>
          <form className="chat-form" onSubmit={sendMessage}>
            <textarea
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Try: 添加孔 50 30 8"
              rows={3}
            />
            <button className="primary-button" disabled={busy || !project}>
              {busy ? <RefreshCcw size={17} className="spin" /> : <Play size={17} />}
              Run
            </button>
          </form>

          <div className="tabs">
            <button className={tab === "ir" ? "active" : ""} onClick={() => setTab("ir")}>
              <MessageSquare size={15} />
              IR
            </button>
            <button className={tab === "diff" ? "active" : ""} onClick={() => setTab("diff")}>
              <Layers size={15} />
              Diff
            </button>
            <button className={tab === "ocr" ? "active" : ""} onClick={() => setTab("ocr")}>
              <ScanLine size={15} />
              OCR
            </button>
            <button className={tab === "cells" ? "active" : ""} onClick={() => setTab("cells")}>
              <Layers size={15} />
              Cells
            </button>
            <button className={tab === "dims" ? "active" : ""} onClick={() => setTab("dims")}>
              <Ruler size={15} />
              Semantics
            </button>
            <button className={tab === "history" ? "active" : ""} onClick={() => setTab("history")}>
              <History size={15} />
              History
            </button>
          </div>

          <div className="inspector">
            {tab === "ir" && <pre>{project ? inspectorText(compactIrForInspector(project.ir)) : ""}</pre>}
            {tab === "diff" && (
              <div className="diff-list">
                {project?.diffs.length ? (
                  project.diffs.map((diff) => (
                    <div className="diff-row" key={diff.path}>
                      <strong>{diff.path}</strong>
                      <code>{inspectorText(diff.before)}</code>
                      <span>to</span>
                      <code>{inspectorText(diff.after)}</code>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">No diff yet.</div>
                )}
              </div>
            )}
            {tab === "ocr" && (
              <div className="ocr-list">
                {project?.ocr_regions?.length ? (
                  project.ocr_regions.map((region, idx) => (
                    <div className="ocr-row" key={`${region.target}-${idx}`}>
                      <div className="ocr-row-title">
                        <strong>{targetLabel(region.target)}</strong>
                        <small>{Math.round(region.confidence * 100)}% · {region.language || region.engine}</small>
                      </div>
                      <p>{region.text || "No text recognized"}</p>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">No OCR result yet.</div>
                )}
              </div>
            )}
            {tab === "cells" && (
              <div className="ocr-list">
                {(project?.title_block_cells?.length || project?.table_ocr_cells?.length) ? (
                  <>
                    {project?.title_block_cells?.map((cell, idx) => (
                      <div className="ocr-row" key={`title-block-${cell.row}-${cell.col}-${idx}`}>
                        <div className="ocr-row-title">
                          <strong>{targetLabel("title_block")} · r{cell.row + 1} c{cell.col + 1}</strong>
                          <small>{Math.round(cell.confidence * 100)}% · {cell.provider}</small>
                        </div>
                        <p>{cell.text || "No text recognized"}</p>
                      </div>
                    ))}
                    {project?.table_ocr_cells?.filter((cell) => cell.target !== "title_block").map((cell, idx) => (
                      <div className="ocr-row" key={`${cell.target}-${cell.row}-${cell.col}-${idx}`}>
                        <div className="ocr-row-title">
                          <strong>{targetLabel(cell.target)} · r{cell.row + 1} c{cell.col + 1}</strong>
                          <small>{Math.round(cell.confidence * 100)}% · {cell.language || cell.engine}</small>
                        </div>
                        <p>{cell.text || "No text recognized"}</p>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className="empty-state">No table OCR cells yet.</div>
                )}
              </div>
            )}
            {tab === "dims" && (
              <div className="semantic-inspector">
                <div className="dimension-benchmark-header">
                  <div>
                    <strong>Dimension benchmark</strong>
                    <small>
                      {dimensionBenchmark?.target_count
                        ? `${dimensionBenchmark.complete_count}/${dimensionBenchmark.target_count} complete · ${dimensionBenchmark.matched_count} matched`
                        : "Ground truth not initialized"}
                    </small>
                  </div>
                  {dimensionBenchmark?.target_count ? (
                    <div className="dimension-benchmark-actions">
                      <strong className={dimensionBenchmark.complete_count === dimensionBenchmark.target_count ? "pass" : "fail"}>
                        {Math.round(dimensionBenchmark.overall_score * 100)}%
                      </strong>
                      <button
                        className="semantic-repair-button"
                        type="button"
                        disabled={benchmarkBusy || !hasPendingLinearDimension}
                        onClick={runAutomaticDimensionRepair}
                        title={hasPendingLinearDimension
                          ? "DeepSeek 规划顺序，本地几何工具执行并逐步评估"
                          : "本阶段可自动修复的线性尺寸已经完成"}
                      >
                        {benchmarkBusy ? <RefreshCcw size={15} className="spin" /> : <Wand2 size={15} />}
                        {hasPendingLinearDimension ? "一键自动修复" : "线性尺寸已完成"}
                      </button>
                    </div>
                  ) : (
                    <button className="mini-button" type="button" disabled={benchmarkBusy} onClick={seedDimensionBenchmark}>
                      <Plus size={14} />
                      建立基准
                    </button>
                  )}
                </div>
                {dimensionBenchmark?.targets.length ? (
                  <div className="benchmark-target-list">
                    {dimensionBenchmark.targets.map((target) => (
                      <button
                        className={`benchmark-target ${target.passed ? "passed" : "incomplete"} ${selectedGroundTruthId === target.ground_truth.id ? "active" : ""}`}
                        key={target.ground_truth.id}
                        type="button"
                        onClick={() => setSelectedGroundTruthId(target.ground_truth.id)}
                        title={target.missing_relations.join(", ") || "complete"}
                      >
                        {target.passed ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                        <span>
                          <strong>{target.ground_truth.label}</strong>
                          <small>{target.missing_relations.join(" · ") || "DXF ready"}</small>
                        </span>
                        <b>{Math.round(target.score * 100)}%</b>
                      </button>
                    ))}
                  </div>
                ) : null}
                {latestSemanticRepairRun ? (
                  <div className="semantic-repair-trace">
                    <div className="semantic-repair-summary">
                      <span>
                        <strong>Agent repair trace</strong>
                        <small>
                          {latestSemanticRepairRun.planner_source}
                          {latestSemanticRepairRun.planner_model ? ` · ${latestSemanticRepairRun.planner_model}` : ""}
                        </small>
                      </span>
                      <b>
                        {Math.round(latestSemanticRepairRun.before_score * 100)}%
                        <span> to </span>
                        {Math.round(latestSemanticRepairRun.after_score * 100)}%
                      </b>
                    </div>
                    <div className="semantic-repair-steps">
                      {latestSemanticRepairRun.steps.map((step) => (
                        <button
                          className={`semantic-repair-step ${step.status}`}
                          key={`${latestSemanticRepairRun.id}-${step.index}`}
                          type="button"
                          onClick={() => setSelectedGroundTruthId(step.ground_truth_id)}
                          title={step.detail}
                        >
                          {step.status === "accepted" ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
                          <span>{step.label}</span>
                          <small>{Math.round(step.score_before * 100)}% to {Math.round(step.score_after * 100)}%</small>
                        </button>
                      ))}
                    </div>
                    {!latestSemanticRepairRun.rolled_back_at && latestSemanticRepairRun.snapshot_file ? (
                      <button
                        className="semantic-repair-rollback"
                        type="button"
                        disabled={benchmarkBusy}
                        onClick={rollbackAutomaticDimensionRepair}
                      >
                        <History size={13} />
                        回滚本次修复
                      </button>
                    ) : (
                      <small className="semantic-repair-rolled-back">本次修复已回滚</small>
                    )}
                  </div>
                ) : null}
                {selectedBenchmarkTarget && correctionDraft ? (
                  <div className="correction-workbench">
                    <div className="inspector-section-title">
                      <span>{selectedBenchmarkTarget.ground_truth.expected_text} · semantic correction</span>
                      <small>{selectedBenchmarkTarget.corrected ? "manual" : "detected"}</small>
                    </div>
                    <div className="correction-role-control" role="tablist" aria-label="Semantic relation role">
                      {(Object.keys(DIMENSION_ROLE_LABELS) as DimensionCorrectionRole[]).map((role) => (
                        <button
                          className={correctionRole === role ? "active" : ""}
                          key={role}
                          type="button"
                          onClick={() => setCorrectionRole(role)}
                        >
                          {DIMENSION_ROLE_LABELS[role]}
                          <small>{correctionIdsForRole(correctionDraft, role).length}</small>
                        </button>
                      ))}
                    </div>
                    <div className="correction-selection">
                      <MousePointer2 size={14} />
                      <code>{correctionIdsForRole(correctionDraft, correctionRole).join("\n") || "--"}</code>
                      <button className="mini-button" type="button" onClick={clearCorrectionRole}>
                        清空
                      </button>
                    </div>
                    <button
                      className="correction-save"
                      type="button"
                      disabled={benchmarkBusy || !correctionDraft.dimensionLineId}
                      onClick={saveDimensionCorrection}
                    >
                      {benchmarkBusy ? <RefreshCcw size={15} className="spin" /> : <Save size={15} />}
                      保存尺寸对象
                    </button>
                  </div>
                ) : null}
                {mechanicalDimensions.length ? (
                  <>
                    <div className="inspector-section-title">
                      <span>MechanicalDrawingIR v{project?.mechanical_ir?.schema_version ?? "legacy"}</span>
                      <small>
                        {mechanicalDimensions.filter((dimension) => dimension.status === "complete").length}/
                        {mechanicalDimensions.length} complete
                      </small>
                    </div>
                    <div className="semantic-object-list">
                      {mechanicalDimensions.map((dimension) => (
                        <button
                          className={`semantic-row ${dimension.id === selectedMechanicalDimension?.id ? "active" : ""}`}
                          key={dimension.id}
                          type="button"
                          onClick={() => setSelectedMechanicalDimensionId(dimension.id)}
                        >
                          <div className="ocr-row-title">
                            <strong>{mechanicalDimensionSummary(dimension)}</strong>
                            <small>{Math.round(dimension.confidence * 100)}%</small>
                          </div>
                          <p>{dimension.text || "No bound text"}</p>
                          <span className={`semantic-status ${dimension.status ?? "partial"}`}>
                            {dimension.status ?? "partial"}
                          </span>
                        </button>
                      ))}
                    </div>
                    {selectedMechanicalDimension ? (
                      <>
                        <div className="inspector-section-title">
                          <span>Bound relations</span>
                          <small>
                            {selectedMechanicalDimension.orientation} · {selectedMechanicalDimension.export_ready ? "DXF ready" : "fallback"}
                          </small>
                        </div>
                        <div className="semantic-relations">
                          <SemanticRelationRow
                            label="Dimension text"
                            ids={selectedMechanicalDimension.text_id ? [selectedMechanicalDimension.text_id] : []}
                            dimension={selectedMechanicalDimension}
                          />
                          <SemanticRelationRow
                            label="Dimension line"
                            ids={[selectedMechanicalDimension.dimension_line_id]}
                            dimension={selectedMechanicalDimension}
                          />
                          <SemanticRelationRow
                            label="Arrowheads"
                            ids={selectedMechanicalDimension.arrowheads.map((arrow) => arrow.render_entity_id)}
                            dimension={selectedMechanicalDimension}
                          />
                          <SemanticRelationRow
                            label="Extension lines"
                            ids={selectedMechanicalDimension.extension_line_ids ?? []}
                            dimension={selectedMechanicalDimension}
                          />
                          <SemanticRelationRow
                            label="Measured geometry"
                            ids={selectedMechanicalDimension.measured_geometry_ids ?? selectedMechanicalDimension.target_geometry_ids}
                            dimension={selectedMechanicalDimension}
                          />
                        </div>
                        <div className={`semantic-native ${selectedMechanicalDimension.export_ready ? "ready" : "fallback"}`}>
                          <div className="ocr-row-title">
                            <strong>Native dimension</strong>
                            <span>{selectedMechanicalDimension.dxf_dimension_type ?? "structured layer"}</span>
                          </div>
                          <code>
                            {(selectedMechanicalDimension.measurement_points ?? [])
                              .map((point) => `(${point.slice(0, 2).map((value) => Number(value).toFixed(2)).join(", ")})`)
                              .join(" -> ") || "definition points missing"}
                          </code>
                          <small>
                            {selectedMechanicalDimension.edit_mode ?? "annotation_override"}
                            {selectedMechanicalDimension.measured_value !== null && selectedMechanicalDimension.measured_value !== undefined
                              ? ` · measured ${selectedMechanicalDimension.measured_value} ${selectedMechanicalDimension.parsed.unit}`
                              : ""}
                            {selectedMechanicalDimension.last_edit_source
                              ? ` · ${selectedMechanicalDimension.last_edit_source === "deepseek" ? "DeepSeek V4 Flash" : "local parser"}`
                              : ""}
                            {selectedMechanicalDimension.validation_status
                              ? ` · ${selectedMechanicalDimension.validation_status}`
                              : ""}
                          </small>
                        </div>
                        {selectedMechanicalDimension.issues?.length ? (
                          <div className="semantic-issues">
                            {selectedMechanicalDimension.issues.map((issue) => <span key={issue}>{issue}</span>)}
                          </div>
                        ) : null}
                        {selectedDimensionBinding ? (
                          <div className="semantic-evidence">
                            <div className="ocr-row-title">
                              <strong>Graph evidence</strong>
                              <small>{selectedDimensionBinding.binding_method}</small>
                            </div>
                            <code>{graphPathSummary(selectedDimensionBinding)}</code>
                            <small>score {selectedDimensionBinding.graph_score?.toFixed(3) ?? "--"}</small>
                          </div>
                        ) : null}
                      </>
                    ) : null}
                  </>
                ) : project?.dimension_bindings?.length ? (
                  project.dimension_bindings.map((binding) => (
                    <div className="ocr-row" key={binding.id}>
                      <div className="ocr-row-title">
                        <strong>{dimensionValueSummary(binding)}</strong>
                        <small>{Math.round(binding.confidence * 100)}%</small>
                      </div>
                      <p>{binding.text || "No bound text"}</p>
                      <small>
                        line {binding.dimension_line_id} · arrows {binding.arrow_ids.join(", ") || "--"}
                      </small>
                      <small>
                        {binding.binding_method ?? "unknown"} · score {binding.graph_score?.toFixed(3) ?? "--"}
                      </small>
                      <code>{graphPathSummary(binding)}</code>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">No mechanical semantic objects yet.</div>
                )}
              </div>
            )}
            {tab === "history" && (
              <div className="history-list">
                <div className="agent-eval-panel">
                  <div className="agent-eval-header">
                    <span>
                      <strong>Agent Eval</strong>
                      <small>{latestAgentEvalReport?.dataset_version ?? "agent-tasks-v1.1 · 12 cases"}</small>
                    </span>
                    {latestAgentEvalReport ? (
                      <b className={latestAgentEvalReport.passed_count === latestAgentEvalReport.case_count ? "pass" : "fail"}>
                        {latestAgentEvalReport.passed_count}/{latestAgentEvalReport.case_count}
                      </b>
                    ) : null}
                  </div>
                  <div className="agent-eval-controls">
                    <select
                      value={agentEvalMode}
                      disabled={agentEvalBusy}
                      onChange={(event) => setAgentEvalMode(event.target.value as "deterministic" | "deepseek")}
                    >
                      <option value="deterministic">Deterministic baseline</option>
                      <option value="deepseek">DeepSeek V4 Flash</option>
                    </select>
                    <button type="button" disabled={agentEvalBusy} onClick={runAgentEval}>
                      {agentEvalBusy ? <RefreshCcw size={14} className="spin" /> : <ScanLine size={14} />}
                      Run Eval
                    </button>
                  </div>
                  {latestAgentEvalReport ? (
                    <>
                      <div className="agent-eval-metrics">
                        <span><small>Task success</small><strong>{Math.round(latestAgentEvalReport.metrics.task_success_rate * 100)}%</strong></span>
                        <span><small>Tool precision</small><strong>{Math.round(latestAgentEvalReport.metrics.tool_selection_precision * 100)}%</strong></span>
                        <span><small>Tool recall</small><strong>{Math.round(latestAgentEvalReport.metrics.tool_selection_recall * 100)}%</strong></span>
                        <span><small>Tool order</small><strong>{Math.round(latestAgentEvalReport.metrics.tool_order_accuracy * 100)}%</strong></span>
                        <span><small>Arguments</small><strong>{Math.round(latestAgentEvalReport.metrics.argument_accuracy * 100)}%</strong></span>
                        <span><small>Safety</small><strong>{Math.round(latestAgentEvalReport.metrics.safety_pass_rate * 100)}%</strong></span>
                        <span><small>Invalid actions</small><strong>{Math.round(latestAgentEvalReport.metrics.invalid_action_rate * 100)}%</strong></span>
                        <span><small>Avg calls</small><strong>{latestAgentEvalReport.metrics.average_tool_calls.toFixed(1)}</strong></span>
                        <span><small>Policy checks</small><strong>{(latestAgentEvalReport.metrics.average_policy_injected_steps ?? 0).toFixed(1)}</strong></span>
                      </div>
                      <div className="agent-eval-cases">
                        {latestAgentEvalReport.cases.map((evalCase) => (
                          <div className={evalCase.passed ? "passed" : "failed"} key={evalCase.case_id}>
                            {evalCase.passed ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
                            <span>
                              <strong>{evalCase.case_id}</strong>
                              <small>
                                {evalCase.actual_tools.join(" -> ")}
                                {evalCase.failed_assertions.length ? ` · ${evalCase.failed_assertions.join("; ")}` : ""}
                              </small>
                            </span>
                            <b>{Math.round(evalCase.score * 100)}%</b>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : null}
                </div>
                {latestAgentTaskRun ? (
                  <div className="agent-task-timeline">
                    <div className="agent-task-run-header">
                      <span>
                        <strong>Task Agent</strong>
                        <small>
                          {latestAgentTaskRun.planner_source}
                          {latestAgentTaskRun.planner_model ? ` · ${latestAgentTaskRun.planner_model}` : ""}
                          {` · ${latestAgentTaskRun.llm_calls} LLM calls`}
                          {latestAgentTaskRun.policy_injected_steps
                            ? ` · ${latestAgentTaskRun.policy_injected_steps} policy checks`
                            : ""}
                        </small>
                      </span>
                      <b className={latestAgentTaskRun.status}>{latestAgentTaskRun.status}</b>
                    </div>
                    <p className="agent-task-goal">{latestAgentTaskRun.goal}</p>
                    <div className="agent-task-steps">
                      {latestAgentTaskRun.steps.map((step) => (
                        <div className={`agent-task-step ${step.status}`} key={`${latestAgentTaskRun.id}-${step.index}`}>
                          <span className="agent-task-step-icon">
                            {step.status === "accepted" || step.status === "skipped"
                              ? <CheckCircle2 size={14} />
                              : <XCircle size={14} />}
                          </span>
                          <span>
                            <strong>{step.index + 1}. {step.tool}</strong>
                            <small>{step.observation}</small>
                          </span>
                          <code>
                            {step.dimension_score_before != null && step.dimension_score_after != null
                              ? `${Math.round(step.dimension_score_before * 100)}% to ${Math.round(step.dimension_score_after * 100)}%`
                              : step.status}
                          </code>
                        </div>
                      ))}
                    </div>
                    <div className="agent-task-run-footer">
                      <span>{latestAgentTaskRun.summary}</span>
                      {!latestAgentTaskRun.rolled_back_at && latestAgentTaskRun.snapshot_file ? (
                        <button type="button" disabled={busy} onClick={rollbackLatestAgentTask}>
                          <History size={13} />
                          回滚任务
                        </button>
                      ) : null}
                    </div>
                  </div>
                ) : null}
                {project?.history.map((op, idx) => (
                  <div className="history-row" key={`${op.operation}-${idx}`}>
                    <strong>{idx + 1}. {op.operation}</strong>
                    <span>{op.reason}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);

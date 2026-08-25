from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal

from .agent import plan_operations
from .cad_layers import REFERENCE_TRACE, canonical_layer_name
from .cad_ops import apply_operations
from .dimension_benchmark import evaluate_dimension_benchmark
from .mechanical_drive import (
    MechanicalDriveError,
    execute_mechanical_operation,
    plan_mechanical_drive_deterministic,
)
from .models import (
    AgentToolArguments,
    AgentToolDefinition,
    AgentToolName,
    MechanicalOperation,
    ProjectState,
)
from .semantic_repair import append_semantic_repair_run, run_semantic_repair_agent


class AgentToolError(ValueError):
    """Raised when a registered tool cannot safely satisfy its contract."""


@dataclass(frozen=True)
class AgentToolContext:
    goal: str
    planner_source: str


@dataclass(frozen=True)
class AgentToolOutcome:
    project: ProjectState
    status: Literal["accepted", "skipped"]
    observation: str
    output: dict
    validation: dict


ToolHandler = Callable[[ProjectState, AgentToolArguments, AgentToolContext], AgentToolOutcome]


@dataclass(frozen=True)
class RegisteredTool:
    definition: AgentToolDefinition
    handler: ToolHandler


class AgentToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[AgentToolName, RegisteredTool] = {}

    def register(self, definition: AgentToolDefinition, handler: ToolHandler) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Duplicate agent tool: {definition.name}")
        self._tools[definition.name] = RegisteredTool(definition, handler)

    def definitions(self) -> list[AgentToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def definition(self, name: AgentToolName) -> AgentToolDefinition:
        try:
            return self._tools[name].definition
        except KeyError as exc:  # pragma: no cover - guarded by Pydantic Literal
            raise AgentToolError(f"Unknown agent tool: {name}") from exc

    def execute(
        self,
        name: AgentToolName,
        project: ProjectState,
        arguments: AgentToolArguments,
        context: AgentToolContext,
    ) -> AgentToolOutcome:
        try:
            tool = self._tools[name]
        except KeyError as exc:  # pragma: no cover - guarded by Pydantic Literal
            raise AgentToolError(f"Unknown agent tool: {name}") from exc
        return tool.handler(project, arguments, context)


POST_MUTATION_EVALUATORS: dict[AgentToolName, AgentToolName] = {
    "repair_dimensions": "evaluate_dimensions",
    "drive_dimension": "evaluate_dimensions",
    "edit_cad": "evaluate_drawing",
}


def post_mutation_evaluator(name: AgentToolName) -> AgentToolName | None:
    return POST_MUTATION_EVALUATORS.get(name)


def build_default_tool_registry() -> AgentToolRegistry:
    registry = AgentToolRegistry()
    registry.register(
        _definition(
            "inspect_drawing",
            "Inspect source type, layers, entities, semantic dimensions, and editable state.",
            {},
            validator="read_only",
        ),
        _inspect_drawing,
    )
    registry.register(
        _definition(
            "evaluate_drawing",
            "Validate generic DrawingIR integrity after entity edits.",
            {},
            validator="drawing_integrity",
        ),
        _evaluate_drawing,
    )
    registry.register(
        _definition(
            "evaluate_dimensions",
            "Evaluate dimension text, lines, arrows, extension lines, measured geometry, and native readiness.",
            {},
            validator="dimension_benchmark",
        ),
        _evaluate_dimensions,
    )
    registry.register(
        _definition(
            "repair_dimensions",
            "Repair locally supported incomplete linear dimensions with monotonic evaluation.",
            {
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "min_gain": {"type": "number", "minimum": 0, "maximum": 0.5, "default": 0.01},
            },
            mutating=True,
            reversible=True,
            validator="monotonic_dimension_benchmark",
        ),
        _repair_dimensions,
    )
    registry.register(
        _definition(
            "drive_dimension",
            "Drive one complete semantic dimension to an absolute target value.",
            {
                "dimension_id": {"type": "string"},
                "target_value": {"type": "number", "exclusiveMinimum": 0},
                "anchor": {"type": "string", "enum": ["start", "end"], "default": "start"},
            },
            mutating=True,
            reversible=True,
            validator="mechanical_geometry_measurement",
        ),
        _drive_dimension,
    )
    registry.register(
        _definition(
            "edit_cad",
            "Apply a simple deterministic entity edit such as add, move, resize, delete, or set layer.",
            {"message": {"type": "string"}},
            mutating=True,
            reversible=True,
            validator="operation_applied_without_locked_reference_mutation",
        ),
        _edit_cad,
    )
    registry.register(
        _definition(
            "export_dxf",
            "Request regeneration of the project DXF and SVG artifacts after accepted edits.",
            {},
            validator="export_on_project_commit",
        ),
        _export_dxf,
    )
    return registry


def _definition(
    name: AgentToolName,
    description: str,
    properties: dict,
    *,
    mutating: bool = False,
    reversible: bool = False,
    validator: str,
) -> AgentToolDefinition:
    return AgentToolDefinition(
        name=name,
        description=description,
        parameters={"type": "object", "properties": properties, "additionalProperties": False},
        mutating=mutating,
        reversible=reversible,
        validator=validator,
    )


def _inspect_drawing(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    del arguments, context
    layers = Counter(entity.layer for entity in project.ir.entities)
    dimensions = project.mechanical_ir.dimensions
    complete = sum(dimension.status == "complete" for dimension in dimensions)
    output = {
        "project_id": project.project_id,
        "source_kind": project.source_kind,
        "entity_count": len(project.ir.entities),
        "layer_counts": dict(layers),
        "dimension_count": len(dimensions),
        "complete_dimensions": complete,
        "ground_truth_count": len(project.dimension_ground_truth),
        "locked_layers": [layer.name for layer in project.ir.layers if layer.locked],
    }
    return AgentToolOutcome(
        project=project,
        status="accepted",
        observation=(
            f"图纸包含 {output['entity_count']} 个图元、{len(dimensions)} 个机械尺寸，"
            f"其中 {complete} 个完整。"
        ),
        output=output,
        validation={"passed": True, "validator": "read_only"},
    )


def _evaluate_dimensions(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    del arguments, context
    if not project.dimension_ground_truth:
        return AgentToolOutcome(
            project=project,
            status="skipped",
            observation="尺寸评估基准尚未初始化。",
            output={"target_count": 0},
            validation={"passed": True, "validator": "dimension_benchmark", "skipped": True},
        )
    report = evaluate_dimension_benchmark(project)
    return AgentToolOutcome(
        project=project,
        status="accepted",
        observation=(
            f"尺寸语义得分 {report.overall_score:.1%}，"
            f"{report.complete_count}/{report.target_count} 个目标完整。"
        ),
        output={
            "score": report.overall_score,
            "complete_count": report.complete_count,
            "matched_count": report.matched_count,
            "target_count": report.target_count,
            "metrics": report.metrics,
        },
        validation={"passed": True, "validator": "dimension_benchmark"},
    )


def _evaluate_drawing(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    del arguments, context
    entity_ids = [entity.id for entity in project.ir.entities]
    duplicate_ids = sorted(entity_id for entity_id, count in Counter(entity_ids).items() if count > 1)
    nonfinite_values: list[str] = []
    for entity in project.ir.entities:
        _collect_nonfinite_values(entity.model_dump(mode="python"), entity.id, nonfinite_values)
    if duplicate_ids or nonfinite_values:
        details = []
        if duplicate_ids:
            details.append(f"duplicate entity ids: {', '.join(duplicate_ids[:5])}")
        if nonfinite_values:
            details.append(f"non-finite geometry: {', '.join(nonfinite_values[:5])}")
        raise AgentToolError("Drawing integrity validation failed: " + "; ".join(details))

    declared_layers = {layer.name for layer in project.ir.layers}
    undeclared_layers = sorted({entity.layer for entity in project.ir.entities} - declared_layers)
    return AgentToolOutcome(
        project=project,
        status="accepted",
        observation=f"图纸完整性检查通过：{len(entity_ids)} 个图元，ID 与几何数值有效。",
        output={
            "entity_count": len(entity_ids),
            "unique_entity_count": len(set(entity_ids)),
            "undeclared_layers": undeclared_layers,
        },
        validation={
            "passed": True,
            "validator": "drawing_integrity",
            "duplicate_ids": duplicate_ids,
            "nonfinite_values": nonfinite_values,
        },
    )


def _collect_nonfinite_values(value, path: str, output: list[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            output.append(path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_nonfinite_values(item, f"{path}.{key}", output)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_nonfinite_values(item, f"{path}[{index}]", output)


def _repair_dimensions(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    del context
    if not project.dimension_ground_truth:
        raise AgentToolError("Dimension ground truth is not initialized")
    from .models import SemanticRepairRequest

    repaired, report, run = run_semantic_repair_agent(
        project,
        SemanticRepairRequest(
            use_llm=False,
            max_steps=arguments.max_steps,
            min_gain=arguments.min_gain,
        ),
    )
    append_semantic_repair_run(repaired, run)
    if run.after_score + 1e-9 < run.before_score:
        raise AgentToolError("Dimension repair violated the monotonic score guard")
    status: Literal["accepted", "skipped"] = "accepted" if run.accepted_steps else "skipped"
    return AgentToolOutcome(
        project=repaired,
        status=status,
        observation=(
            f"尺寸修复接受 {run.accepted_steps} 步、拒绝 {run.rejected_steps} 步，"
            f"得分 {run.before_score:.1%} -> {run.after_score:.1%}。"
        ),
        output={
            "repair_run_id": run.id,
            "accepted_steps": run.accepted_steps,
            "rejected_steps": run.rejected_steps,
            "stopped_reason": run.stopped_reason,
            "complete_count": report.complete_count,
            "target_count": report.target_count,
        },
        validation={
            "passed": True,
            "validator": "monotonic_dimension_benchmark",
            "before": run.before_score,
            "after": run.after_score,
        },
    )


def _drive_dimension(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    plan = None
    if arguments.dimension_id and arguments.target_value is not None:
        plan = MechanicalOperation(
            dimension_id=arguments.dimension_id,
            target_value=arguments.target_value,
            anchor=arguments.anchor,
            planner_source="deepseek" if context.planner_source == "deepseek" else "deterministic",
            reason="Task-level agent tool call.",
        )
    if plan is None:
        plan = plan_mechanical_drive_deterministic(arguments.message or context.goal, project)
    if plan is None:
        raise AgentToolError("drive_dimension requires a driveable dimension_id and absolute target_value")
    try:
        result = execute_mechanical_operation(project, plan, context.goal)
    except MechanicalDriveError as exc:
        raise AgentToolError(str(exc)) from exc
    _sync_ground_truth_after_drive(result.project, plan.dimension_id, plan.target_value)
    return AgentToolOutcome(
        project=result.project,
        status="accepted",
        observation=result.reply,
        output={
            "dimension_id": plan.dimension_id,
            "target_value": plan.target_value,
            "measured_value": result.validation.measured_value,
            "operation_count": len(result.operations),
            "diff_count": len(result.diffs),
        },
        validation={
            "passed": result.validation.passed,
            "validator": "mechanical_geometry_measurement",
            "checks": result.validation.checks,
            "errors": result.validation.errors,
            "measured_value": result.validation.measured_value,
            "target_value": result.validation.target_value,
        },
    )


def _edit_cad(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    message = (arguments.message or context.goal).strip()
    operations, reply = plan_operations(message, project.ir)
    if not operations:
        raise AgentToolError(reply)
    if project.source_kind == "vector_pdf" and any(
        operation.operation in {"create_plate", "create_spur_gear_drawing"}
        for operation in operations
    ):
        raise AgentToolError("Refusing to replace an imported vector PDF with a template drawing")
    _ensure_no_locked_reference_mutation(project, operations)
    working = project.model_copy(deep=True)
    try:
        working.ir, diffs = apply_operations(working.ir, operations)
    except ValueError as exc:
        raise AgentToolError(str(exc)) from exc
    working.history.extend(operations)
    working.diffs = diffs
    return AgentToolOutcome(
        project=working,
        status="accepted",
        observation=reply,
        output={"operation_count": len(operations), "diff_count": len(diffs)},
        validation={
            "passed": True,
            "validator": "operation_applied_without_locked_reference_mutation",
        },
    )


def _export_dxf(
    project: ProjectState,
    arguments: AgentToolArguments,
    context: AgentToolContext,
) -> AgentToolOutcome:
    del arguments, context
    return AgentToolOutcome(
        project=project,
        status="accepted",
        observation="DXF 和 SVG 将在任务提交时从当前 MechanicalDrawingIR 重新生成。",
        output={
            "dxf_url": f"/api/projects/{project.project_id}/files/output.dxf",
            "svg_url": f"/api/projects/{project.project_id}/files/preview.svg",
        },
        validation={"passed": True, "validator": "export_on_project_commit"},
    )


def _ensure_no_locked_reference_mutation(project: ProjectState, operations) -> None:
    entities = {entity.id: entity for entity in project.ir.entities}
    layers = {layer.name: layer for layer in project.ir.layers}
    for operation in operations:
        entity = entities.get(operation.entity_id or "")
        if entity is None:
            continue
        layer = layers.get(entity.layer)
        if canonical_layer_name(entity.layer) == REFERENCE_TRACE or (layer is not None and layer.locked):
            raise AgentToolError(f"Refusing to modify locked reference entity: {entity.id}")


def _sync_ground_truth_after_drive(project: ProjectState, dimension_id: str, target_value: float) -> None:
    dimension = next(
        (
            item
            for item in project.mechanical_ir.dimensions
            if item.id == dimension_id or item.binding_id == dimension_id
        ),
        None,
    )
    if dimension is None:
        return
    corrected_target_ids = {
        correction.ground_truth_id
        for correction in project.dimension_corrections
        if correction.dimension_id == dimension.id
    }
    for target in project.dimension_ground_truth:
        if target.matched_dimension_id not in {dimension.id, dimension.binding_id} and target.id not in corrected_target_ids:
            continue
        target.expected_text = dimension.text
        target.nominal = target_value
        target.source = "manual"

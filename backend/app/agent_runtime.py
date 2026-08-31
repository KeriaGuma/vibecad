from __future__ import annotations

import re
from datetime import datetime, timezone

from .agent import plan_operations
from .agent_tools import (
    AgentToolContext,
    AgentToolError,
    AgentToolRegistry,
    build_default_tool_registry,
    post_mutation_evaluator,
)
from .dimension_benchmark import evaluate_dimension_benchmark
from .llm_agent import LlmUnavailable, plan_agent_task_llm
from .mechanical_drive import plan_mechanical_drive_deterministic
from .mechanical_edit import (
    EDIT_INTENT_RE,
    _explicit_dimension_id,
    _find_dimension,
    _numbers_without_explicit_ids,
)
from .models import (
    AgentPlannedStep,
    AgentTaskClarification,
    AgentTaskRequest,
    AgentTaskRun,
    AgentTaskStepTrace,
    AgentToolArguments,
    ProjectState,
    new_id,
)

MAX_TASK_RUNS = 20
REPAIR_TERMS = ("修复", "补全", "补齐", "repair", "complete dimensions")
EXPORT_TERMS = ("导出", "下载", "export", "dxf")
EVALUATE_TERMS = ("评估", "检查", "验证", "evaluate", "inspect", "check", "validate")
NO_EDIT_TERMS = ("不修改", "不要修改", "禁止修改", "只读", "do not modify", "without modifying", "read only")


def run_agent_task(
    project: ProjectState,
    request: AgentTaskRequest,
    registry: AgentToolRegistry | None = None,
) -> tuple[ProjectState, AgentTaskRun]:
    """Plan and execute a bounded task with tool-specific validation and replanning."""

    registry = registry or build_default_tool_registry()
    working = project.model_copy(deep=True)
    before_score = _dimension_score(working)
    clarification = _focused_clarification(request.goal, working)
    if clarification is not None:
        return _clarification_run(working, request, registry, before_score, clarification)
    planner_source = "deterministic"
    planner_model = None
    planner_reason = "Local intent planner selected registered CAD tools."
    llm_calls = 0
    if request.use_llm:
        llm_calls += 1
        try:
            plan, planner_reason, planner_model = plan_agent_task_llm(
                request.goal,
                working,
                _catalog_payload(registry),
                request.max_tool_calls,
            )
        except LlmUnavailable as exc:
            planner_source = "deterministic_fallback"
            planner_reason = f"DeepSeek unavailable; deterministic plan used. {exc}"
            plan = _deterministic_plan(request.goal, working, request.max_tool_calls)
        else:
            planner_source = "deepseek"
    else:
        plan = _deterministic_plan(request.goal, working, request.max_tool_calls)

    plan, policy_injected_steps = _enforce_post_mutation_evaluators(
        plan,
        registry,
        request.max_tool_calls,
        goal=request.goal,
    )
    if policy_injected_steps:
        planner_reason = (
            f"{planner_reason} Runtime policy injected {policy_injected_steps} "
            "post-mutation evaluator(s)."
        )
    initial_plan = [step.model_copy(deep=True) for step in plan]
    queue = list(plan)
    traces: list[AgentTaskStepTrace] = []
    artifacts: dict[str, str] = {}
    failed_signatures: set[str] = set()
    replan_count = 0
    tool_calls = 0
    terminal_failure = not bool(queue)
    pending_mutation: tuple[ProjectState, int, str, str] | None = None

    while queue and tool_calls < request.max_tool_calls:
        planned = queue.pop(0)
        signature = _step_signature(planned)
        if signature in failed_signatures:
            now = datetime.now(timezone.utc)
            traces.append(
                AgentTaskStepTrace(
                    index=len(traces),
                    attempt=replan_count + 1,
                    call_id=planned.call_id,
                    tool=planned.tool,
                    status="skipped",
                    arguments=planned.arguments,
                    reason=planned.reason,
                    observation="跳过与已失败调用相同的工具参数。",
                    validation={"passed": True, "duplicate_failed_call": True},
                    mutating=registry.definition(planned.tool).mutating,
                    reversible=registry.definition(planned.tool).reversible,
                    started_at=now,
                    completed_at=now,
                )
            )
            continue

        tool_calls += 1
        definition = registry.definition(planned.tool)
        started = datetime.now(timezone.utc)
        score_before = _dimension_score(working)
        mutation_snapshot = working.model_copy(deep=True) if definition.mutating else None
        try:
            outcome = registry.execute(
                planned.tool,
                working,
                planned.arguments,
                AgentToolContext(goal=request.goal, planner_source=planner_source),
            )
            if outcome.validation.get("passed") is not True:
                raise AgentToolError(
                    f"{definition.validator} did not return a passing validation result"
                )
        except (AgentToolError, ValueError) as exc:
            completed = datetime.now(timezone.utc)
            failed_mutation_tool = planned.tool if definition.mutating else None
            validates_pending = pending_mutation is not None and planned.tool == pending_mutation[2]
            if validates_pending:
                snapshot, trace_index, _, mutation_signature = pending_mutation
                working = snapshot
                mutation_trace = traces[trace_index]
                mutation_trace.status = "rolled_back"
                mutation_trace.observation = (
                    f"{mutation_trace.observation} 后置验证失败，写操作已回滚。"
                )
                mutation_trace.validation = {
                    **mutation_trace.validation,
                    "post_validation_passed": False,
                    "post_validation_error": str(exc),
                }
                failed_signatures.add(mutation_signature)
                failed_mutation_tool = mutation_trace.tool
                pending_mutation = None
            else:
                failed_signatures.add(signature)
            traces.append(
                AgentTaskStepTrace(
                    index=len(traces),
                    attempt=replan_count + 1,
                    call_id=planned.call_id,
                    tool=planned.tool,
                    status="rolled_back" if definition.mutating else "error",
                    arguments=planned.arguments,
                    reason=planned.reason,
                    observation=str(exc),
                    validation={"passed": False, "error": str(exc), "validator": definition.validator},
                    dimension_score_before=score_before,
                    dimension_score_after=score_before,
                    mutating=definition.mutating,
                    reversible=definition.reversible,
                    started_at=started,
                    completed_at=completed,
                )
            )
            terminal_failure = True
            remaining_calls = request.max_tool_calls - tool_calls
            rollback_evaluator = (
                _policy_evaluator_step(failed_mutation_tool, "Validate the restored state before replanning.")
                if failed_mutation_tool is not None and remaining_calls > 0
                else None
            )
            replan_budget = remaining_calls - int(rollback_evaluator is not None)
            if request.use_llm and replan_count < request.max_replans and replan_budget > 0:
                llm_calls += 1
                try:
                    replanned, reason, model = plan_agent_task_llm(
                        request.goal,
                        working,
                        _catalog_payload(registry),
                        replan_budget,
                        execution_context=_execution_context(traces),
                    )
                except LlmUnavailable:
                    replanned = []
                if replanned:
                    replanned, injected = _enforce_post_mutation_evaluators(
                        replanned,
                        registry,
                        replan_budget,
                        goal=request.goal,
                    )
                    replan_count += 1
                    policy_injected_steps += injected
                    planner_model = model
                    planner_reason = f"{planner_reason} Replan: {reason}"
                    replanned_queue = [
                        step for step in replanned if _step_signature(step) not in failed_signatures
                    ]
                    queue = ([rollback_evaluator] if rollback_evaluator is not None else []) + replanned_queue
                    if rollback_evaluator is not None:
                        policy_injected_steps += 1
                    terminal_failure = not bool(queue)
            elif rollback_evaluator is not None:
                queue = [rollback_evaluator]
                policy_injected_steps += 1
            continue

        working = outcome.project
        completed = datetime.now(timezone.utc)
        score_after = _dimension_score(working)
        if planned.tool == "export_dxf":
            artifacts.update(
                {
                    key: value
                    for key, value in outcome.output.items()
                    if key in {"dxf_url", "svg_url"} and isinstance(value, str)
                }
            )
        traces.append(
            AgentTaskStepTrace(
                index=len(traces),
                attempt=replan_count + 1,
                call_id=planned.call_id,
                tool=planned.tool,
                status=outcome.status,
                arguments=planned.arguments,
                reason=planned.reason,
                observation=outcome.observation,
                output=outcome.output,
                validation=outcome.validation,
                dimension_score_before=score_before,
                dimension_score_after=score_after,
                mutating=definition.mutating,
                reversible=definition.reversible,
                started_at=started,
                completed_at=completed,
            )
        )
        if definition.mutating:
            evaluator = post_mutation_evaluator(planned.tool)
            if mutation_snapshot is not None and evaluator is not None:
                pending_mutation = (mutation_snapshot, len(traces) - 1, evaluator, signature)
        elif pending_mutation is not None and planned.tool == pending_mutation[2]:
            pending_mutation = None
        terminal_failure = False

    if queue:
        terminal_failure = True
    if not _goal_has_success_evidence(request.goal, working, traces):
        terminal_failure = True
    accepted = sum(trace.status == "accepted" for trace in traces)
    failed = sum(trace.status in {"error", "rolled_back"} for trace in traces)
    if terminal_failure and accepted:
        status = "partial"
    elif terminal_failure or not traces:
        status = "failed"
    else:
        status = "completed"
    after_score = _dimension_score(working)
    completed_at = datetime.now(timezone.utc)
    run = AgentTaskRun(
        id=new_id("agent_task"),
        goal=request.goal.strip(),
        status=status,
        planner_source=planner_source,
        planner_model=planner_model,
        planner_reason=planner_reason,
        initial_plan=initial_plan,
        steps=traces,
        llm_calls=llm_calls,
        replan_count=replan_count,
        policy_injected_steps=policy_injected_steps,
        max_tool_calls=request.max_tool_calls,
        before_dimension_score=before_score,
        after_dimension_score=after_score,
        summary=_task_summary(status, accepted, failed, traces),
        artifacts=artifacts,
        created_at=traces[0].started_at if traces else completed_at,
        completed_at=completed_at,
    )
    return working, run


def _clarification_run(
    project: ProjectState,
    request: AgentTaskRequest,
    registry: AgentToolRegistry,
    before_score: float | None,
    clarification: AgentTaskClarification,
) -> tuple[ProjectState, AgentTaskRun]:
    """Return a read-only inspection run instead of guessing an edit target."""

    now = datetime.now(timezone.utc)
    inspect = registry.execute(
        "inspect_drawing",
        project,
        AgentToolArguments(),
        AgentToolContext(goal=request.goal, planner_source="deterministic"),
    )
    step = AgentTaskStepTrace(
        index=0,
        attempt=1,
        call_id="policy_inspect_for_clarification",
        tool="inspect_drawing",
        status=inspect.status,
        arguments=AgentToolArguments(),
        reason="Runtime policy: inspect candidates before an ambiguous mutation.",
        observation=inspect.observation,
        output=inspect.output,
        validation=inspect.validation,
        mutating=False,
        reversible=False,
        started_at=now,
        completed_at=now,
    )
    run = AgentTaskRun(
        id=new_id("agent_task"),
        goal=request.goal.strip(),
        status="needs_clarification",
        planner_source="deterministic",
        planner_reason="Runtime blocked an ambiguous mutation before planning.",
        initial_plan=[
            AgentPlannedStep(
                call_id=step.call_id,
                tool=step.tool,
                arguments=step.arguments,
                reason=step.reason,
            )
        ],
        steps=[step],
        max_tool_calls=request.max_tool_calls,
        before_dimension_score=before_score,
        after_dimension_score=before_score,
        summary=clarification.question,
        clarification=clarification,
        created_at=now,
        completed_at=now,
    )
    return project, run


def append_agent_task_run(project: ProjectState, run: AgentTaskRun) -> None:
    project.agent_task_runs = [*project.agent_task_runs, run][-MAX_TASK_RUNS:]


def _deterministic_plan(goal: str, project: ProjectState, max_tool_calls: int) -> list[AgentPlannedStep]:
    lower = goal.lower()
    steps: list[AgentPlannedStep] = []

    def add(tool, arguments=None, reason=""):
        if len(steps) >= max_tool_calls:
            return
        steps.append(
            AgentPlannedStep(
                call_id=f"step_{len(steps) + 1}",
                tool=tool,
                arguments=arguments or AgentToolArguments(),
                reason=reason,
            )
        )

    add("inspect_drawing", reason="Observe the current drawing before acting.")
    wants_repair = any(term in lower for term in REPAIR_TERMS)
    wants_export = any(term in lower for term in EXPORT_TERMS)
    wants_evaluate = wants_repair or any(term in lower for term in EVALUATE_TERMS)
    mutating_planned = wants_repair
    if wants_evaluate:
        add("evaluate_dimensions", reason="Record the semantic baseline.")
    if wants_repair:
        add(
            "repair_dimensions",
            AgentToolArguments(max_steps=min(10, max_tool_calls), min_gain=0.01),
            "Repair locally supported incomplete dimensions.",
        )

    drive = _drive_arguments(goal, project)
    if drive is not None:
        add("drive_dimension", drive, "Apply the requested absolute semantic dimension change.")
        mutating_planned = True
    elif not wants_repair and not _explicit_no_edit(goal) and _has_generic_edit(goal, project):
        add("edit_cad", AgentToolArguments(message=goal), "Apply a deterministic CAD entity edit.")
        mutating_planned = True

    if mutating_planned and (wants_evaluate or drive is not None):
        add("evaluate_dimensions", reason="Observe the post-action semantic state.")
    if wants_export:
        add("export_dxf", reason="Generate final CAD artifacts after validated edits.")
    return steps[:max_tool_calls]


def _focused_clarification(goal: str, project: ProjectState) -> AgentTaskClarification | None:
    """Detect generic edits whose selected entity would otherwise be a parser guess.

    The deterministic parser intentionally offers ergonomic defaults for demos.
    The task runtime is stricter: once a drawing has several editable objects,
    an edit must name an id, a uniquely resolvable semantic label, or a spatial
    selector such as "left hole". This keeps the Agent from silently mutating
    whichever entity happens to be first in the IR.
    """

    if _explicit_no_edit(goal):
        return None
    operations, _ = plan_operations(goal, project.ir)
    target_ids = {
        operation.entity_id
        for operation in operations
        if operation.operation in {"modify_entity", "delete_entity", "move_entity", "set_layer"}
        and operation.entity_id
    }
    if not target_ids:
        return None

    entities = {entity.id: entity for entity in project.ir.entities}
    target = next((entities[entity_id] for entity_id in target_ids if entity_id in entities), None)
    editable = [entity for entity in project.ir.entities if _is_editable_entity(entity.id, project)]
    if target is None or len(editable) <= 1 or _has_explicit_target_reference(goal, target, editable):
        return None

    candidates = _candidate_entities(goal, target.type, editable)
    if not candidates:
        candidates = editable
    candidate_labels = [_entity_candidate_label(entity) for entity in candidates[:6]]
    candidate_hint = "、".join(candidate_labels)
    return AgentTaskClarification(
        reason="The requested mutation does not identify one editable entity unambiguously.",
        candidates=candidate_labels,
        question=(
            f"需要先确认要编辑的对象：{candidate_hint}。"
            "请使用对象 ID，或使用明确描述（例如“左边孔”“右边孔”）后再执行。"
        ),
    )


def _is_editable_entity(entity_id: str, project: ProjectState) -> bool:
    entity = next((item for item in project.ir.entities if item.id == entity_id), None)
    if entity is None:
        return False
    layers = {layer.name: layer for layer in project.ir.layers}
    layer = layers.get(entity.layer)
    return layer is None or (layer.editable and not layer.locked)


def _has_explicit_target_reference(goal: str, target, editable) -> bool:
    lower = goal.lower()
    if target.id.lower() in lower:
        return True
    if target.label and target.label.lower() in lower:
        return True
    if target.type == "circle":
        mentions_circle = any(token in lower for token in ("孔", "圆", "hole", "circle"))
        if mentions_circle and any(token in lower for token in ("左", "右", "left", "right")):
            return True
        return mentions_circle and len([item for item in editable if item.type == "circle"]) == 1
    if target.type == "rectangle":
        mentions_plate = any(token in lower for token in ("板", "矩形", "plate", "rectangle"))
        return mentions_plate and len([item for item in editable if item.type == "rectangle"]) == 1
    return False


def _candidate_entities(goal: str, target_type: str, editable):
    lower = goal.lower()
    if any(token in lower for token in ("孔", "圆", "hole", "circle")):
        return [entity for entity in editable if entity.type == "circle"]
    if any(token in lower for token in ("板", "矩形", "plate", "rectangle")):
        return [entity for entity in editable if entity.type == "rectangle"]
    same_type = [entity for entity in editable if entity.type == target_type]
    return same_type if len(same_type) > 1 else editable


def _entity_candidate_label(entity) -> str:
    if entity.type == "circle":
        return f"{entity.id}（圆孔，中心 {entity.cx:g}, {entity.cy:g}）"
    if entity.type == "rectangle":
        return f"{entity.id}（矩形板件）"
    return f"{entity.id}（{entity.type}）"


def _enforce_post_mutation_evaluators(
    plan: list[AgentPlannedStep],
    registry: AgentToolRegistry,
    max_tool_calls: int,
    *,
    goal: str = "",
) -> tuple[list[AgentPlannedStep], int]:
    """Canonicalize planner output and enforce evaluator transaction boundaries."""

    policy_plan: list[AgentPlannedStep] = []
    injected = 0
    for index, step in enumerate(plan):
        if step.tool == "evaluate_drawing" and (
            index == 0 or plan[index - 1].tool != "edit_cad"
        ):
            continue
        edit_message = step.arguments.message or ""
        if (
            step.tool == "edit_cad"
            and goal
            and (not edit_message or _numeric_tokens(edit_message) == _numeric_tokens(goal))
        ):
            step = step.model_copy(
                update={"arguments": step.arguments.model_copy(update={"message": goal})}
            )
        policy_plan.append(step)

    has_mutation = any(registry.definition(step.tool).mutating for step in policy_plan)
    wants_read_only_evaluation = any(term in goal.lower() for term in EVALUATE_TERMS)
    if (
        wants_read_only_evaluation
        and not has_mutation
        and not any(step.tool == "evaluate_dimensions" for step in policy_plan)
    ):
        insert_at = next(
            (index for index, step in enumerate(policy_plan) if step.tool == "export_dxf"),
            len(policy_plan),
        )
        policy_plan.insert(
            insert_at,
            AgentPlannedStep(
                call_id="policy_eval_read_only_dimensions",
                tool="evaluate_dimensions",
                reason="Runtime policy: evaluate semantic dimensions for this inspection request.",
            ),
        )
        injected += 1

    normalized: list[AgentPlannedStep] = []
    index = 0
    while index < len(policy_plan) and len(normalized) < max_tool_calls:
        step = policy_plan[index]
        definition = registry.definition(step.tool)
        if not definition.mutating:
            normalized.append(step)
            index += 1
            continue

        evaluator = post_mutation_evaluator(step.tool)
        if evaluator is None:
            index += 1
            continue
        needs_semantic_precheck = (
            step.tool in {"repair_dimensions", "drive_dimension"}
            and (not normalized or normalized[-1].tool != "evaluate_dimensions")
        )
        next_is_evaluator = (
            index + 1 < len(policy_plan) and policy_plan[index + 1].tool == evaluator
        )
        required_slots = 2 + int(needs_semantic_precheck)
        if len(normalized) + required_slots > max_tool_calls:
            index += 2 if next_is_evaluator else 1
            continue

        if needs_semantic_precheck:
            normalized.append(
                _policy_evaluator_step(
                    step.tool,
                    f"Runtime policy: inspect semantic dimensions before {step.tool}.",
                    sequence=len(normalized),
                )
            )
            injected += 1
        normalized.append(step)
        if next_is_evaluator:
            normalized.append(policy_plan[index + 1])
            index += 2
        else:
            normalized.append(
                _policy_evaluator_step(
                    step.tool,
                    f"Runtime policy: validate {step.tool} before continuing.",
                    sequence=len(normalized),
                )
            )
            injected += 1
            index += 1
    return normalized, injected


def _numeric_tokens(value: str) -> list[float]:
    return [float(token) for token in re.findall(r"[-+]?\d+(?:\.\d+)?", value)]


def _policy_evaluator_step(
    mutation_tool,
    reason: str,
    *,
    sequence: int = 1,
) -> AgentPlannedStep | None:
    evaluator = post_mutation_evaluator(mutation_tool)
    if evaluator is None:
        return None
    return AgentPlannedStep(
        call_id=f"policy_eval_{sequence}_{evaluator}",
        tool=evaluator,
        arguments=AgentToolArguments(),
        reason=reason,
    )


def _drive_arguments(goal: str, project: ProjectState) -> AgentToolArguments | None:
    if _explicit_no_edit(goal) or not EDIT_INTENT_RE.search(goal):
        return None
    deterministic = plan_mechanical_drive_deterministic(goal, project)
    if deterministic is not None:
        return AgentToolArguments(
            message=goal,
            dimension_id=deterministic.dimension_id,
            target_value=deterministic.target_value,
            anchor=deterministic.anchor,
        )
    values = _numbers_without_explicit_ids(goal)
    if not values:
        return None
    explicit_id = _explicit_dimension_id(goal)
    old_value = values[-2] if len(values) >= 2 else None
    dimension = _find_dimension(project.mechanical_ir.dimensions, explicit_id, old_value)
    if dimension is None:
        return None
    return AgentToolArguments(
        message=goal,
        dimension_id=dimension.id,
        target_value=values[-1],
    )


def _has_generic_edit(goal: str, project: ProjectState) -> bool:
    operations, _ = plan_operations(goal, project.ir)
    return bool(operations)


def _goal_has_success_evidence(
    goal: str,
    project: ProjectState,
    traces: list[AgentTaskStepTrace],
) -> bool:
    lower = goal.lower()
    successful = {
        trace.tool
        for trace in traces
        if trace.status in {"accepted", "skipped"}
    }
    requirements: list[bool] = []
    if any(term in lower for term in REPAIR_TERMS):
        requirements.append("repair_dimensions" in successful)
    if not _explicit_no_edit(goal) and (EDIT_INTENT_RE.search(goal) or _has_generic_edit(goal, project)):
        requirements.append(bool({"drive_dimension", "edit_cad"}.intersection(successful)))
    if any(term in lower for term in EVALUATE_TERMS):
        requirements.append(bool({"inspect_drawing", "evaluate_dimensions"}.intersection(successful)))
    if any(term in lower for term in EXPORT_TERMS):
        requirements.append("export_dxf" in successful)
    return all(requirements) if requirements else False


def _explicit_no_edit(goal: str) -> bool:
    lower = goal.lower()
    return any(term in lower for term in NO_EDIT_TERMS)


def _dimension_score(project: ProjectState) -> float | None:
    if not project.dimension_ground_truth:
        return None
    return evaluate_dimension_benchmark(project).overall_score


def _catalog_payload(registry: AgentToolRegistry) -> list[dict]:
    return [definition.model_dump(mode="json") for definition in registry.definitions()]


def _step_signature(step: AgentPlannedStep) -> str:
    return f"{step.tool}:{step.arguments.model_dump_json(exclude_none=True)}"


def _execution_context(traces: list[AgentTaskStepTrace]) -> list[dict]:
    return [
        {
            "tool": trace.tool,
            "arguments": trace.arguments.model_dump(mode="json", exclude_none=True),
            "status": trace.status,
            "observation": trace.observation,
            "validation": trace.validation,
        }
        for trace in traces[-8:]
    ]


def _task_summary(
    status: str,
    accepted: int,
    failed: int,
    traces: list[AgentTaskStepTrace],
) -> str:
    accepted_tools = [trace.tool for trace in traces if trace.status == "accepted"]
    if status == "completed":
        return f"任务完成：{accepted} 个工具调用通过验证（{' -> '.join(accepted_tools)}）。"
    if status == "partial":
        return f"任务部分完成：{accepted} 步通过，{failed} 步失败或回滚。"
    detail = next((trace.observation for trace in reversed(traces) if trace.status in {"error", "rolled_back"}), "没有可执行计划。")
    return f"任务失败：{detail}"

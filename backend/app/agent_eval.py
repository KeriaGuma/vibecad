from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .agent_eval_fixtures import build_agent_eval_fixture
from .agent_runtime import run_agent_task
from .models import (
    AgentEvalAssertion,
    AgentEvalCase,
    AgentEvalCaseResult,
    AgentEvalMetrics,
    AgentEvalReport,
    AgentEvalRequest,
    AgentTaskRequest,
    AgentTaskRun,
    ProjectState,
    new_id,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "evals" / "agent_tasks_v1.json"
MAX_EVAL_REPORTS = 10


def load_agent_eval_dataset() -> tuple[str, list[AgentEvalCase]]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return payload["version"], [AgentEvalCase.model_validate(item) for item in payload["cases"]]


def run_agent_eval(request: AgentEvalRequest) -> AgentEvalReport:
    version, available = load_agent_eval_dataset()
    by_id = {case.id: case for case in available}
    if request.case_ids:
        missing = [case_id for case_id in request.case_ids if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown agent eval cases: {', '.join(missing)}")
        cases = [by_id[case_id] for case_id in request.case_ids]
    else:
        cases = available
    cases = cases[: request.max_cases]
    created_at = datetime.now(timezone.utc)
    results = [_run_case(case, request.mode == "deepseek") for case in cases]
    completed_at = datetime.now(timezone.utc)
    return AgentEvalReport(
        id=new_id("agent_eval"),
        dataset_version=version,
        mode=request.mode,
        case_count=len(results),
        passed_count=sum(result.passed for result in results),
        metrics=_aggregate_metrics(results),
        cases=results,
        created_at=created_at,
        completed_at=completed_at,
    )


def append_agent_eval_report(project: ProjectState, report: AgentEvalReport) -> None:
    project.agent_eval_reports = [*project.agent_eval_reports, report][-MAX_EVAL_REPORTS:]


def _run_case(case: AgentEvalCase, use_llm: bool) -> AgentEvalCaseResult:
    initial = build_agent_eval_fixture(case.fixture, case.id)
    started = perf_counter()
    result, run = run_agent_task(
        initial,
        AgentTaskRequest(
            goal=case.goal,
            use_llm=use_llm,
            max_tool_calls=10,
            max_replans=1,
        ),
    )
    duration_ms = (perf_counter() - started) * 1000
    actual_tools = [trace.tool for trace in run.steps]
    precision, recall = _tool_selection(case.expected_tools, actual_tools)
    order_match = _is_subsequence(case.expected_tools, actual_tools)
    argument_accuracy = _argument_accuracy(case, run)
    assertion_results = [_evaluate_assertion(assertion, initial, result, run) for assertion in case.assertions]
    assertion_accuracy = sum(passed for passed, _ in assertion_results) / max(len(assertion_results), 1)
    failed_assertions = [detail for passed, detail in assertion_results if not passed]
    forbidden_called = sorted(set(case.forbidden_tools).intersection(actual_tools))
    safety_passed = not forbidden_called
    if forbidden_called:
        failed_assertions.append(f"forbidden tools called: {', '.join(forbidden_called)}")
    status_match = run.status == case.expected_status
    invalid_action_count = sum(trace.status in {"error", "rolled_back"} for trace in run.steps)
    rollback_observed = any(trace.status == "rolled_back" for trace in run.steps)
    f1 = _f1(precision, recall)
    score = round(
        (
            float(status_match)
            + f1
            + float(order_match)
            + argument_accuracy
            + assertion_accuracy
            + float(safety_passed)
        )
        / 6,
        4,
    )
    passed = (
        status_match
        and recall == 1.0
        and precision >= 0.8
        and order_match
        and argument_accuracy == 1.0
        and assertion_accuracy == 1.0
        and safety_passed
    )
    task_success = status_match and assertion_accuracy == 1.0 and safety_passed
    return AgentEvalCaseResult(
        case_id=case.id,
        category=case.category,
        goal=case.goal,
        passed=passed,
        task_success=task_success,
        score=score,
        status_match=status_match,
        tool_precision=precision,
        tool_recall=recall,
        tool_order_match=order_match,
        argument_accuracy=argument_accuracy,
        assertion_accuracy=assertion_accuracy,
        safety_passed=safety_passed,
        expected_tools=case.expected_tools,
        actual_tools=actual_tools,
        failed_assertions=failed_assertions,
        invalid_action_count=invalid_action_count,
        rollback_observed=rollback_observed,
        replan_count=run.replan_count,
        llm_calls=run.llm_calls,
        policy_injected_steps=run.policy_injected_steps,
        duration_ms=round(duration_ms, 3),
        run_status=run.status,
        planner_source=run.planner_source,
    )


def _evaluate_assertion(
    assertion: AgentEvalAssertion,
    initial: ProjectState,
    result: ProjectState,
    run: AgentTaskRun,
) -> tuple[bool, str]:
    if assertion.kind == "entity_field":
        entity = next((item for item in result.ir.entities if item.id == assertion.entity_id), None)
        actual = getattr(entity, assertion.field or "", None) if entity is not None else None
        passed = _values_equal(actual, assertion.expected)
        return passed, f"{assertion.entity_id}.{assertion.field}: expected {assertion.expected!r}, got {actual!r}"
    if assertion.kind == "entity_count":
        actual = sum(entity.type == assertion.field for entity in result.ir.entities)
        passed = _values_equal(actual, assertion.expected)
        return passed, f"{assertion.field} count: expected {assertion.expected!r}, got {actual!r}"
    if assertion.kind == "dimension_value":
        expected = float(assertion.expected)
        actual_values = [
            dimension.measured_value
            for dimension in result.mechanical_ir.dimensions
            if dimension.measured_value is not None
        ]
        passed = any(math.isclose(value, expected, abs_tol=0.01) for value in actual_values)
        return passed, f"dimension value: expected {expected:g}, got {actual_values!r}"
    if assertion.kind == "dimension_complete_min":
        actual = sum(dimension.status == "complete" for dimension in result.mechanical_ir.dimensions)
        passed = actual >= int(assertion.expected)
        return passed, f"complete dimensions: expected >= {assertion.expected}, got {actual}"
    if assertion.kind == "ir_unchanged":
        actual = initial.ir == result.ir
        passed = actual is bool(assertion.expected)
        return passed, f"IR unchanged: expected {assertion.expected!r}, got {actual!r}"
    if assertion.kind == "artifact_present":
        actual = bool(run.artifacts.get(assertion.artifact or ""))
        passed = actual is bool(assertion.expected)
        return passed, f"artifact {assertion.artifact}: expected {assertion.expected!r}, got {actual!r}"
    if assertion.kind == "no_mutating_calls":
        actual = not any(trace.mutating for trace in run.steps)
        passed = actual is bool(assertion.expected)
        return passed, f"no mutating calls: expected {assertion.expected!r}, got {actual!r}"
    if assertion.kind == "rollback_observed":
        actual = any(trace.status == "rolled_back" for trace in run.steps)
        passed = actual is bool(assertion.expected)
        return passed, f"rollback observed: expected {assertion.expected!r}, got {actual!r}"
    raise ValueError(f"Unsupported agent eval assertion: {assertion.kind}")


def _argument_accuracy(case: AgentEvalCase, run: AgentTaskRun) -> float:
    expected_count = 0
    matched_count = 0
    for tool, fields in case.expected_arguments.items():
        trace = next((item for item in run.steps if item.tool == tool), None)
        for field, expected in fields.items():
            expected_count += 1
            actual = getattr(trace.arguments, field, None) if trace is not None else None
            if _values_equal(actual, expected):
                matched_count += 1
    return matched_count / expected_count if expected_count else 1.0


def _tool_selection(expected: list[str], actual: list[str]) -> tuple[float, float]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    intersection = sum((expected_counts & actual_counts).values())
    return (
        round(intersection / max(len(actual), 1), 4),
        round(intersection / max(len(expected), 1), 4),
    )


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(any(candidate == item for candidate in iterator) for item in expected)


def _values_equal(actual, expected) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-6)
    return actual == expected


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 1e-12:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _aggregate_metrics(results: list[AgentEvalCaseResult]) -> AgentEvalMetrics:
    count = max(len(results), 1)
    total_calls = sum(len(result.actual_tools) for result in results)
    invalid_actions = sum(result.invalid_action_count for result in results)
    rollback_cases = [result for result in results if result.rollback_observed or "recovery" == result.category]
    replan_cases = [result for result in results if result.replan_count > 0]
    return AgentEvalMetrics(
        task_success_rate=round(sum(result.task_success for result in results) / count, 4),
        average_score=round(sum(result.score for result in results) / count, 4),
        tool_selection_precision=round(sum(result.tool_precision for result in results) / count, 4),
        tool_selection_recall=round(sum(result.tool_recall for result in results) / count, 4),
        tool_order_accuracy=round(sum(result.tool_order_match for result in results) / count, 4),
        argument_accuracy=round(sum(result.argument_accuracy for result in results) / count, 4),
        assertion_accuracy=round(sum(result.assertion_accuracy for result in results) / count, 4),
        safety_pass_rate=round(sum(result.safety_passed for result in results) / count, 4),
        invalid_action_rate=round(invalid_actions / max(total_calls, 1), 4),
        rollback_success_rate=round(
            sum(result.rollback_observed for result in rollback_cases) / max(len(rollback_cases), 1),
            4,
        ),
        replan_success_rate=round(
            sum(result.passed for result in replan_cases) / max(len(replan_cases), 1),
            4,
        ),
        average_tool_calls=round(total_calls / count, 3),
        average_llm_calls=round(sum(result.llm_calls for result in results) / count, 3),
        average_replans=round(sum(result.replan_count for result in results) / count, 3),
        average_duration_ms=round(sum(result.duration_ms for result in results) / count, 3),
        average_policy_injected_steps=round(
            sum(result.policy_injected_steps for result in results) / count,
            3,
        ),
    )

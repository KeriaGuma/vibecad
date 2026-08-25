"""LLM-backed CAD operation planner (Slice 1 skeleton, multi-provider).

This is the "real agent" replacement for the deterministic regex planner in
:mod:`app.agent`. It takes a natural-language edit request plus the current
drawing IR and asks an LLM to emit a list of structured CAD operations that map
1:1 onto the existing :class:`app.models.Operation` schema — so the rest of the
pipeline (``cad_ops.apply_operations`` → DXF/SVG export → diff) is unchanged.

Providers
---------
Most mainstream LLM APIs speak the **OpenAI Chat Completions** wire format, so we
standardise on it: one adapter (the ``openai`` SDK pointed at the provider's
``base_url``) covers DeepSeek, OpenAI, Moonshot/Kimi, Zhipu/GLM, Qwen/DashScope,
Groq, OpenRouter, Mistral, SiliconFlow, Gemini (its OpenAI-compat endpoint),
local Ollama/vLLM, and any other compatible endpoint. **Anthropic** is the one
mainstream exception and gets a small native adapter (``anthropic`` SDK).

Select a provider with env vars (all optional except the key):

    VIBECAD_LLM_PROVIDER   one of PROVIDERS below (default "deepseek")
    VIBECAD_LLM_API_KEY    overrides the provider's own key env var
    VIBECAD_LLM_BASE_URL   overrides the provider's base_url (use any endpoint)
    VIBECAD_LLM_MODEL      overrides the provider's default model

Each provider also reads its conventional key env var (e.g. ``DEEPSEEK_API_KEY``)
so existing setups work without renaming anything.

Structured output: OpenAI-compatible providers use JSON mode
(``response_format={"type": "json_object"}``); the returned text is validated
with :class:`LlmPlan`. A tolerant extractor strips markdown fences and locates
the JSON object, so providers that ignore ``response_format`` still parse.

Availability: if the needed SDK is missing or no API key is set,
:func:`plan_operations_llm` raises :class:`LlmUnavailable`; callers fall back to
the deterministic planner so offline dev and the test suite keep working with no
network and no key.

DeepSeek defaults to the low-cost ``deepseek-v4-flash`` model. Mechanical
dimension edits use a second, much narrower schema: the model may select a
semantic dimension and target value, but local code owns all geometry changes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from .models import (
    AgentPlannedStep,
    DimensionBenchmarkReport,
    DrawingIR,
    MechanicalOperation,
    Operation,
    ProjectState,
)

MAX_TOKENS = 4096


@dataclass(frozen=True)
class ProviderSpec:
    kind: Literal["openai", "anthropic"]
    base_url: str | None
    default_model: str
    key_envs: tuple[str, ...]


# Built-in provider defaults. base_url=None means the SDK's own default endpoint.
# Model defaults are sensible starting points; override with VIBECAD_LLM_MODEL.
PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec("openai", "https://api.deepseek.com", "deepseek-v4-flash", ("DEEPSEEK_API_KEY",)),
    "openai": ProviderSpec("openai", "https://api.openai.com/v1", "gpt-4o-mini", ("OPENAI_API_KEY",)),
    "moonshot": ProviderSpec("openai", "https://api.moonshot.cn/v1", "moonshot-v1-8k", ("MOONSHOT_API_KEY",)),
    "zhipu": ProviderSpec("openai", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", ("ZHIPUAI_API_KEY",)),
    "qwen": ProviderSpec(
        "openai",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    ),
    "groq": ProviderSpec("openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", ("GROQ_API_KEY",)),
    "openrouter": ProviderSpec("openai", "https://openrouter.ai/api/v1", "deepseek/deepseek-chat", ("OPENROUTER_API_KEY",)),
    "mistral": ProviderSpec("openai", "https://api.mistral.ai/v1", "mistral-large-latest", ("MISTRAL_API_KEY",)),
    "siliconflow": ProviderSpec("openai", "https://api.siliconflow.cn/v1", "deepseek-ai/DeepSeek-V3", ("SILICONFLOW_API_KEY",)),
    "gemini": ProviderSpec(
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-2.0-flash",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "ollama": ProviderSpec("openai", "http://localhost:11434/v1", "llama3.1", ()),  # local; key optional
    "anthropic": ProviderSpec("anthropic", None, "claude-opus-4-8", ("ANTHROPIC_API_KEY",)),
}
DEFAULT_PROVIDER = "deepseek"


class LlmUnavailable(RuntimeError):
    """Raised when the LLM path cannot run (no SDK / no API key / API error)."""


class ChangeField(BaseModel):
    """One field mutation. Split value types so the shape has no free-form dict."""

    key: str
    number_value: float | None = None
    text_value: str | None = None


class LlmOperation(BaseModel):
    op: Literal[
        "create_plate",
        "create_spur_gear_drawing",
        "add_entity",
        "modify_entity",
        "delete_entity",
        "move_entity",
        "set_layer",
    ]
    entity_id: str | None = None
    changes: list[ChangeField] = Field(default_factory=list)
    dx: float = 0
    dy: float = 0
    layer: str | None = None
    reason: str = ""


class LlmPlan(BaseModel):
    """Top-level structured response: a human reply plus the operations."""

    reply: str
    operations: list[LlmOperation] = Field(default_factory=list)


class MechanicalLlmPlan(BaseModel):
    intent: Literal["drive_dimension", "none"] = "none"
    dimension_id: str | None = None
    target_value: float | None = None
    anchor: Literal["start", "end"] = "start"
    confidence: float = Field(default=0, ge=0, le=1)
    reply: str = ""
    reason: str = ""


class SemanticRepairLlmPlan(BaseModel):
    target_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)
    reason: str = ""


class AgentTaskLlmPlan(BaseModel):
    goal_summary: str
    steps: list[AgentPlannedStep] = Field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    spec: ProviderSpec
    api_key: str
    base_url: str | None
    model: str


SYSTEM_PROMPT = """\
You are the planning brain of a 2D mechanical-CAD agent. The user sends an edit
request in natural language (Chinese or English). You are given the current
drawing as an IR summary. Respond with a single JSON object (and nothing else).

The JSON must match this shape:
{
  "reply": "<short reply in the user's language>",
  "operations": [
    {
      "op": "create_plate | create_spur_gear_drawing | add_entity | modify_entity | delete_entity | move_entity | set_layer",
      "entity_id": "<existing entity id, or null>",
      "changes": [{"key": "<field>", "number_value": <number or null>, "text_value": "<string or null>"}],
      "dx": <number>, "dy": <number>,
      "layer": "<layer name or null>",
      "reason": "<why>"
    }
  ]
}

Rules:
- Reference existing entities by their exact id from the IR summary.
- Numeric edits (radius/width/...) go in changes[].number_value; string edits in text_value.
- Pure moves use dx/dy; layer changes use op "set_layer" + layer.
- If the request is ambiguous or cannot be expressed, return an empty
  operations array and explain in reply.

Example — request "把左边孔直径改成 12":
{"reply": "已把左孔直径改为 12（半径 6）。", "operations": [
  {"op": "modify_entity", "entity_id": "hole_1", "changes": [{"key": "r", "number_value": 6, "text_value": null}], "dx": 0, "dy": 0, "layer": null, "reason": "diameter 12 -> radius 6"}
]}
"""

MECHANICAL_SYSTEM_PROMPT = """\
You are a low-cost intent planner for a 2D mechanical CAD system. Return one
JSON object and nothing else. You may only select an existing semantic
dimension and an absolute target value. Local deterministic code will modify
and validate the geometry; never propose entity coordinates or DXF code.

Schema:
{
  "intent": "drive_dimension | none",
  "dimension_id": "exact id from the catalog or null",
  "target_value": "positive number or null",
  "anchor": "start | end",
  "confidence": "number from 0 to 1",
  "reply": "short reply in the user's language",
  "reason": "short selection reason"
}

Rules:
- Only choose entries where driveable=true.
- Use intent=none when the target dimension or absolute value is ambiguous.
- A command such as '把总长改成250' means absolute target_value=250.
- Do not calculate a relative target unless the current nominal is explicit in
  the catalog and the user's arithmetic intent is unambiguous.
"""

SEMANTIC_REPAIR_SYSTEM_PROMPT = """\
You are the planning component of a constrained mechanical-drawing repair
agent. Return one JSON object and nothing else. You may only prioritize target
ids from the supplied repairable catalog. Local deterministic tools own all
geometry selection, mutation, evaluation, and rollback.

Schema:
{
  "target_ids": ["exact ground_truth_id"],
  "confidence": "number from 0 to 1",
  "reason": "short Chinese reason"
}

Rules:
- Return at most max_steps unique ids from repairable_targets.
- Prefer already matched linear dimensions with high text/kind confidence.
- Prefer targets where local tools can close missing arrows, extension lines,
  measured geometry, or definition points.
- Never invent ids, entity ids, coordinates, or CAD operations.
"""

AGENT_TASK_SYSTEM_PROMPT = """\
You are the task planner for a constrained 2D mechanical CAD agent. Return one
JSON object and nothing else. You may only select tools from the supplied tool
catalog. Local code owns geometry, validation, persistence, export, and
rollback. Never invent entity coordinates or arbitrary code.

Schema:
{
  "goal_summary": "short Chinese summary",
  "reason": "short planning reason",
  "steps": [
    {
      "call_id": "step_1",
      "tool": "exact tool name",
      "arguments": {
        "message": "original subtask or null",
        "dimension_id": "exact catalog id or null",
        "target_value": "positive number or null",
        "anchor": "start | end",
        "max_steps": 3,
        "min_gain": 0.01
      },
      "reason": "why this tool is needed"
    }
  ]
}

Rules:
- Start with inspect_drawing when the task depends on current drawing state.
- Use evaluate_dimensions before and after repair_dimensions.
- Use evaluate_drawing immediately after edit_cad.
- Use evaluate_dimensions immediately after drive_dimension.
- repair_dimensions repairs currently supported incomplete linear dimensions.
- drive_dimension requires an exact dimension_id and absolute target_value.
- A dimension may become driveable after repair_dimensions in the same plan.
- Use edit_cad only for simple generic entity edits that are not semantic
  dimension drives; pass the subtask in arguments.message.
- Put export_dxf last when the user asks to export or produce the final file.
- Do not include duplicate calls with identical arguments.
- Stay within max_tool_calls.
- When execution_context contains a failed call, choose a different valid tool
  or arguments. Do not repeat the same failed call.
"""


def plan_operations_llm(message: str, ir: DrawingIR) -> tuple[list[Operation], str]:
    """Plan CAD operations with the configured LLM. Raises LlmUnavailable on failure."""
    config = resolve_config()
    user_content = f"IR summary (JSON):\n{_ir_summary(ir)}\n\nEdit request:\n{message.strip()}"
    try:
        if config.spec.kind == "anthropic":
            raw = _complete_anthropic(config, SYSTEM_PROMPT, user_content)
        else:
            raw = _complete_openai(config, SYSTEM_PROMPT, user_content)
    except LlmUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - any SDK/network error → fall back
        raise LlmUnavailable(f"LLM request failed: {exc}") from exc

    try:
        plan = LlmPlan.model_validate_json(_extract_json(raw))
    except Exception as exc:  # noqa: BLE001 - malformed JSON / schema mismatch
        raise LlmUnavailable(f"LLM returned unparseable plan: {exc}") from exc

    return _to_operations(plan.operations), plan.reply


def plan_mechanical_operation_llm(
    message: str,
    project: ProjectState,
) -> tuple[MechanicalOperation | None, str]:
    """Use DeepSeek only to select a constrained semantic dimension operation."""

    config = resolve_config()
    if config.provider != "deepseek":
        raise LlmUnavailable("Mechanical semantic planning currently requires the DeepSeek provider")
    from .mechanical_drive import is_driveable_dimension

    catalog = [
        {
            "id": dimension.id,
            "binding_id": dimension.binding_id,
            "kind": dimension.kind,
            "text": dimension.text,
            "nominal": dimension.parsed.nominal,
            "orientation": dimension.orientation,
            "driveable": is_driveable_dimension(project, dimension),
        }
        for dimension in project.mechanical_ir.dimensions
    ]
    user_content = json.dumps(
        {"dimension_catalog": catalog, "edit_request": message.strip()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        raw = _complete_openai(config, MECHANICAL_SYSTEM_PROMPT, user_content)
        llm_plan = MechanicalLlmPlan.model_validate_json(_extract_json(raw))
    except LlmUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - network/schema failures stay offline-safe
        raise LlmUnavailable(f"DeepSeek mechanical plan failed: {exc}") from exc

    if llm_plan.intent == "none":
        return None, llm_plan.reply
    if not llm_plan.dimension_id or llm_plan.target_value is None or llm_plan.target_value <= 0:
        raise LlmUnavailable("DeepSeek returned an incomplete mechanical operation")
    valid_ids = {
        value
        for dimension in project.mechanical_ir.dimensions
        if is_driveable_dimension(project, dimension)
        for value in (dimension.id, dimension.binding_id)
    }
    if llm_plan.dimension_id not in valid_ids:
        raise LlmUnavailable("DeepSeek selected a missing or non-driveable dimension")
    return (
        MechanicalOperation(
            dimension_id=llm_plan.dimension_id,
            target_value=llm_plan.target_value,
            anchor=llm_plan.anchor,
            planner_source="deepseek",
            confidence=llm_plan.confidence,
            reason=llm_plan.reason,
        ),
        llm_plan.reply,
    )


def plan_dimension_repair_order_llm(
    report: DimensionBenchmarkReport,
    repairable_ids: list[str],
    max_steps: int,
) -> tuple[list[str], str, str]:
    """Ask DeepSeek only to prioritize locally repairable benchmark targets."""

    config = resolve_config()
    if config.provider != "deepseek":
        raise LlmUnavailable("Semantic repair planning currently requires the DeepSeek provider")
    repairable = set(repairable_ids)
    catalog = [
        {
            "ground_truth_id": target.ground_truth.id,
            "label": target.ground_truth.label,
            "kind": target.ground_truth.kind,
            "matched": target.matched_dimension_id is not None,
            "score": target.score,
            "missing_relations": target.missing_relations,
        }
        for target in report.targets
        if target.ground_truth.id in repairable
    ]
    user_content = json.dumps(
        {"max_steps": max_steps, "repairable_targets": catalog},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        raw = _complete_openai(config, SEMANTIC_REPAIR_SYSTEM_PROMPT, user_content)
        plan = SemanticRepairLlmPlan.model_validate_json(_extract_json(raw))
    except LlmUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - network/schema failures stay offline-safe
        raise LlmUnavailable(f"DeepSeek semantic repair plan failed: {exc}") from exc

    ordered: list[str] = []
    for target_id in plan.target_ids:
        if target_id in repairable and target_id not in ordered:
            ordered.append(target_id)
        if len(ordered) >= max_steps:
            break
    if not ordered:
        raise LlmUnavailable("DeepSeek returned no valid semantic repair targets")
    return ordered, plan.reason, config.model


def plan_agent_task_llm(
    goal: str,
    project: ProjectState,
    tool_catalog: list[dict],
    max_tool_calls: int,
    execution_context: list[dict] | None = None,
) -> tuple[list[AgentPlannedStep], str, str]:
    """Plan or replan a bounded CAD task using only registered tools."""

    config = resolve_config()
    if config.provider != "deepseek":
        raise LlmUnavailable("Task-level CAD planning currently requires the DeepSeek provider")
    from .mechanical_drive import is_driveable_dimension

    dimensions = [
        {
            "id": dimension.id,
            "binding_id": dimension.binding_id,
            "kind": dimension.kind,
            "text": dimension.text,
            "nominal": dimension.parsed.nominal,
            "status": dimension.status,
            "export_ready": dimension.export_ready,
            "driveable": is_driveable_dimension(project, dimension),
        }
        for dimension in project.mechanical_ir.dimensions
    ]
    ground_truth = [
        {
            "id": target.id,
            "label": target.label,
            "kind": target.kind,
            "nominal": target.nominal,
            "matched_dimension_id": target.matched_dimension_id,
        }
        for target in project.dimension_ground_truth
    ]
    user_content = json.dumps(
        {
            "goal": goal.strip(),
            "max_tool_calls": max_tool_calls,
            "tools": tool_catalog,
            "drawing": {
                "project_id": project.project_id,
                "source_kind": project.source_kind,
                "entity_count": len(project.ir.entities),
                "dimensions": dimensions,
                "dimension_ground_truth": ground_truth,
            },
            "execution_context": execution_context or [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        raw = _complete_openai(config, AGENT_TASK_SYSTEM_PROMPT, user_content)
        plan = AgentTaskLlmPlan.model_validate_json(_extract_json(raw))
    except LlmUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed plans use deterministic fallback
        raise LlmUnavailable(f"DeepSeek task plan failed: {exc}") from exc

    steps: list[AgentPlannedStep] = []
    seen: set[tuple[str, str]] = set()
    for index, step in enumerate(plan.steps[:max_tool_calls]):
        step.call_id = step.call_id.strip() or f"step_{index + 1}"
        signature = (step.tool, step.arguments.model_dump_json(exclude_none=True))
        if signature in seen:
            continue
        seen.add(signature)
        steps.append(step)
    if not steps:
        raise LlmUnavailable("DeepSeek returned an empty task plan")
    return steps, plan.reason, config.model


def resolve_config() -> LlmConfig:
    """Resolve provider, key, base_url, and model from env (with overrides)."""
    provider = os.environ.get("VIBECAD_LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LlmUnavailable(f"Unknown LLM provider '{provider}'. Known: {', '.join(sorted(PROVIDERS))}")

    api_key = os.environ.get("VIBECAD_LLM_API_KEY")
    if not api_key:
        for env in spec.key_envs:
            if os.environ.get(env):
                api_key = os.environ[env]
                break
    if not api_key and provider != "ollama":  # local ollama needs no key
        wanted = " / ".join(("VIBECAD_LLM_API_KEY", *spec.key_envs))
        raise LlmUnavailable(f"No API key set for provider '{provider}' ({wanted})")

    return LlmConfig(
        provider=provider,
        spec=spec,
        api_key=api_key or "ollama",
        base_url=os.environ.get("VIBECAD_LLM_BASE_URL", spec.base_url),
        model=os.environ.get("VIBECAD_LLM_MODEL", spec.default_model),
    )


def _complete_openai(config: LlmConfig, system: str, user: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dep optional in skeleton
        raise LlmUnavailable("openai SDK is not installed") from exc
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=MAX_TOKENS,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content or ""


def _complete_anthropic(config: LlmConfig, system: str, user: str) -> str:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - dep optional in skeleton
        raise LlmUnavailable("anthropic SDK is not installed") from exc
    kwargs = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    client = anthropic.Anthropic(**kwargs)
    message = client.messages.create(
        model=config.model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply (tolerant of ``` fences / prose)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned.strip()


def _to_operations(ops: list[LlmOperation]) -> list[Operation]:
    """Map the structured-output-safe LlmOperation onto the real Operation."""
    result: list[Operation] = []
    for op in ops:
        changes: dict[str, object] = {}
        for field in op.changes:
            changes[field.key] = field.number_value if field.number_value is not None else field.text_value
        result.append(
            Operation(
                operation=op.op,
                entity_id=op.entity_id,
                changes=changes,
                dx=op.dx,
                dy=op.dy,
                layer=op.layer,
                reason=op.reason,
            )
        )
    return result


def _ir_summary(ir: DrawingIR) -> str:
    """Compact JSON view of the IR for the model — ids, types, key geometry.

    Slice 2 will enrich this with OCR text / parameter-table data so the model
    can reason about the drawing's semantics, not just its geometry.
    """
    entities = []
    for entity in ir.entities:
        item = {"id": entity.id, "type": entity.type, "layer": entity.layer}
        if entity.label:
            item["label"] = entity.label
        entities.append(item)
    return json.dumps(
        {"units": ir.units, "layers": [layer.name for layer in ir.layers], "entities": entities},
        ensure_ascii=False,
    )

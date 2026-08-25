"""Offline tests for the multi-provider LLM planner skeleton (no network)."""
from __future__ import annotations

import pytest

from app import llm_agent
from app.llm_agent import ChangeField, LlmOperation, LlmPlan, LlmUnavailable


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch):
    for key in (
        "VIBECAD_LLM_PROVIDER",
        "VIBECAD_LLM_API_KEY",
        "VIBECAD_LLM_BASE_URL",
        "VIBECAD_LLM_MODEL",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_resolve_defaults_to_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    config = llm_agent.resolve_config()
    assert config.provider == "deepseek"
    assert config.spec.kind == "openai"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"
    assert config.api_key == "sk-test"


def test_resolve_provider_and_overrides(monkeypatch):
    monkeypatch.setenv("VIBECAD_LLM_PROVIDER", "groq")
    monkeypatch.setenv("VIBECAD_LLM_API_KEY", "key-override")
    monkeypatch.setenv("VIBECAD_LLM_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setenv("VIBECAD_LLM_BASE_URL", "https://proxy.example/v1")
    config = llm_agent.resolve_config()
    assert config.provider == "groq"
    assert config.api_key == "key-override"  # explicit override beats provider env
    assert config.model == "llama-3.1-8b-instant"
    assert config.base_url == "https://proxy.example/v1"


def test_resolve_anthropic_kind(monkeypatch):
    monkeypatch.setenv("VIBECAD_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    config = llm_agent.resolve_config()
    assert config.spec.kind == "anthropic"
    assert config.model == "claude-opus-4-8"


def test_resolve_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("VIBECAD_LLM_PROVIDER", "nope")
    with pytest.raises(LlmUnavailable, match="Unknown LLM provider"):
        llm_agent.resolve_config()


def test_resolve_missing_key_raises(monkeypatch):
    monkeypatch.setenv("VIBECAD_LLM_PROVIDER", "openai")
    with pytest.raises(LlmUnavailable, match="No API key"):
        llm_agent.resolve_config()


def test_resolve_ollama_needs_no_key(monkeypatch):
    monkeypatch.setenv("VIBECAD_LLM_PROVIDER", "ollama")
    config = llm_agent.resolve_config()
    assert config.provider == "ollama"
    assert config.base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "raw",
    [
        '{"reply":"ok","operations":[]}',
        '```json\n{"reply":"ok","operations":[]}\n```',
        'Sure, here you go:\n{"reply":"ok","operations":[]}\nThanks!',
    ],
)
def test_extract_json_is_tolerant(raw):
    assert LlmPlan.model_validate_json(llm_agent._extract_json(raw)).reply == "ok"


def test_plan_operations_llm_maps_provider_output(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    canned = (
        '{"reply":"已把左孔直径改为12","operations":['
        '{"op":"modify_entity","entity_id":"hole_1",'
        '"changes":[{"key":"r","number_value":6,"text_value":null}],'
        '"dx":0,"dy":0,"layer":null,"reason":"d12"}]}'
    )
    monkeypatch.setattr(llm_agent, "_complete_openai", lambda config, system, user: canned)

    from app.models import default_ir

    ops, reply = llm_agent.plan_operations_llm("把左边孔直径改成 12", default_ir())
    assert reply.startswith("已把左孔")
    assert len(ops) == 1
    assert ops[0].operation == "modify_entity"
    assert ops[0].entity_id == "hole_1"
    assert ops[0].changes == {"r": 6.0}


def test_plan_operations_llm_unavailable_without_key():
    from app.models import default_ir

    with pytest.raises(LlmUnavailable):
        llm_agent.plan_operations_llm("把右孔右移 12", default_ir())


def test_to_operations_text_value_mapping():
    ops = llm_agent._to_operations(
        [LlmOperation(op="add_entity", changes=[ChangeField(key="text", text_value="REV A")])]
    )
    assert ops[0].changes == {"text": "REV A"}


def test_deepseek_mechanical_planner_returns_constrained_operation(monkeypatch):
    from datetime import datetime, timezone

    from app.models import (
        DimensionBinding,
        DrawingIR,
        LineEntity,
        MechanicalDimensionObject,
        MechanicalDrawingIR,
        ParsedDimensionValue,
        ProjectState,
    )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)
    dimension = MechanicalDimensionObject(
        id="mechanical_dimension_dim_binding_244",
        binding_id="dim_binding_244",
        kind="linear",
        text="244",
        parsed=parsed,
        confidence=0.95,
        dimension_line_id="dim_line",
        extension_line_ids=["ext1", "ext2"],
        measured_geometry_ids=["outline"],
        measurement_points=[[0, 0], [244, 0]],
        orientation="horizontal",
        dxf_dimension_type="linear",
        export_ready=True,
        status="complete",
    )
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="p",
        name="p",
        created_at=now,
        updated_at=now,
        ir=DrawingIR(entities=[LineEntity(id="outline", x1=0, y1=0, x2=244, y2=0)]),
        dimension_bindings=[
            DimensionBinding(
                id="dim_binding_244",
                dimension_line_id="dim_line",
                text="244",
                parsed=parsed,
                confidence=0.95,
                kind="linear",
                line_x1=0,
                line_y1=10,
                line_x2=244,
                line_y2=10,
            )
        ],
        mechanical_ir=MechanicalDrawingIR(dimensions=[dimension]),
        mechanical_dimensions=[dimension.model_copy(deep=True)],
    )
    canned = (
        '{"intent":"drive_dimension","dimension_id":"mechanical_dimension_dim_binding_244",'
        '"target_value":250,"anchor":"start","confidence":0.93,'
        '"reply":"将总长改为250","reason":"matched total length"}'
    )

    def complete(config, system, user):
        assert config.model == "deepseek-v4-flash"
        assert "dimension_catalog" in user
        return canned

    monkeypatch.setattr(llm_agent, "_complete_openai", complete)
    operation, reply = llm_agent.plan_mechanical_operation_llm("总长调整到250", project)

    assert operation is not None
    assert operation.operation == "drive_dimension"
    assert operation.dimension_id == dimension.id
    assert operation.target_value == 250
    assert operation.planner_source == "deepseek"
    assert reply == "将总长改为250"


def test_deepseek_task_planner_returns_registered_tool_plan(monkeypatch):
    from datetime import datetime, timezone

    from app.models import ProjectState, default_ir

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="task-plan",
        name="task-plan",
        created_at=now,
        updated_at=now,
        ir=default_ir(),
    )
    canned = (
        '{"goal_summary":"检查并导出","reason":"先观察再导出",'
        '"steps":[{"call_id":"step_1","tool":"inspect_drawing","arguments":{},"reason":"inspect"},'
        '{"call_id":"step_2","tool":"export_dxf","arguments":{},"reason":"export"}]}'
    )

    def complete(config, system, user):
        assert config.model == "deepseek-v4-flash"
        assert "inspect_drawing" in user
        assert "execution_context" in user
        return canned

    monkeypatch.setattr(llm_agent, "_complete_openai", complete)
    steps, reason, model = llm_agent.plan_agent_task_llm(
        "检查图纸并导出 DXF",
        project,
        [
            {"name": "inspect_drawing", "description": "inspect"},
            {"name": "export_dxf", "description": "export"},
        ],
        4,
    )

    assert [step.tool for step in steps] == ["inspect_drawing", "export_dxf"]
    assert reason == "先观察再导出"
    assert model == "deepseek-v4-flash"

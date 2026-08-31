"""Offline tests for the multi-provider LLM planner skeleton (no network)."""
from __future__ import annotations

import json

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


def test_project_context_compresses_real_drawing_semantics():
    from datetime import datetime, timezone

    from app.models import (
        DimensionBinding,
        DrawingIR,
        LineEntity,
        MechanicalDimensionObject,
        MechanicalDrawingIR,
        OcrRegion,
        ParsedDimensionValue,
        ProjectState,
        TableCellOcr,
        TextEntity,
        TitleBlockCell,
    )

    now = datetime.now(timezone.utc)
    parsed = ParsedDimensionValue(kind="linear", raw_text="244", nominal=244)
    binding = DimensionBinding(
        id="binding_overall_length",
        dimension_line_id="dim_line",
        text_id="dim_text",
        arrow_ids=["arrow_left", "arrow_right"],
        text="244",
        parsed=parsed,
        confidence=0.96,
        kind="linear",
        line_x1=0,
        line_y1=10,
        line_x2=244,
        line_y2=10,
    )
    mechanical = MechanicalDimensionObject(
        id="dimension_overall_length",
        binding_id=binding.id,
        kind="linear",
        text="244",
        parsed=parsed,
        confidence=0.96,
        dimension_line_id="dim_line",
        text_id="dim_text",
        measured_geometry_ids=["outline"],
        target_geometry_ids=["outline"],
        orientation="horizontal",
        export_ready=True,
        status="complete",
    )
    project = ProjectState(
        project_id="semantic-context",
        name="semantic-context",
        created_at=now,
        updated_at=now,
        source_kind="scan",
        ir=DrawingIR(
            entities=[
                LineEntity(id="outline", label="零件外轮廓", x1=0, y1=0, x2=244, y2=0),
                LineEntity(id="dim_line", x1=0, y1=10, x2=244, y2=10),
                TextEntity(id="dim_text", x=122, y=14, text="244"),
            ]
        ),
        dimension_bindings=[binding],
        mechanical_ir=MechanicalDrawingIR(dimensions=[mechanical]),
        ocr_regions=[
            OcrRegion(
                target="dimensions",
                label="总长",
                text="总长 244 mm",
                confidence=0.93,
                x=0.1,
                y=0.1,
                width=0.2,
                height=0.04,
            ),
            OcrRegion(
                target="dimensions",
                label="低置信度",
                text="不应进入上下文",
                confidence=0.2,
                x=0.1,
                y=0.2,
                width=0.2,
                height=0.04,
            ),
        ],
        table_ocr_cells=[
            TableCellOcr(
                target="parameter_table",
                row=1,
                col=0,
                text="长度",
                confidence=0.92,
                x=0.7,
                y=0.4,
                width=0.1,
                height=0.04,
            ),
            TableCellOcr(
                target="parameter_table",
                row=1,
                col=1,
                text="244 mm",
                confidence=0.92,
                x=0.8,
                y=0.4,
                width=0.1,
                height=0.04,
            ),
        ],
        title_block_cells=[
            TitleBlockCell(
                row=0,
                col=0,
                text="零件名称",
                confidence=0.94,
                x=0.7,
                y=0.8,
                width=0.1,
                height=0.04,
            ),
            TitleBlockCell(
                row=0,
                col=1,
                text="支撑板",
                confidence=0.94,
                x=0.8,
                y=0.8,
                width=0.1,
                height=0.04,
            ),
        ],
    )

    context = llm_agent._project_context_summary(project)

    assert context["schema_version"] == "drawing_context_v1"
    assert context["drawing"]["entities"][0]["id"] == "dim_line"
    assert context["dimension_bindings"][0]["nominal"] == 244
    assert context["mechanical_dimensions"][0]["measured_geometry_ids"] == ["outline"]
    observations = context["semantic_observations"]
    assert observations["ocr_regions"] == [
        {
            "text": "总长 244 mm",
            "confidence": 0.93,
            "label": "总长",
            "target": "dimensions",
            "bounds": [0.1, 0.1, 0.2, 0.04],
        }
    ]
    assert [cell["text"] for cell in observations["parameter_table_cells"]] == ["长度", "244 mm"]
    assert [cell["text"] for cell in observations["title_block_cells"]] == ["零件名称", "支撑板"]


def test_project_context_enforces_bucket_and_text_limits():
    from datetime import datetime, timezone

    from app.models import OcrRegion, ProjectState, TableCellOcr, TitleBlockCell, default_ir

    now = datetime.now(timezone.utc)
    long_text = "X" * (llm_agent.MAX_CONTEXT_TEXT_CHARS + 30)
    project = ProjectState(
        project_id="bounded-context",
        name="bounded-context",
        created_at=now,
        updated_at=now,
        ir=default_ir(),
        ocr_regions=[
            OcrRegion(
                target="dimensions",
                label=f"ocr-{index}",
                text=long_text,
                confidence=0.9,
                x=0.1,
                y=index / 100,
                width=0.01,
                height=0.01,
            )
            for index in range(llm_agent.MAX_CONTEXT_OCR_REGIONS + 2)
        ],
        table_ocr_cells=[
            TableCellOcr(
                target="parameter_table",
                row=index,
                col=0,
                text=long_text,
                confidence=0.9,
                x=0.1,
                y=index / 100,
                width=0.01,
                height=0.01,
            )
            for index in range(llm_agent.MAX_CONTEXT_PARAMETER_CELLS + 2)
        ],
        title_block_cells=[
            TitleBlockCell(
                row=index,
                col=0,
                text=long_text,
                confidence=0.9,
                x=0.1,
                y=index / 100,
                width=0.01,
                height=0.01,
            )
            for index in range(llm_agent.MAX_CONTEXT_TITLE_BLOCK_CELLS + 2)
        ],
    )

    observations = llm_agent._project_context_summary(project)["semantic_observations"]

    assert len(observations["ocr_regions"]) == llm_agent.MAX_CONTEXT_OCR_REGIONS
    assert len(observations["parameter_table_cells"]) == llm_agent.MAX_CONTEXT_PARAMETER_CELLS
    assert len(observations["title_block_cells"]) == llm_agent.MAX_CONTEXT_TITLE_BLOCK_CELLS
    assert observations["omitted"] == {
        "ocr_regions": 2,
        "parameter_table_cells": 2,
        "title_block_cells": 2,
    }
    assert len(observations["ocr_regions"][0]["text"]) == llm_agent.MAX_CONTEXT_TEXT_CHARS


def test_generic_planner_receives_project_semantics(monkeypatch):
    from datetime import datetime, timezone

    from app.models import OcrRegion, ProjectState, default_ir

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    now = datetime.now(timezone.utc)
    project = ProjectState(
        project_id="generic-context",
        name="generic-context",
        created_at=now,
        updated_at=now,
        ir=default_ir(),
        ocr_regions=[
            OcrRegion(
                target="dimensions",
                label="孔径",
                text="左孔 Ø12",
                confidence=0.9,
                x=0.1,
                y=0.1,
                width=0.1,
                height=0.02,
            )
        ],
    )

    def complete(config, system, user):
        context_json = user.removeprefix("Drawing context (JSON):\n").split("\n\nEdit request:", 1)[0]
        context = json.loads(context_json)
        assert context["semantic_observations"]["ocr_regions"][0]["text"] == "左孔 Ø12"
        assert context["drawing"]["entities"]
        return '{"reply":"ok","operations":[]}'

    monkeypatch.setattr(llm_agent, "_complete_openai", complete)
    operations, reply = llm_agent.plan_operations_llm("把左孔改为 Ø14", project)
    assert operations == []
    assert reply == "ok"


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
        payload = json.loads(user)
        assert payload["drawing_context"]["schema_version"] == "drawing_context_v1"
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
        payload = json.loads(user)
        assert payload["drawing"]["schema_version"] == "drawing_context_v1"
        assert "semantic_observations" in payload["drawing"]
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

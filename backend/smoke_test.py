from app.agent import plan_operations
from app.cad_ops import apply_operation, apply_operations
from app.exporter import export_dxf, export_svg
from app.models import Operation
from app.models import default_ir
from app.structure_eval import evaluate_structure
from pathlib import Path


def main() -> None:
    ir = default_ir()
    ops, reply = plan_operations("创建 100 60 8 两个孔", ir)
    assert ops and ops[0].changes["hole_count"] == 2, reply

    ops, reply = plan_operations("把左边孔直径改成 10", ir)
    assert ops, reply
    ir, diffs = apply_operations(ir, ops)
    assert diffs

    try:
        apply_operation(ir, Operation(operation="delete_entity", entity_id="hole_404"))
    except ValueError as exc:
        assert "Entity not found" in str(exc)
    else:
        raise AssertionError("missing entity should raise ValueError")

    ops, reply = plan_operations("画圆柱直齿轮图", ir)
    assert ops and ops[0].operation == "create_spur_gear_drawing", reply
    ir, diffs = apply_operations(ir, ops)
    assert diffs and len(ir.entities) > 50
    report = evaluate_structure(ir)
    assert report.passed, report.model_dump()

    out = Path(__file__).resolve().parents[0] / "data" / "smoke"
    export_dxf(ir, out / "output.dxf")
    export_svg(ir, out / "preview.svg")
    assert (out / "output.dxf").exists()
    assert (out / "preview.svg").exists()
    print("smoke ok")


if __name__ == "__main__":
    main()

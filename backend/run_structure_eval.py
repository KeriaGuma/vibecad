from __future__ import annotations

import argparse
import json

from app.structure_eval import evaluate_structure
from app.templates import spur_gear_drawing_ir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current spur gear drawing structure target.")
    parser.add_argument("--json", action="store_true", help="Print the full eval report as JSON.")
    args = parser.parse_args()

    report = evaluate_structure(spur_gear_drawing_ir())
    if args.json:
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        return

    status = "PASS" if report.passed else "FAIL"
    print(f"structure_eval: {status} overall_score={report.overall_score:.3f}")
    for target in report.targets:
        target_status = "PASS" if target.passed else "FAIL"
        missing = ", ".join(target.missing) if target.missing else "-"
        print(f"- {target.name}: {target_status} score={target.score:.3f} missing={missing}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare generated-video GT fields with the real-video GT contract."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = ("perception", "cognition", "grounding", "planning")


def type_name(value: Any) -> str:
    return type(value).__name__


def collect_schema(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        filename = f"{dimension}_gt.json"
        field_types: dict[str, set[str]] = defaultdict(set)
        entry_count = 0
        file_count = 0
        for path in sorted(root.rglob(filename)):
            file_count += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for entry in payload:
                if not isinstance(entry, dict):
                    raise ValueError(f"Expected object entries in {path}")
                entry_count += 1
                for key, value in entry.items():
                    if key != "video_id":
                        field_types[key].add(type_name(value))
        result[dimension] = {
            "file_count": file_count,
            "entry_count": entry_count,
            "fields": {
                key: sorted(types) for key, types in sorted(field_types.items())
            },
        }
    return result


def compare(real: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    compatible = True
    for dimension in DIMENSIONS:
        real_fields = real[dimension]["fields"]
        generated_fields = generated[dimension]["fields"]
        missing_from_generated = sorted(set(real_fields) - set(generated_fields))
        extra_in_generated = sorted(set(generated_fields) - set(real_fields))
        type_mismatches = {
            key: {
                "real": real_fields[key],
                "generated": generated_fields[key],
            }
            for key in sorted(set(real_fields) & set(generated_fields))
            if real_fields[key] != generated_fields[key]
        }
        dimension_compatible = not (
            missing_from_generated or extra_in_generated or type_mismatches
        )
        compatible = compatible and dimension_compatible
        dimensions[dimension] = {
            "compatible": dimension_compatible,
            "real": real[dimension],
            "generated": generated[dimension],
            "missing_from_generated": missing_from_generated,
            "extra_in_generated": extra_in_generated,
            "type_mismatches": type_mismatches,
        }
    return {"compatible": compatible, "dimensions": dimensions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-gt-dir", type=Path, required=True)
    parser.add_argument("--generated-gt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    real = collect_schema(args.real_gt_dir)
    generated = collect_schema(args.generated_gt_dir)
    report = compare(real, generated)
    report["real_gt_dir"] = str(args.real_gt_dir.resolve())
    report["generated_gt_dir"] = str(args.generated_gt_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

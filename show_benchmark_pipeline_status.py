#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "benchmark_inference"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
STATUS_ROOT = OUTPUT_ROOT / "status"


def read_pid(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def main() -> int:
    payload = {
        "download_queue_pid": read_pid(MODELS_ROOT / "download_all_benchmark_models_mirror_fast.pid"),
        "autopipeline_pid": None,
        "summary": None,
        "status_files": {},
    }

    for candidate in ("benchmark_autopipeline.pid", "auto_infer.pid"):
        pid = read_pid(OUTPUT_ROOT / candidate)
        if pid:
            payload["autopipeline_pid"] = pid
            break

    if SUMMARY_PATH.exists():
        payload["summary"] = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    for status_file in sorted(STATUS_ROOT.glob("*.json")):
        payload["status_files"][status_file.stem] = json.loads(status_file.read_text(encoding="utf-8"))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

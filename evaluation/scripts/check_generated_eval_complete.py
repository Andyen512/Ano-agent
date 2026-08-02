#!/usr/bin/env python3
"""Verify that every prediction for one generated-video model was evaluated."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.official_evaluation.real_videos.common_real import (
    _build_basename_index,
    discover_prediction_files,
    load_real_video_gt,
    match_prediction_to_gt,
    parse_prediction_response,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Incomplete: summary not found: {args.summary}")
        return 1

    gt_by_video = load_real_video_gt(args.gt_dir)
    basename_index = _build_basename_index(gt_by_video)
    payloads = discover_prediction_files(args.predictions_root, args.model_id)

    matched_keys: set[tuple[str, str]] = set()
    unmatched = 0
    parse_fail = 0
    for payload in payloads:
        source_path = payload.get("__source_path__", "")
        gt_video_id = match_prediction_to_gt(
            Path(source_path).stem if source_path else "",
            payload.get("video_path"),
            gt_by_video,
            basename_index,
        )
        if gt_video_id is None:
            unmatched += 1
            continue
        if parse_prediction_response(payload) is None:
            parse_fail += 1
            continue
        matched_keys.add((payload.get("model_id", ""), gt_video_id))

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    evaluated = int(summary.get("total_predictions_evaluated", -1))
    model_summary = summary.get("model_summary", {}).get(args.model_id, {})
    sample_count = int(model_summary.get("sample_count", -1))
    expected = len(matched_keys)

    print(
        f"{args.model_id}: predictions={len(payloads)}, expected={expected}, "
        f"evaluated={evaluated}, sample_count={sample_count}, "
        f"unmatched={unmatched}, parse_fail={parse_fail}"
    )
    if unmatched or parse_fail or evaluated != expected or sample_count != expected:
        print("Incomplete: evaluation gate failed")
        return 1

    print("Complete: evaluation gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_view_risk_review import summarize_results


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_outputs" / "persistent_full"
DEFAULT_OUTPUT_STEM = "batch_risk_judgment_from_existing_outputs"
PUBLIC_DATA_ROOT = PROJECT_ROOT / "data" / "public_data"


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate current five-model review outputs and decide whether each video contains risk."
        )
    )
    parser.add_argument(
        "--reviews-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "reviews",
        help="Root directory that stores per-video review outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for aggregated JSON/CSV outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on the number of review folders to aggregate.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def video_relpath_from_summary(summary_path: Path, payload: dict[str, Any]) -> str:
    video_path = str(payload.get("manifest", {}).get("video_path", "")).strip()
    if video_path:
        try:
            return str(Path(video_path).resolve().relative_to(PUBLIC_DATA_ROOT.resolve()))
        except ValueError:
            return video_path
    parts = summary_path.parts
    if "reviews" in parts:
        idx = parts.index("reviews")
        return str(Path(*parts[idx + 1 :]).with_suffix(".mp4"))
    return str(summary_path)


def discover_review_summaries(reviews_root: Path) -> list[Path]:
    summary_paths: list[Path] = []
    for review_dir in sorted(path for path in reviews_root.rglob("*") if path.is_dir()):
        repaired = review_dir / "review_summary.repaired.json"
        original = review_dir / "review_summary.json"
        if repaired.exists():
            summary_paths.append(repaired)
        elif original.exists():
            summary_paths.append(original)
    return summary_paths


def final_status_from_majority(majority: dict[str, Any]) -> str:
    if majority.get("complete") is not True:
        return "incomplete"
    if majority.get("keep") is True:
        return "risk"
    if majority.get("keep") is False:
        return "safe"
    return "unknown"


def summarize_agents(agent_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in agent_results:
        parsed = item.get("parsed_decision", {}) or {}
        model = item.get("model", {}) or {}
        perspective = item.get("perspective", {}) or {}
        rows.append(
            {
                "perspective": perspective.get("name"),
                "backend": model.get("backend"),
                "model_id": model.get("model_id"),
                "returncode": item.get("returncode"),
                "keep": parsed.get("keep"),
                "risk_presence": parsed.get("risk_presence"),
                "level1_scene": parsed.get("level1_scene"),
                "level2_subject": parsed.get("level2_subject"),
                "level3_risk_type": parsed.get("level3_risk_type"),
                "output_json": item.get("output_json"),
            }
        )
    return rows


def row_from_summary(summary_path: Path) -> dict[str, Any]:
    payload = read_json(summary_path)
    agent_results = payload.get("agent_results", [])
    majority = summarize_results(agent_results, expected_votes=len(agent_results) or 5)
    elapsed = max([float(item.get("elapsed_seconds") or 0.0) for item in agent_results] or [0.0])
    video_path = str(payload.get("manifest", {}).get("video_path", "")).strip()
    return {
        "video_path": video_path,
        "video_relpath": video_relpath_from_summary(summary_path, payload),
        "summary_json": str(summary_path),
        "elapsed_seconds": elapsed,
        "final_status": final_status_from_majority(majority),
        "has_risk": True if majority.get("keep") is True else False if majority.get("keep") is False else None,
        "majority_vote": majority,
        "agent_votes": summarize_agents(agent_results),
    }


def summarize_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["final_status"] for row in rows)
    return {
        "processed_videos": len(rows),
        "counts": dict(counts),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_relpath",
                "video_path",
                "final_status",
                "has_risk",
                "risk_presence",
                "yes_count",
                "no_count",
                "votes_cast",
                "expected_votes",
                "threshold",
                "level1_scene_top",
                "level2_subject_top",
                "level3_risk_type_top",
                "normal_level1_scene_top",
                "normal_level2_subject_top",
                "kept_by_perspectives",
                "rejected_by_perspectives",
                "agent_vote_summary",
                "elapsed_seconds",
                "summary_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            majority = row["majority_vote"]
            writer.writerow(
                {
                    "video_relpath": row["video_relpath"],
                    "video_path": row["video_path"],
                    "final_status": row["final_status"],
                    "has_risk": row["has_risk"],
                    "risk_presence": majority.get("risk_presence"),
                    "yes_count": majority.get("yes_count"),
                    "no_count": majority.get("no_count"),
                    "votes_cast": majority.get("votes_cast"),
                    "expected_votes": majority.get("expected_votes"),
                    "threshold": majority.get("threshold"),
                    "level1_scene_top": majority.get("level1_scene_top"),
                    "level2_subject_top": majority.get("level2_subject_top"),
                    "level3_risk_type_top": majority.get("level3_risk_type_top"),
                    "normal_level1_scene_top": majority.get("normal_level1_scene_top"),
                    "normal_level2_subject_top": majority.get("normal_level2_subject_top"),
                    "kept_by_perspectives": "|".join(majority.get("kept_by_perspectives", [])),
                    "rejected_by_perspectives": "|".join(majority.get("rejected_by_perspectives", [])),
                    "agent_vote_summary": "; ".join(
                        f"{item['perspective']}={item['risk_presence']}"
                        for item in row["agent_votes"]
                    ),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "summary_json": row["summary_json"],
                }
            )


def main() -> int:
    args = build_parser().parse_args()
    reviews_root = args.reviews_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not reviews_root.exists():
        raise FileNotFoundError(f"reviews root does not exist: {reviews_root}")

    summary_paths = discover_review_summaries(reviews_root)
    if args.limit is not None:
        summary_paths = summary_paths[: max(0, args.limit)]

    rows = [row_from_summary(path) for path in summary_paths]
    rows.sort(key=lambda item: item["video_relpath"])

    payload = {
        "manifest": {
            "generated_at": utc_tag(),
            "reviews_root": str(reviews_root),
            "output_root": str(output_root),
            "output_stem": args.output_stem,
            "limit": args.limit,
            "aggregation_rule": "5-model majority vote; keep=True means risky, keep=False means safe",
        },
        "summary": summarize_batch(rows),
        "results": rows,
    }

    json_path = output_root / f"{args.output_stem}.json"
    csv_path = output_root / f"{args.output_stem}.csv"
    write_json(json_path, payload)
    write_csv(csv_path, rows)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(str(json_path))
    print(str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = (
    SCRIPT_DIR
    / "batch_outputs"
    / "persistent_full"
    / "batch_risk_judgment_from_existing_outputs.json"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_outputs" / "persistent_full"
DEFAULT_OUTPUT_STEM = "risk_vote_ranking"


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-video risk vote counts and the overall 5-to-0 vote distribution."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Aggregated review JSON produced from existing five-model outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for ranking outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem without extension.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def risk_votes_from_row(row: dict[str, Any]) -> int:
    return sum(1 for item in row.get("agent_votes", []) if item.get("keep") is True)


def safe_votes_from_row(row: dict[str, Any]) -> int:
    return sum(1 for item in row.get("agent_votes", []) if item.get("keep") is False)


def build_vote_row(row: dict[str, Any]) -> dict[str, Any]:
    risk_votes = risk_votes_from_row(row)
    safe_votes = safe_votes_from_row(row)
    agent_votes = row.get("agent_votes", [])
    incomplete_votes = len(agent_votes) - risk_votes - safe_votes
    return {
        "video_relpath": row.get("video_relpath"),
        "video_path": row.get("video_path"),
        "risk_votes": risk_votes,
        "safe_votes": safe_votes,
        "incomplete_votes": incomplete_votes,
        "final_status": row.get("final_status"),
        "has_risk": row.get("has_risk"),
        "summary_json": row.get("summary_json"),
        "agent_vote_summary": "; ".join(
            f"{item.get('backend')}={item.get('risk_presence')}"
            for item in agent_votes
        ),
    }


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            -int(item["risk_votes"]),
            int(item["incomplete_votes"]),
            str(item["video_relpath"]),
        ),
    )


def build_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(int(row["risk_votes"]) for row in rows)
    return {str(votes): counter.get(votes, 0) for votes in range(5, -1, -1)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_relpath",
                "video_path",
                "risk_votes",
                "safe_votes",
                "incomplete_votes",
                "final_status",
                "has_risk",
                "agent_vote_summary",
                "summary_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    input_json = args.input_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_json.exists():
        raise FileNotFoundError(f"input json does not exist: {input_json}")

    payload = read_json(input_json)
    source_rows = payload.get("results", [])
    rows = sort_rows([build_vote_row(row) for row in source_rows])
    distribution = build_distribution(rows)

    summary = {
        "generated_at": utc_tag(),
        "input_json": str(input_json),
        "processed_videos": len(rows),
        "risk_vote_distribution": distribution,
    }

    csv_path = output_root / f"{args.output_stem}.csv"
    json_path = output_root / f"{args.output_stem}.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "summary": summary,
            "results": rows,
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(csv_path))
    print(str(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

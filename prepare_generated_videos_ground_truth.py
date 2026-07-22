#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.common import parse_ground_truth_tsv_line


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS_ROOT = PROJECT_ROOT / "data" / "generated_videos"
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "data" / "prompts.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract GT for data/generated_videos and write both TSV and parsed JSON."
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help="Root directory of generated videos. Default: data/generated_videos.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="Prompt source file in prompts.txt format.",
    )
    parser.add_argument(
        "--ground-truth-out",
        type=Path,
        help="Output TSV GT path. Default: <videos-root>/gen_prompt_generated_videos_v2.txt",
    )
    parser.add_argument(
        "--parsed-ground-truth-out",
        type=Path,
        help="Output parsed JSON GT path. Default: <videos-root>/gen_prompt_generated_videos_v2.parsed.json",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Output summary JSON path. Default: <videos-root>/gen_prompt_generated_videos_v2.summary.json",
    )
    return parser


def default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stem = "gen_prompt_generated_videos_v2"
    return (
        args.ground_truth_out or (args.videos_root / f"{stem}.txt"),
        args.parsed_ground_truth_out or (args.videos_root / f"{stem}.parsed.json"),
        args.summary_json or (args.videos_root / f"{stem}.summary.json"),
    )


def main() -> int:
    args = build_parser().parse_args()
    args.videos_root = args.videos_root.resolve()
    args.prompts_file = args.prompts_file.resolve()
    ground_truth_out, parsed_ground_truth_out, summary_json = default_paths(args)
    ground_truth_out = ground_truth_out.resolve()
    parsed_ground_truth_out = parsed_ground_truth_out.resolve()
    summary_json = summary_json.resolve()

    video_files = sorted(args.videos_root.rglob("*.mp4"))
    video_ids = {path.stem for path in video_files}

    matched_lines: list[str] = []
    parsed_entries: list[dict[str, Any]] = []
    prompt_ids: list[str] = []
    matched_ids: list[str] = []

    for raw_line in args.prompts_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        video_id = line.split("\t", 1)[0].strip()
        prompt_ids.append(video_id)
        if video_id not in video_ids:
            continue
        matched_lines.append(line)
        matched_ids.append(video_id)
        parsed_entries.append(parse_ground_truth_tsv_line(line).to_serializable_dict())

    prompt_id_set = set(prompt_ids)
    missing_prompt_ids = sorted(video_ids - prompt_id_set)

    ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_out.write_text(
        "\n".join(matched_lines) + ("\n" if matched_lines else ""),
        encoding="utf-8",
    )

    parsed_ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
    parsed_ground_truth_out.write_text(
        json.dumps(
            {
                "format": "lifebench_ground_truth_parsed_v1",
                "source_tsv": str(ground_truth_out),
                "source_prompts": str(args.prompts_file),
                "videos_root": str(args.videos_root),
                "entry_count": len(parsed_entries),
                "entries": parsed_entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "videos_root": str(args.videos_root),
        "prompts_path": str(args.prompts_file),
        "ground_truth_out": str(ground_truth_out),
        "parsed_ground_truth_out": str(parsed_ground_truth_out),
        "video_file_count": len(video_files),
        "video_id_count": len(video_ids),
        "matched_prompt_count": len(matched_lines),
        "missing_prompt_count": len(missing_prompt_ids),
        "missing_prompt_ids": missing_prompt_ids,
        "matched_ids": matched_ids,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

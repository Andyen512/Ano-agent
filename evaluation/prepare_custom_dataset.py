#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


RELATIVE_RANGE_RE = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)\s*(?:-|–|—|~|到|至)\s*(?P<end>\d+(?:\.\d+)?)\s*(?:s|秒)\s*[:：]?"
)
CLOCK_RANGE_RE = re.compile(
    r"(?P<start_m>\d{1,2}):(?P<start_s>\d{2})\s*(?:-|–|—|~|到|至)\s*"
    r"(?P<end_m>\d{1,2}):(?P<end_s>\d{2})\s*[:：]?"
)

SOLUTION_SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("person_solution", "对人的解决方案："),
    ("hazard_solution", "对危险源的解决方案："),
    ("prevention_solution", "后续防止危险复发的解决方案："),
)


@dataclass(frozen=True)
class PromptRecord:
    video_id: str
    scenario_path: str
    description: str
    solution_text: str = ""


@dataclass(frozen=True)
class TimeSegment:
    start: float
    end: float
    description: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a custom LifeBench dataset into flat videos plus gen_prompt ground truth.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset directory, e.g. data/data_0315.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="A_expand-style prompt file. Supports an optional 4th tab-separated solution column.",
    )
    parser.add_argument("--zip-path", type=Path, help="Optional dataset zip to extract before scanning videos.")
    parser.add_argument("--extract-dir", type=Path, help="Directory used to extract the zip. Defaults to <dataset-root>/extracted.")
    parser.add_argument("--flat-videos-dir", type=Path, help="Output directory for flattened videos. Defaults to <dataset-root>/videos_flat.")
    parser.add_argument(
        "--ground-truth-out",
        type=Path,
        help="Output path for generated gen_prompt-compatible txt. Defaults to <dataset-root>/gen_prompt_custom.txt.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Output path for preparation summary JSON. Defaults to <dataset-root>/prepared_summary.json.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize flattened videos. Default: symlink.",
    )
    parser.add_argument(
        "--include-unmatched-videos",
        action="store_true",
        help="Also flatten videos without a matching prompt record. Ground truth still only includes the intersection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite extracted files, flattened targets, and generated outputs when they already exist.",
    )
    return parser


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u3000", " ")).strip()


def normalize_segment_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = text.lstrip("：:，,；;。 ")
    text = text.replace("\n", " ")
    # The ground-truth parser uses "；" between segments and stops at "。".
    text = text.replace("；", "，").replace(";", ",")
    text = text.replace("。", "，")
    return text.strip("，, ")


def normalize_solution_section(text: str) -> str:
    text = normalize_whitespace(text).replace("\n", " ")
    text = text.strip("：:，,；;。 ")
    text = text.replace("。", "；")
    return text.strip("；;，, ")


def parse_solution_sections(text: str) -> dict[str, str]:
    sections = {key: "" for key, _ in SOLUTION_SECTION_LABELS}
    normalized = normalize_whitespace(text).replace(":", "：")
    if not normalized:
        return sections

    positions: list[tuple[str, int, str]] = []
    for key, label in SOLUTION_SECTION_LABELS:
        position = normalized.find(label)
        if position == -1:
            return sections
        positions.append((key, position, label))
    positions.sort(key=lambda item: item[1])

    for index, (key, start_pos, label) in enumerate(positions):
        start = start_pos + len(label)
        end = positions[index + 1][1] if index + 1 < len(positions) else len(normalized)
        sections[key] = normalize_solution_section(normalized[start:end])
    return sections


def parse_prompt_records(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    current_id: str | None = None
    current_scenario: str | None = None
    current_lines: list[str] = []
    current_solution_text = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        parts = line.split("\t")
        if len(parts) >= 3:
            if current_id is not None:
                records.append(
                    PromptRecord(
                        video_id=current_id,
                        scenario_path=current_scenario or "",
                        description="\n".join(current_lines).strip(),
                        solution_text=current_solution_text,
                    )
                )
            current_id = parts[0].strip()
            current_scenario = parts[1].strip()
            current_lines = [parts[2].strip()]
            current_solution_text = "\t".join(parts[3:]).strip() if len(parts) >= 4 else ""
            continue
        if current_id is None:
            continue
        current_lines.append(line)

    if current_id is not None:
        records.append(
            PromptRecord(
                video_id=current_id,
                scenario_path=current_scenario or "",
                description="\n".join(current_lines).strip(),
                solution_text=current_solution_text,
            )
        )
    return records


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = extract_dir / member.filename
            if target.exists() and not overwrite:
                continue
            archive.extract(member, path=extract_dir)


def scan_videos(dataset_root: Path, extract_dir: Path | None) -> dict[str, Path]:
    roots: list[Path] = []
    if extract_dir is not None and extract_dir.exists():
        roots.append(extract_dir)
    roots.append(dataset_root)

    by_id: dict[str, Path] = {}
    for root in roots:
        for path in sorted(root.rglob("*.mp4")):
            if path.name.startswith("."):
                continue
            by_id.setdefault(path.stem, path)
    return by_id


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize_video(source: Path, target: Path, link_mode: str, overwrite: bool) -> None:
    ensure_parent(target)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()

    if link_mode == "symlink":
        target.symlink_to(source.resolve())
        return
    if link_mode == "hardlink":
        os.link(source, target)
        return
    shutil.copy2(source, target)


def parse_clock_seconds(match: re.Match[str]) -> tuple[float, float]:
    start = int(match.group("start_m")) * 60 + int(match.group("start_s"))
    end = int(match.group("end_m")) * 60 + int(match.group("end_s"))
    return float(start), float(end)


def parse_relative_seconds(match: re.Match[str]) -> tuple[float, float]:
    return float(match.group("start")), float(match.group("end"))


def extract_time_segments(text: str) -> list[TimeSegment]:
    matches: list[tuple[int, int, float, float]] = []
    for match in RELATIVE_RANGE_RE.finditer(text):
        start, end = parse_relative_seconds(match)
        matches.append((match.start(), match.end(), start, end))
    for match in CLOCK_RANGE_RE.finditer(text):
        start, end = parse_clock_seconds(match)
        matches.append((match.start(), match.end(), start, end))

    matches.sort(key=lambda item: item[0])
    if not matches:
        return []

    clock_starts = [start for _, _, start, _ in matches if start > 59]
    relative_offset = min(clock_starts) if clock_starts else 0.0

    segments: list[TimeSegment] = []
    for index, (start_pos, end_pos, start_value, end_value) in enumerate(matches):
        next_pos = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        description = normalize_segment_text(text[end_pos:next_pos])
        if not description:
            continue

        start = start_value
        end = end_value
        if start_value > 59 or end_value > 59:
            start -= relative_offset
            end -= relative_offset
        if end <= start:
            continue
        segments.append(TimeSegment(start=round(start, 3), end=round(end, 3), description=description))
    return segments


def scenario_parts(path_text: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in path_text.split("/") if part.strip()]
    location = parts[0] if parts else ""
    subject = parts[1] if len(parts) > 1 else ""
    scenario = parts[2] if len(parts) > 2 else path_text.strip()
    return location, subject, scenario


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def to_ground_truth_line(record: PromptRecord) -> str:
    location, subject, scenario = scenario_parts(record.scenario_path)
    segments = extract_time_segments(record.description)
    solution_sections = parse_solution_sections(record.solution_text)

    if not segments:
        fallback_desc = normalize_segment_text(record.description) or normalize_segment_text(scenario) or "视频中出现居家安全风险"
        segments = [TimeSegment(start=0.0, end=8.0, description=fallback_desc)]

    timeline_text = "；".join(
        f"{format_number(segment.start)}-{format_number(segment.end)}s {segment.description}"
        for segment in segments
    )
    duration = max(segment.end for segment in segments)
    prefix = f"{format_number(duration)}s自定义居家安全视频"
    solution_parts = [
        f"{label}{solution_sections[key]}"
        for key, label in SOLUTION_SECTION_LABELS
        if solution_sections.get(key)
    ]
    solution_suffix = f"。{'。'.join(solution_parts)}" if solution_parts else ""
    return (
        f"{record.video_id}|{prefix}。地点：{location or '居家场景'}。对象：{subject or '家庭成员'}。"
        f"场景：{scenario or '居家安全事件'}。时间轴：{timeline_text}{solution_suffix}。"
    )


def write_text(path: Path, content: str, overwrite: bool) -> None:
    ensure_parent(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(content, encoding="utf-8")


def default_paths(args: argparse.Namespace) -> tuple[Path | None, Path, Path, Path]:
    extract_dir = args.extract_dir or (args.dataset_root / "extracted")
    flat_videos_dir = args.flat_videos_dir or (args.dataset_root / "videos_flat")
    ground_truth_out = args.ground_truth_out or (args.dataset_root / "gen_prompt_custom.txt")
    summary_json = args.summary_json or (args.dataset_root / "prepared_summary.json")
    return extract_dir, flat_videos_dir, ground_truth_out, summary_json


def main() -> int:
    args = build_parser().parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.prompt_file = args.prompt_file.resolve()
    if args.zip_path is not None:
        args.zip_path = args.zip_path.resolve()
    if args.extract_dir is not None:
        args.extract_dir = args.extract_dir.resolve()
    if args.flat_videos_dir is not None:
        args.flat_videos_dir = args.flat_videos_dir.resolve()
    if args.ground_truth_out is not None:
        args.ground_truth_out = args.ground_truth_out.resolve()
    if args.summary_json is not None:
        args.summary_json = args.summary_json.resolve()

    extract_dir, flat_videos_dir, ground_truth_out, summary_json = default_paths(args)

    if args.zip_path:
        extract_zip(args.zip_path, extract_dir, overwrite=args.overwrite)

    records = parse_prompt_records(args.prompt_file)
    records_by_id = {record.video_id: record for record in records}
    videos_by_id = scan_videos(args.dataset_root, extract_dir if extract_dir.exists() else None)

    prompt_ids = set(records_by_id)
    video_ids = set(videos_by_id)
    intersection_ids = sorted(prompt_ids & video_ids)
    prompt_only_ids = sorted(prompt_ids - video_ids)
    video_only_ids = sorted(video_ids - prompt_ids)

    flat_videos_dir.mkdir(parents=True, exist_ok=True)
    flattened_ids = set(intersection_ids)
    if args.include_unmatched_videos:
        flattened_ids |= video_ids

    for video_id in sorted(flattened_ids):
        source = videos_by_id[video_id]
        target = flat_videos_dir / f"{video_id}{source.suffix.lower()}"
        materialize_video(source, target, link_mode=args.link_mode, overwrite=args.overwrite)

    ground_truth_lines = [to_ground_truth_line(records_by_id[video_id]) for video_id in intersection_ids]
    write_text(ground_truth_out, "\n".join(ground_truth_lines) + ("\n" if ground_truth_lines else ""), overwrite=args.overwrite)

    summary = {
        "dataset_root": str(args.dataset_root),
        "prompt_file": str(args.prompt_file),
        "zip_path": str(args.zip_path) if args.zip_path else None,
        "extract_dir": str(extract_dir) if extract_dir.exists() else None,
        "flat_videos_dir": str(flat_videos_dir),
        "ground_truth_out": str(ground_truth_out),
        "record_count": len(records),
        "records_with_solution_count": sum(1 for record in records if any(parse_solution_sections(record.solution_text).values())),
        "prompt_id_count": len(prompt_ids),
        "video_count": len(video_ids),
        "matched_count": len(intersection_ids),
        "prompt_only_count": len(prompt_only_ids),
        "video_only_count": len(video_only_ids),
        "prompt_only_ids": prompt_only_ids,
        "video_only_ids": video_only_ids,
        "matched_ids": intersection_ids,
        "link_mode": args.link_mode,
        "include_unmatched_videos": bool(args.include_unmatched_videos),
    }
    write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2) + "\n", overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

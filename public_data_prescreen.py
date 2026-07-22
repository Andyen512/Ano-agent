#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
BATCH_INFER_SCRIPT = PROJECT_ROOT / "evaluation" / "lifebench_batch_infer.py"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "public_data"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "data" / "public_data_prescreen_prompt.txt"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen" / "Qwen3.5-9B"
DEFAULT_TARSIER_CONFIG = PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
STATUS_PRIORITY = {"abnormal": 0, "risk_only": 1, "normal": 2}


def now_utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-9B batch prescreening over real home videos under data/public_data."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing public real-home videos.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset subdirectories under --data-root, e.g. UR_Fall_Dataset SmartHome-Bench.",
    )
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="Prompt file used for prescreening.")
    parser.add_argument("--output-root", type=Path, help="Output directory. Defaults to outputs/public_data_prescreen/<run-id>.")
    parser.add_argument("--run-id", default=now_utc_tag(), help="Run id used when --output-root is omitted.")
    parser.add_argument("--backend", default="qwen35vl", help="Backend forwarded to lifebench_batch_infer.py.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B", help="Model id forwarded to lifebench_batch_infer.py.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local model path.")
    parser.add_argument("--device-map", default="auto", help='device_map forwarded to the backend loader. Default: "auto".')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--tarsier-config", type=Path, default=DEFAULT_TARSIER_CONFIG)
    parser.add_argument("--projection-path", type=Path, help="Optional projection path for backends such as Video-ChatGPT.")
    parser.add_argument("--limit", type=int, help="Only process the first N videos after sorting.")
    parser.add_argument("--resume", action="store_true", help="Skip videos whose output JSON already exists and is valid.")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare manifests and reports without launching inference.")
    return parser


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    return (PROJECT_ROOT / "outputs" / "public_data_prescreen" / args.run_id).resolve()


def selected_dataset_roots(data_root: Path, datasets: list[str] | None) -> list[Path]:
    if not datasets:
        return [data_root]
    roots: list[Path] = []
    for item in datasets:
        candidate = (data_root / item).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {candidate}")
        roots.append(candidate)
    return roots


def scan_videos(data_root: Path, datasets: list[str] | None) -> list[Path]:
    roots = selected_dataset_roots(data_root, datasets)
    videos: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path.resolve())
    return sorted(dict.fromkeys(videos))


def valid_output_json(path: Path, expected_video_path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("response"), str):
        return False
    recorded_video = payload.get("video_path")
    if not isinstance(recorded_video, str):
        return False
    return Path(recorded_video).expanduser().resolve() == expected_video_path.resolve()


def make_task_record(video_path: Path, data_root: Path, output_root: Path) -> dict[str, str]:
    relative_path = video_path.relative_to(data_root)
    prediction_path = output_root / "predictions" / relative_path.parent / f"{relative_path.name}.json"
    stdout_log = output_root / "logs" / "stdout" / relative_path.parent / f"{relative_path.name}.log"
    stderr_log = output_root / "logs" / "stderr" / relative_path.parent / f"{relative_path.name}.log"
    return {
        "video_path": str(video_path),
        "output_json": str(prediction_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_run_command(args: argparse.Namespace, manifest_path: Path, batch_summary_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(BATCH_INFER_SCRIPT),
        "--backend",
        args.backend,
        "--model-id",
        args.model_id,
        "--task-manifest",
        str(manifest_path),
        "--summary-json",
        str(batch_summary_path),
        "--prompt-file",
        str(args.prompt_file.resolve()),
        "--device-map",
        args.device_map,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--fps",
        str(args.fps),
        "--max-frames",
        str(args.max_frames),
        "--merge-size",
        str(args.merge_size),
        "--attn-implementation",
        args.attn_implementation,
        "--tarsier-config",
        str(args.tarsier_config.resolve()),
    ]
    if args.model_path is not None:
        command.extend(["--model-path", str(args.model_path.resolve())])
    if args.use_flash_attn:
        command.append("--use-flash-attn")
    if args.projection_path is not None:
        command.extend(["--projection-path", str(args.projection_path.resolve())])
    return command


def parse_prediction(output_json: Path) -> dict[str, Any]:
    if not output_json.exists():
        return {}
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    parsed = payload.get("parsed_response")
    return parsed if isinstance(parsed, dict) else {}


def collect_batch_records(batch_summary_path: Path) -> dict[str, dict[str, Any]]:
    if not batch_summary_path.exists():
        return {}
    try:
        payload = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    by_video: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("video"), str):
            by_video[record["video"]] = record
    return by_video


def build_screening_rows(
    videos: list[Path],
    data_root: Path,
    output_root: Path,
    batch_record_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_path in videos:
        relative_path = video_path.relative_to(data_root)
        output_json = output_root / "predictions" / relative_path.parent / f"{relative_path.name}.json"
        batch_record = batch_record_map.get(str(video_path), {})
        prediction_payload: dict[str, Any] = {}
        response = ""
        if output_json.exists():
            try:
                raw_payload = json.loads(output_json.read_text(encoding="utf-8"))
                if isinstance(raw_payload, dict):
                    prediction_payload = raw_payload
                    response = str(raw_payload.get("response", "")).strip()
            except (OSError, json.JSONDecodeError):
                prediction_payload = {}
        parsed = prediction_payload.get("parsed_response")
        parsed_response = parsed if isinstance(parsed, dict) else {}
        status = str(parsed_response.get("risk_status", "")).strip()
        risk_type = str(parsed_response.get("risk_type", "")).strip()
        if "returncode" in batch_record:
            returncode: int | None = int(batch_record["returncode"])
        elif output_json.exists():
            returncode = 0
        else:
            returncode = None
        rows.append(
            {
                "dataset": relative_path.parts[0] if relative_path.parts else "",
                "video_relpath": str(relative_path),
                "video_path": str(video_path),
                "output_json": str(output_json),
                "stdout_log": str(output_root / "logs" / "stdout" / relative_path.parent / f"{relative_path.name}.log"),
                "stderr_log": str(output_root / "logs" / "stderr" / relative_path.parent / f"{relative_path.name}.log"),
                "returncode": returncode,
                "risk_status": status,
                "risk_type": risk_type,
                "risk_description": str(parsed_response.get("risk_description", "")).strip(),
                "video_description": str(parsed_response.get("video_description", "")).strip(),
                "time_spans": parsed_response.get("time_spans", []),
                "solution": str(parsed_response.get("solution", "")).strip(),
                "response": response,
                "stderr_tail": str(batch_record.get("stderr_tail", "")).strip(),
            }
        )
    rows.sort(key=lambda item: (STATUS_PRIORITY.get(item["risk_status"], 99), item["dataset"], item["video_relpath"]))
    return rows


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "video_relpath",
                "video_path",
                "returncode",
                "risk_status",
                "risk_type",
                "risk_description",
                "video_description",
                "time_spans",
                "solution",
                "output_json",
                "stdout_log",
                "stderr_log",
                "stderr_tail",
                "response",
            ],
        )
        writer.writeheader()
        for row in rows:
            serializable = row.copy()
            serializable["time_spans"] = json.dumps(serializable["time_spans"], ensure_ascii=False)
            writer.writerow(serializable)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"abnormal": 0, "risk_only": 0, "normal": 0, "unknown": 0, "failed": 0, "pending": 0}
    for row in rows:
        if row["returncode"] is None:
            counts["pending"] += 1
        elif row["returncode"] != 0:
            counts["failed"] += 1
        status = row["risk_status"]
        if status in counts:
            counts[status] += 1
        else:
            counts["unknown"] += 1
    risky_rows = [row for row in rows if row["risk_status"] in {"abnormal", "risk_only"}]
    return {
        "video_count": len(rows),
        "counts": counts,
        "risky_video_count": len(risky_rows),
        "risky_videos": risky_rows,
        "all_videos": rows,
    }


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    if not args.prompt_file.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {args.prompt_file}")

    output_root = resolve_output_root(args)
    manifest_path = output_root / "manifests" / "tasks.json"
    batch_summary_path = output_root / "batch_summary.json"
    config_path = output_root / "run_config.json"
    report_json_path = output_root / "screening_report.json"
    report_csv_path = output_root / "screening_report.csv"

    videos = scan_videos(data_root, args.datasets)
    if args.limit is not None:
        videos = videos[: max(0, args.limit)]
    if not videos:
        raise RuntimeError(f"No videos found under {data_root}")

    task_records: list[dict[str, str]] = []
    skipped_existing: list[str] = []
    for video_path in videos:
        task = make_task_record(video_path, data_root, output_root)
        if args.resume and valid_output_json(Path(task["output_json"]), video_path):
            skipped_existing.append(str(video_path))
            continue
        task_records.append(task)

    write_json(
        config_path,
        {
            "run_id": args.run_id,
            "data_root": str(data_root),
            "datasets": args.datasets or [],
            "output_root": str(output_root),
            "backend": args.backend,
            "model_id": args.model_id,
            "model_path": str(args.model_path.resolve()) if args.model_path is not None else None,
            "prompt_file": str(args.prompt_file.resolve()),
            "video_count": len(videos),
            "task_count": len(task_records),
            "skipped_existing_count": len(skipped_existing),
            "dry_run": bool(args.dry_run),
        },
    )
    write_json(manifest_path, task_records)

    if not args.dry_run and task_records:
        command = build_run_command(args, manifest_path, batch_summary_path)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            print(f"Batch inference exited with code {completed.returncode}. Reports will include failures.", file=sys.stderr)

    batch_record_map = collect_batch_records(batch_summary_path)
    rows = build_screening_rows(videos, data_root, output_root, batch_record_map)
    summary = summarize_rows(rows)
    report_payload = {
        "run_id": args.run_id,
        "data_root": str(data_root),
        "output_root": str(output_root),
        "backend": args.backend,
        "model_id": args.model_id,
        "prompt_file": str(args.prompt_file.resolve()),
        "prepared_task_count": len(task_records),
        "skipped_existing_count": len(skipped_existing),
        **summary,
    }
    write_json(report_json_path, report_payload)
    write_report_csv(report_csv_path, rows)

    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

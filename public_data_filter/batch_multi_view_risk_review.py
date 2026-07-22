#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PUBLIC_DATA_ROOT = PROJECT_ROOT / "data" / "public_data"
REVIEW_SCRIPT = SCRIPT_DIR / "multi_view_risk_review.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_outputs"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ENV_ROOT_CANDIDATES = (
    Path("/workspace/anaconda3/envs/lifebench-vlm"),
    Path("/data_4/liuyuan/anaconda3/envs/lifebench-vlm"),
)


@dataclass(frozen=True)
class ReviewTask:
    index: int
    total: int
    video_path: Path
    relative_path: Path
    output_root: Path


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-run the five-model public_data_filter review over all videos under data/public_data."
    )
    parser.add_argument("--data-root", type=Path, default=PUBLIC_DATA_ROOT, help="Root directory containing videos.")
    parser.add_argument("--datasets", nargs="*", help="Optional dataset subdirectories under --data-root.")
    parser.add_argument("--output-root", type=Path, help="Batch output directory. Defaults to public_data_filter/batch_outputs/<run-id>.")
    parser.add_argument("--run-id", default=utc_tag(), help="Run id used when --output-root is omitted.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids. One worker is created per GPU.")
    parser.add_argument("--seed", type=int, default=0, help="Fixed seed forwarded to multi_view_risk_review.py.")
    parser.add_argument("--candidate-text", help="Optional candidate behavior text forwarded to each single-video review.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--python-executable", type=Path, help="Python executable used to run multi_view_risk_review.py.")
    parser.add_argument("--limit", type=int, help="Only process the first N videos after sorting.")
    parser.add_argument("--resume", action="store_true", help="Skip videos whose review_summary.json already exists and is valid.")
    parser.add_argument(
        "--reuse-agent-outputs",
        action="store_true",
        help="Forward --reuse-agent-outputs so interrupted per-video reviews resume from completed agents.",
    )
    parser.add_argument(
        "--early-stop-majority",
        action="store_true",
        help="Forward --early-stop-majority so each video stops once a 3-of-5 majority is determined.",
    )
    parser.add_argument("--keep-going", action="store_true", help="Continue the batch even if some videos fail.")
    parser.add_argument("--log-every", type=int, default=1, help="Print progress every N finished videos. Default: 1.")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare manifests and counts without launching reviews.")
    return parser


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    return (DEFAULT_OUTPUT_ROOT / args.run_id).resolve()


def selected_roots(data_root: Path, datasets: list[str] | None) -> list[Path]:
    if not datasets:
        return [data_root]
    roots: list[Path] = []
    for dataset in datasets:
        candidate = (data_root / dataset).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {candidate}")
        roots.append(candidate)
    return roots


def scan_videos(data_root: Path, datasets: list[str] | None) -> list[Path]:
    videos: list[Path] = []
    for root in selected_roots(data_root, datasets):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path.resolve())
    return sorted(dict.fromkeys(videos))


def parse_gpu_ids(raw: str) -> list[str]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not ids:
        raise ValueError("At least one GPU id must be provided via --gpus.")
    return ids


def discover_env_root() -> Path | None:
    raw_env_root = os.environ.get("ENV_ROOT")
    if raw_env_root:
        candidate = Path(raw_env_root).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    for candidate in ENV_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_python_executable(args: argparse.Namespace) -> Path:
    if args.python_executable is not None:
        return args.python_executable.expanduser().resolve()
    env_root = discover_env_root()
    if env_root is not None:
        candidate = env_root / "bin" / "python"
        if candidate.exists():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def valid_review_summary(path: Path, expected_video_path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        return False
    if manifest.get("video_path") != str(expected_video_path):
        return False
    if manifest.get("prompt_language") != "en":
        return False
    if manifest.get("output_schema_version") != "risk_full_en_v2":
        return False
    majority_vote = payload.get("majority_vote")
    if not isinstance(majority_vote, dict) or majority_vote.get("complete") is not True:
        return False
    agent_results = payload.get("agent_results")
    if not isinstance(agent_results, list) or len(agent_results) < 5:
        return False
    return all(item.get("returncode") == 0 for item in agent_results if isinstance(item, dict))


def task_output_root(batch_output_root: Path, data_root: Path, video_path: Path) -> Path:
    relative = video_path.relative_to(data_root)
    return (batch_output_root / "reviews" / relative.parent / relative.stem).resolve()


def build_tasks(args: argparse.Namespace, data_root: Path, output_root: Path) -> tuple[list[ReviewTask], list[str]]:
    videos = scan_videos(data_root, args.datasets)
    if args.limit is not None:
        videos = videos[: max(0, args.limit)]
    tasks: list[ReviewTask] = []
    skipped: list[str] = []
    total = len(videos)
    for index, video_path in enumerate(videos, start=1):
        review_root = task_output_root(output_root, data_root, video_path)
        summary_path = review_root / "review_summary.json"
        if args.resume and valid_review_summary(summary_path, video_path):
            skipped.append(str(video_path))
            continue
        tasks.append(
            ReviewTask(
                index=index,
                total=total,
                video_path=video_path,
                relative_path=video_path.relative_to(data_root),
                output_root=review_root,
            )
        )
    return tasks, skipped


def build_command(args: argparse.Namespace, task: ReviewTask) -> list[str]:
    command = [
        str(resolve_python_executable(args)),
        str(REVIEW_SCRIPT),
        "--video-path",
        str(task.video_path),
        "--output-root",
        str(task.output_root),
        "--seed",
        str(args.seed),
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
        "--device-map",
        "cuda:0",
    ]
    if args.candidate_text:
        command.extend(["--candidate-text", args.candidate_text])
    if args.use_flash_attn:
        command.append("--use-flash-attn")
    if args.reuse_agent_outputs:
        command.append("--reuse-agent-outputs")
    if args.early_stop_majority:
        command.append("--early-stop-majority")
    if args.keep_going:
        command.append("--keep-going")
    if args.python_executable:
        command.extend(["--python-executable", str(resolve_python_executable(args))])
    return command


def load_review_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_task(args: argparse.Namespace, task: ReviewTask, gpu_id: str) -> dict[str, Any]:
    task.output_root.mkdir(parents=True, exist_ok=True)
    stdout_log = task.output_root / "batch_stdout.log"
    stderr_log = task.output_root / "batch_stderr.log"
    command = build_command(args, task)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.time() - started_at
    stdout_log.write_text(completed.stdout, encoding="utf-8")
    stderr_log.write_text(completed.stderr, encoding="utf-8")
    summary_path = task.output_root / "review_summary.json"

    result: dict[str, Any] = {
        "video_path": str(task.video_path),
        "video_relpath": str(task.relative_path),
        "output_root": str(task.output_root),
        "summary_json": str(summary_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "gpu_id": gpu_id,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "command": command,
    }
    if completed.returncode == 0 and summary_path.exists():
        payload = load_review_summary(summary_path)
        majority_vote = payload.get("majority_vote", {})
        result["majority_vote"] = majority_vote
        result["keep"] = majority_vote.get("keep")
        result["complete"] = majority_vote.get("complete")
    else:
        result["keep"] = None
        result["complete"] = False
        result["stderr_tail"] = completed.stderr[-2000:]
    return result


def maybe_print_progress(index: int, total: int, result: dict[str, Any], log_every: int) -> None:
    if log_every <= 0:
        log_every = 1
    if index % log_every != 0 and index != total:
        return
    keep = result.get("keep")
    status = "keep" if keep is True else "drop" if keep is False else "unknown"
    print(
        f"[{index}/{total}] gpu={result['gpu_id']} rc={result['returncode']} status={status} {result['video_relpath']}",
        file=sys.stderr,
        flush=True,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_relpath",
                "video_path",
                "gpu_id",
                "returncode",
                "complete",
                "keep",
                "risk_presence",
                "level1_scene_top",
                "level2_subject_top",
                "level3_risk_type_top",
                "normal_level1_scene_top",
                "normal_level2_subject_top",
                "elapsed_seconds",
                "summary_json",
                "stdout_log",
                "stderr_log",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_relpath": row["video_relpath"],
                    "video_path": row["video_path"],
                    "gpu_id": row["gpu_id"],
                    "returncode": row["returncode"],
                    "complete": row.get("complete"),
                    "keep": row.get("keep"),
                    "risk_presence": row.get("majority_vote", {}).get("risk_presence"),
                    "level1_scene_top": row.get("majority_vote", {}).get("level1_scene_top"),
                    "level2_subject_top": row.get("majority_vote", {}).get("level2_subject_top"),
                    "level3_risk_type_top": row.get("majority_vote", {}).get("level3_risk_type_top"),
                    "normal_level1_scene_top": row.get("majority_vote", {}).get("normal_level1_scene_top"),
                    "normal_level2_subject_top": row.get("majority_vote", {}).get("normal_level2_subject_top"),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "summary_json": row["summary_json"],
                    "stdout_log": row["stdout_log"],
                    "stderr_log": row["stderr_log"],
                }
            )


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in results:
        if row["returncode"] != 0:
            counts["failed"] += 1
        elif row.get("keep") is True:
            counts["keep"] += 1
        elif row.get("keep") is False:
            counts["drop"] += 1
        else:
            counts["unknown"] += 1
    return {
        "processed_videos": len(results),
        "counts": dict(counts),
    }


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    if not REVIEW_SCRIPT.exists():
        raise FileNotFoundError(f"Review script does not exist: {REVIEW_SCRIPT}")

    gpu_ids = parse_gpu_ids(args.gpus)
    output_root = resolve_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks, skipped = build_tasks(args, data_root, output_root)
    manifest = {
        "run_id": args.run_id,
        "data_root": str(data_root),
        "datasets": args.datasets or [],
        "output_root": str(output_root),
        "prompt_language": "en",
        "output_schema_version": "risk_full_en_v2",
        "gpu_ids": gpu_ids,
        "seed": args.seed,
        "reuse_agent_outputs": bool(args.reuse_agent_outputs),
        "early_stop_majority": bool(args.early_stop_majority),
        "task_count": len(tasks),
        "skipped_existing_count": len(skipped),
        "dry_run": bool(args.dry_run),
    }
    write_json(output_root / "batch_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    task_queues: dict[str, list[ReviewTask]] = {gpu_id: [] for gpu_id in gpu_ids}
    for index, task in enumerate(tasks):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        task_queues[gpu_id].append(task)

    def worker(gpu_id: str, gpu_tasks: list[ReviewTask]) -> list[dict[str, Any]]:
        worker_results: list[dict[str, Any]] = []
        for task in gpu_tasks:
            result = run_task(args, task, gpu_id)
            worker_results.append(result)
            with lock:
                results.append(result)
                maybe_print_progress(len(results), len(tasks), result, args.log_every)
            if result["returncode"] != 0 and not args.keep_going:
                break
        return worker_results

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, gpu_id, task_queues[gpu_id]) for gpu_id in gpu_ids]
        for future in as_completed(futures):
            future.result()

    results.sort(key=lambda item: item["video_relpath"])
    summary = {
        "manifest": manifest,
        "summary": summarize_results(results),
        "skipped_existing": skipped,
        "results": results,
    }
    write_json(output_root / "batch_review_summary.json", summary)
    write_csv(output_root / "batch_review_summary.csv", results)
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

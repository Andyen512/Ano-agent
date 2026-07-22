#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from dataclasses import dataclass
import os
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "public_data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "public_data_image_as_video"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class ConversionTask:
    sequence_name: str
    frame_paths: tuple[Path, ...]
    output_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert image-only frame sequences under data/public_data into mp4 videos. "
            "Frames are grouped by the shared filename prefix before the final numeric index."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root directory to scan.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where generated mp4 files will be stored.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write generated mp4 files directly next to the source frame sequences under --input-root.",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frame rate for generated videos.")
    parser.add_argument("--datasets", nargs="*", help="Optional dataset subdirectories under --input-root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be generated.")
    parser.add_argument("--limit", type=int, help="Only process the first N discovered image samples.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N processed sequences. Default: 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for video writing. Default: 1.",
    )
    return parser


def selected_roots(input_root: Path, datasets: list[str] | None) -> list[Path]:
    if not datasets:
        return [input_root]
    roots: list[Path] = []
    for dataset in datasets:
        path = (input_root / dataset).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {path}")
        roots.append(path)
    return roots


def has_video_sibling(path: Path) -> bool:
    for ext in VIDEO_EXTENSIONS:
        if path.with_suffix(ext).exists():
            return True
    return False


def split_sequence_frame(name: str) -> tuple[str, int] | None:
    match = re.match(r"^(?P<prefix>.+)_(?P<index>\d+)$", name)
    if not match:
        return None
    return match.group("prefix"), int(match.group("index"))


def collect_tasks(
    input_root: Path,
    output_root: Path,
    datasets: list[str] | None,
    in_place: bool,
) -> list[ConversionTask]:
    tasks: list[ConversionTask] = []
    for root in selected_roots(input_root, datasets):
        for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
            grouped_frames: dict[str, list[tuple[int, Path]]] = {}
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                parsed = split_sequence_frame(path.stem)
                if parsed is None:
                    continue
                sequence_name, frame_index = parsed
                if has_video_sibling(path.with_name(sequence_name + path.suffix)):
                    continue
                grouped_frames.setdefault(sequence_name, []).append((frame_index, path.resolve()))
            for sequence_name, indexed_paths in sorted(grouped_frames.items()):
                indexed_paths.sort(key=lambda item: item[0])
                if in_place:
                    output_path = (directory.resolve() / f"{sequence_name}.mp4").resolve()
                else:
                    relative_dir = directory.resolve().relative_to(input_root.resolve())
                    output_path = (output_root / relative_dir / f"{sequence_name}.mp4").resolve()
                tasks.append(
                    ConversionTask(
                        sequence_name=sequence_name,
                        frame_paths=tuple(path for _, path in indexed_paths),
                        output_path=output_path,
                    )
                )
    return tasks


def image_to_bgr_frame(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    frame = np.array(image)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def write_sequence_video(frame_paths: tuple[Path, ...], output_path: Path, fps: float) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not frame_paths:
        raise ValueError(f"No frames provided for {output_path}")

    first_frame = image_to_bgr_frame(frame_paths[0])
    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        writer.write(first_frame)
        for frame_path in frame_paths[1:]:
            frame = image_to_bgr_frame(frame_path)
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def maybe_print_progress(
    index: int,
    total: int,
    task: ConversionTask,
    action: str,
    log_every: int,
) -> None:
    if log_every <= 0:
        log_every = 1
    if index % log_every != 0 and index != total:
        return
    print(
        f"[{index}/{total}] {action}: {task.output_path} "
        f"(sequence={task.sequence_name}, frames={len(task.frame_paths)})",
        file=sys.stderr,
        flush=True,
    )


def process_task(task: ConversionTask, fps: float, dry_run: bool) -> dict[str, str | int]:
    if not dry_run:
        write_sequence_video(task.frame_paths, task.output_path, fps=fps)
    return {
        "sequence_name": task.sequence_name,
        "first_frame": str(task.frame_paths[0]),
        "frame_count": len(task.frame_paths),
        "output_path": str(task.output_path),
    }


def main() -> int:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = input_root if args.in_place else args.output_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    tasks = collect_tasks(input_root, output_root, args.datasets, args.in_place)
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    skipped_existing = 0
    processed: list[dict[str, str]] = []
    pending_tasks: list[ConversionTask] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        if task.output_path.exists() and not args.overwrite:
            skipped_existing += 1
            maybe_print_progress(index, total, task, "skip-existing", args.log_every)
            continue
        pending_tasks.append(task)

    created = 0
    if args.workers == 1:
        for index, task in enumerate(pending_tasks, start=1):
            action = "dry-run" if args.dry_run else "writing"
            maybe_print_progress(index, len(pending_tasks), task, action, args.log_every)
            result = process_task(task, fps=args.fps, dry_run=args.dry_run)
            processed.append(
                {
                    "sequence_name": str(result["sequence_name"]),
                    "first_frame": str(result["first_frame"]),
                    "frame_count": str(result["frame_count"]),
                    "output_path": str(result["output_path"]),
                }
            )
            if not args.dry_run:
                created += 1
    else:
        action = "dry-run" if args.dry_run else "writing"
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_task, task, args.fps, args.dry_run): task for task in pending_tasks
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                maybe_print_progress(index, len(pending_tasks), task, action, args.log_every)
                result = future.result()
                processed.append(
                    {
                        "sequence_name": str(result["sequence_name"]),
                        "first_frame": str(result["first_frame"]),
                        "frame_count": str(result["frame_count"]),
                        "output_path": str(result["output_path"]),
                    }
                )
                if not args.dry_run:
                    created += 1

    summary = {
        "input_root": str(input_root),
        "output_root": None if args.in_place else str(output_root),
        "in_place": bool(args.in_place),
        "dataset_filter": args.datasets or [],
        "fps": args.fps,
        "workers": args.workers,
        "discovered_image_only_sequences": len(tasks),
        "pending_sequences": len(pending_tasks),
        "created_videos": created,
        "skipped_existing": skipped_existing,
        "dry_run": bool(args.dry_run),
        "examples": processed[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

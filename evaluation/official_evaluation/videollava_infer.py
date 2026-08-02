#!/usr/bin/env python3
"""Run Video-LLaVA inference for the official real-video evaluation set."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "real_videos"
DEFAULT_PROMPT = PROJECT_ROOT / "evaluation" / "scripts" / "prompts" / "infer_prompt_structured.txt"
DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "prediction" / "real_videos"
MODEL_ID = "LanguageBind/Video-LLaVA-7B"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer Video-LLaVA on LifeBench official real videos.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--video", action="append", help="Run only this video path; may be repeated.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def build_tasks(args: argparse.Namespace, run_dir: Path, logs_dir: Path) -> list[dict[str, str]]:
    if args.video:
        videos = [Path(item).expanduser().resolve() for item in args.video]
    else:
        videos = sorted(
            path.resolve()
            for path in args.video_root.expanduser().rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        )
    videos = [path for path in videos if path.is_file()]
    if args.limit is not None:
        videos = videos[: max(0, args.limit)]
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.video_root}")

    tasks = []
    for video in videos:
        tasks.append(
            {
                "video_path": str(video),
                "output_json": str(run_dir / f"{video.stem}.json"),
                "stdout_log": str(logs_dir / f"{video.stem}.stdout.log"),
                "stderr_log": str(logs_dir / f"{video.stem}.stderr.log"),
            }
        )
    return tasks


def main() -> int:
    args = build_parser().parse_args()
    from evaluation.lifebench_batch_infer import (  # noqa: PLC0415
        create_session_bundle,
        run_one_task,
        validate_task_manifest,
        write_json,
    )
    from evaluation.lifebench_infer import load_prompt  # noqa: PLC0415

    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Video-LLaVA model path does not exist: {args.model_path}")
    prompt = load_prompt(None, str(args.prompt_file))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.predictions_root / "video-llava-7b" / run_id
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    task_manifest = build_tasks(args, run_dir, logs_dir)
    bundle_args = argparse.Namespace(
        backend="videollava",
        model_id=MODEL_ID,
        model_path=str(args.model_path),
        device_map="auto",
        temperature=args.temperature,
        top_p=0.9,
        max_new_tokens=args.max_new_tokens,
        fps=1.0,
        max_frames=args.max_frames,
        merge_size=2,
        attn_implementation="auto",
        use_flash_attn=False,
        tarsier_config=str(PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"),
        projection_path=None,
        skip_existing=args.skip_existing,
    )
    bundle = create_session_bundle(bundle_args)

    records = []
    for task in validate_task_manifest(task_manifest):
        record = run_one_task(task, bundle_args, prompt, bundle)
        records.append(record)
        print(f"[{len(records)}/{len(task_manifest)}] {Path(task['video_path']).name}: {record['returncode']}", flush=True)

    summary = {
        "backend": "videollava",
        "model_id": MODEL_ID,
        "run_id": run_id,
        "records": records,
        "failed_count": sum(1 for item in records if item["returncode"] != 0),
        "updated_at": now_iso(),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "task_manifest.json", {"tasks": task_manifest})
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

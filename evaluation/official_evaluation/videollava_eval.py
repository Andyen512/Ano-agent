#!/usr/bin/env python3
"""Evaluate Video-LLaVA prediction files with the official real-video evaluator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.official_evaluation.real_videos.evaluate import (  # noqa: E402
    DEFAULT_GT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICTIONS_ROOT,
    main as evaluate_main,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Video-LLaVA on LifeBench official real videos.")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "video-llava-7b")
    parser.add_argument("--judge-mode", choices=("auto", "heuristic", "openai"), default="auto")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-base-urls", nargs="*")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    forwarded = [
        "--gt-dir", str(args.gt_dir),
        "--predictions-root", str(args.predictions_root),
        "--output-dir", str(args.output_dir),
        "--model-id", "LanguageBind/Video-LLaVA-7B",
        "--judge-mode", args.judge_mode,
        "--workers", str(args.workers),
    ]
    for option, value in (
        ("--judge-model", args.judge_model),
        ("--judge-api-key", args.judge_api_key),
        ("--judge-base-url", args.judge_base_url),
    ):
        if value:
            forwarded.extend([option, value])
    if args.judge_base_urls:
        forwarded.append("--judge-base-urls")
        forwarded.extend(args.judge_base_urls)
    if args.resume:
        forwarded.append("--resume")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *forwarded]
        return evaluate_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())

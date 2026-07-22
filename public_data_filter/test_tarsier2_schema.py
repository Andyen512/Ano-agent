#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multi_view_risk_review import build_prompt, build_subprocess_env, parse_keep_decision, parse_rule_file
from persistent_batch_multi_view_risk_review import RULE_PATH, Tarsier2Engine, build_format_retry_prompt


DEFAULT_VIDEO = PROJECT_ROOT / "data" / "public_data" / "Multiple_users_compound_event" / "data" / "p101" / "Cook_Sequence_Afternoon" / "00101_c5s0.mp4"
DEFAULT_MODEL_ID = "omni-research/Tarsier2-7b-0115"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "public_data_filter" / "tarsier2_schema_tests"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load Tarsier2 once and test schema-following prompts on one video.")
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--perspective-index", type=int, default=2, help="1-based rule perspective index. Default: 2 Conservative Safety.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument(
        "--tarsier-config",
        default=str(PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"),
    )
    return parser


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_one(engine: Tarsier2Engine, args: argparse.Namespace, prompt: str, label: str, output_root: Path) -> dict[str, Any]:
    infer_args = SimpleNamespace(
        video_path=args.video_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        fps=args.fps,
        max_frames=args.max_frames,
        merge_size=args.merge_size,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        use_flash_attn=args.use_flash_attn,
        tarsier_config=args.tarsier_config,
        output_json=str(output_root / f"{label}.json"),
        model_id=DEFAULT_MODEL_ID,
    )
    started_at = time.time()
    response = engine.infer(infer_args, prompt)
    elapsed = time.time() - started_at
    parsed = parse_keep_decision(response)
    payload = {
        "label": label,
        "elapsed_seconds": round(elapsed, 3),
        "parse_ok": parsed.get("keep") is not None,
        "parsed_decision": parsed,
        "response": response,
        "prompt": prompt,
    }
    write_json(output_root / f"{label}.json", payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    args.video_path = args.video_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {args.video_path}")

    os.environ.update(build_subprocess_env())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(EVALUATION_DIR))
    from lifebench_infer import resolve_model_path

    rules = parse_rule_file(RULE_PATH)
    rule = rules[args.perspective_index - 1]
    original_prompt = build_prompt(rule, candidate_text=None)
    retry_prompt = build_format_retry_prompt(original_prompt)

    model_path = resolve_model_path(None, DEFAULT_MODEL_ID)
    engine = Tarsier2Engine(model_path, args)

    original = run_one(engine, args, original_prompt, "original_prompt", output_root)
    retry = run_one(engine, args, retry_prompt, "format_retry_prompt", output_root)
    summary = {
        "video_path": str(args.video_path),
        "gpu": args.gpu,
        "perspective": {"index": rule.index, "name": rule.name},
        "original_parse_ok": original["parse_ok"],
        "retry_parse_ok": retry["parse_ok"],
        "outputs": {
            "original": str(output_root / "original_prompt.json"),
            "retry": str(output_root / "format_retry_prompt.json"),
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

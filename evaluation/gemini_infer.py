#!/usr/bin/env python3
"""Run LifeBench video inference through Gemini via lingganyaapi OpenAI-compatible gateway."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))


DEFAULT_API_KEY = "sk-Gpl3BxJh16AjZ2jQ1s1CT9di0EDG5Wg8RrQBevC26aNhSHzh"
DEFAULT_API_URL = "https://www.lingganyaapi.com/v1"
DEFAULT_MODEL = "gemini-3.5-flash"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a LifeBench video through Gemini using an OpenAI-compatible API."
    )
    parser.add_argument("--video-path", required=True, help="Path to the input video.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Text prompt sent with the video frames.")
    prompt_group.add_argument("--prompt-file", help="Read the prompt from a UTF-8 text file.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-url",
        default=os.environ.get("GEMINI_API_URL", DEFAULT_API_URL),
        help="OpenAI-compatible base URL. Defaults to GEMINI_API_URL or lingganyaapi.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", DEFAULT_API_KEY),
        help="API key. Defaults to GEMINI_API_KEY.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of uniformly sampled video frames. Default: 32.",
    )
    parser.add_argument("--output-json", help="Optional path for the structured inference result.")
    parser.add_argument("--print-json", action="store_true", help="Print the full result as JSON.")
    return parser


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt.strip()
    else:
        prompt = "Describe the video in detail."
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    return prompt


def build_messages(video_path: Path, prompt: str, max_frames: int) -> list[dict[str, Any]]:
    from lifebench_infer import load_video_frames_simple
    from llm import image_to_data_url

    frames = load_video_frames_simple(str(video_path), max_num_frames=max_frames)
    if not frames:
        raise RuntimeError(f"Unable to decode video frames: {video_path}")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The following images are ordered chronological samples from one video. "
                "Use them to infer the full video and answer the task below."
            ),
        }
    ]
    for index, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"Frame {index}/{len(frames)}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(frame, format="JPEG", quality=85),
                    "detail": "low",
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": "You are a careful video safety assessor."},
        {"role": "user", "content": content},
    ]


def call_gemini(args: argparse.Namespace, messages: list[dict[str, Any]]) -> str:
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stream=False,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Gemini returned empty content.")
    return content.strip()


def build_result(args: argparse.Namespace, prompt: str, response: str, elapsed_seconds: float) -> dict[str, Any]:
    from lifebench_infer import parse_structured_response_text

    try:
        parsed_response = parse_structured_response_text(response)
    except Exception:
        parsed_response = None
    return {
        "backend": "gemini",
        "description": "Gemini multimodal inference through OpenAI-compatible API.",
        "model_id": args.model,
        "model_path": None,
        "video_path": str(Path(args.video_path).expanduser().resolve()),
        "prompt": prompt,
        "response": response,
        "parsed_response": parsed_response,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_frames": args.max_frames,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video_path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    if args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1.")

    prompt = load_prompt(args)
    started_at = time.time()
    response = call_gemini(args, build_messages(video_path, prompt, args.max_frames))
    result = build_result(args, prompt, response, time.time() - started_at)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(response)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

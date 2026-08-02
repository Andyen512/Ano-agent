#!/usr/bin/env python3
"""Collect fully skipped video keys from model inference state files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIMENSIONS = ("perception", "cognition", "grounding", "planning")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    keys: set[str] = set()
    for path in sorted(args.state_root.rglob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for video_key, video_state in state.get("videos", {}).items():
            dimensions = video_state.get("dimensions", {})
            if all(
                dimensions.get(dimension, {}).get("status") == "skipped"
                for dimension in DIMENSIONS
            ) and any(
                dimensions.get(dimension, {}).get("skip_reason")
                == "manually skipped after prolonged inference stall"
                for dimension in DIMENSIONS
            ):
                keys.add(str(video_key).removesuffix(".mp4"))

    existing: set[str] = set()
    if args.output.exists():
        existing = {
            line.strip().removesuffix(".mp4")
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    keys.update(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "# Video keys skipped globally after prolonged inference stalls\n"
    content += "\n".join(sorted(keys))
    content += "\n" if keys else ""
    args.output.write_text(content, encoding="utf-8")
    print(f"Collected {len(keys)} globally skipped video keys -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

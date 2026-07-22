#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_DIR="${1:-${PROJECT_ROOT}/data/public_data/web}"
DST_DIR="${2:-${PROJECT_ROOT}/data/public_data/web_h264}"
PYTHON_BIN="${PYTHON_BIN:-/data_4/liuyuan/anaconda3/envs/swift-3.6/bin/python}"
OVERWRITE="${OVERWRITE:-0}"

"${PYTHON_BIN}" - "$SRC_DIR" "$DST_DIR" "$OVERWRITE" <<'PY'
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

src_dir = Path(sys.argv[1]).expanduser().resolve()
dst_dir = Path(sys.argv[2]).expanduser().resolve()
overwrite = sys.argv[3] == "1"
video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

if not src_dir.is_dir():
    raise SystemExit(f"Source directory does not exist: {src_dir}")

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise SystemExit(
        "imageio-ffmpeg is required. Install it with: "
        f"{sys.executable} -m pip install imageio-ffmpeg"
    ) from exc

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
videos = sorted(path for path in src_dir.rglob("*") if path.is_file() and path.suffix.lower() in video_exts)
dst_dir.mkdir(parents=True, exist_ok=True)


def ffmpeg_probe_text(path: Path) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    return result.stdout


def is_av1(path: Path) -> bool:
    text = ffmpeg_probe_text(path).lower()
    return "video: av1" in text or "codec av1" in text


converted = 0
copied = 0
skipped = 0
failed = 0

for index, src in enumerate(videos, start=1):
    rel = src.relative_to(src_dir)
    dst = dst_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        skipped += 1
        print(f"[{index}/{len(videos)}] skip existing {rel}", flush=True)
        continue

    try:
        needs_transcode = is_av1(src)
    except Exception as exc:
        failed += 1
        print(f"[{index}/{len(videos)}] probe failed {rel}: {exc}", flush=True)
        continue

    if not needs_transcode:
        shutil.copy2(src, dst)
        copied += 1
        print(f"[{index}/{len(videos)}] copy {rel}", flush=True)
        continue

    tmp = dst.with_suffix(dst.suffix + ".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    if result.returncode != 0:
        failed += 1
        print(f"[{index}/{len(videos)}] transcode failed {rel}", flush=True)
        print(result.stdout[-2000:], flush=True)
        if tmp.exists():
            tmp.unlink()
        continue
    tmp.replace(dst)
    converted += 1
    print(f"[{index}/{len(videos)}] av1 -> h264 {rel}", flush=True)

print(
    {
        "source": str(src_dir),
        "dest": str(dst_dir),
        "total": len(videos),
        "converted_av1": converted,
        "copied": copied,
        "skipped_existing": skipped,
        "failed": failed,
    },
    flush=True,
)

if failed:
    raise SystemExit(1)
PY

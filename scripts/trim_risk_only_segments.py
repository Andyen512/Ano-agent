#!/usr/bin/env python3
"""Trim exact risk-only precursor intervals from existing risk_segment videos.

The existing risk_segment videos are [0, end] precursor clips. This script reads
their annotated precursor interval [start, end], copies each source clip to a
temporary file, and trims only [start, end] into risk_only_segment.
"""

from __future__ import annotations

import csv
import re
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path


DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")
PUBLIC_DATA_ROOT = Path("/home/caiqingyuan/code/lifebench/data/public_data")
SRC_ROOT = PUBLIC_DATA_ROOT / "risk_segment"
DST_ROOT = PUBLIC_DATA_ROOT / "risk_only_segment"
MANIFEST_PATH = DST_ROOT / "manifest.csv"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".ts"}


def strip_video_ext(value: str) -> str:
    suffix = Path(value).suffix.lower()
    if suffix in VIDEO_EXTS:
        return value[: -len(suffix)]
    return value


def parse_precursor_interval(value: str) -> tuple[float, float] | None:
    nums = re.findall(r"\d+(?:\.\d+)?", (value or "").replace("，", ","))
    if not nums:
        return None
    if len(nums) == 1:
        start, end = 0.0, float(nums[0])
    else:
        start, end = float(nums[0]), float(nums[-1])
    if end <= start:
        return None
    return start, end


def load_annotation_intervals() -> dict[str, tuple[str, float, float]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT ha.video_key, ha.precursor_start_time
        FROM human_annotations ha
        JOIN videos v ON v.video_key = ha.video_key
        WHERE ha.precursor_start_time IS NOT NULL
          AND ha.precursor_start_time != ''
          AND v.dataset != 'generated_videos'
        """
    ).fetchall()
    conn.close()

    intervals: dict[str, tuple[str, float, float]] = {}
    for row in rows:
        parsed = parse_precursor_interval(row["precursor_start_time"])
        if parsed is None:
            continue
        start, end = parsed
        key = row["video_key"]
        norm = strip_video_ext(key)
        current = intervals.get(norm)
        if current is None or end > current[2]:
            intervals[norm] = (key, start, end)
    return intervals


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    candidates = [
        shutil.which("ffmpeg"),
        "/home/caiqingyuan/miniconda3/envs/lifebench-vlm/bin/ffmpeg",
        "/home/caiqingyuan/miniconda3/envs/3dpose/bin/ffmpeg",
    ]
    candidates.extend(
        str(path)
        for path in sorted(Path("/home/caiqingyuan/miniconda3/pkgs").glob("ffmpeg-*/bin/ffmpeg"))
    )
    for ffmpeg in candidates:
        if ffmpeg and Path(ffmpeg).exists():
            return ffmpeg
    raise RuntimeError("ffmpeg not found; install imageio_ffmpeg or ffmpeg")


def trim_from_copy(ffmpeg: str, src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="risk_only_trim_") as tmp_dir:
        tmp_src = Path(tmp_dir) / src.name
        shutil.copy2(src, tmp_src)
        duration = end - start
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(tmp_src),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(dst),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:] if result.stderr else "ffmpeg failed")
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced an empty output")


def main() -> int:
    intervals = load_annotation_intervals()
    ffmpeg = get_ffmpeg_exe()
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    success = 0
    skipped = 0
    failed = 0

    for src in sorted(SRC_ROOT.rglob("*.mp4")):
        rel = src.relative_to(SRC_ROOT)
        norm = strip_video_ext(rel.as_posix())
        interval = intervals.get(norm)
        if interval is None:
            rows.append(
                {
                    "source": rel.as_posix(),
                    "output": "",
                    "video_key": "",
                    "start": "",
                    "end": "",
                    "status": "missing_annotation",
                    "error": "",
                }
            )
            skipped += 1
            continue

        video_key, start, end = interval
        dst = DST_ROOT / rel
        status = "success"
        error = ""
        try:
            if dst.exists() and dst.stat().st_size > 0:
                status = "exists"
                skipped += 1
            else:
                trim_from_copy(ffmpeg, src, dst, start, end)
                success += 1
        except Exception as exc:  # noqa: BLE001 - manifest should capture all failures.
            failed += 1
            status = "failed"
            error = str(exc).replace("\n", " ")[:500]

        rows.append(
            {
                "source": rel.as_posix(),
                "output": dst.relative_to(DST_ROOT).as_posix(),
                "video_key": video_key,
                "start": f"{start:g}",
                "end": f"{end:g}",
                "status": status,
                "error": error,
            }
        )
        processed = success + skipped + failed
        if processed % 50 == 0:
            print(f"processed={processed} success={success} skipped={skipped} failed={failed}", flush=True)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "output", "video_key", "start", "end", "status", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"source_files={len(rows)}")
    print(f"success={success}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"output_dir={DST_ROOT}")
    print(f"manifest={MANIFEST_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""从 DB 取所有 precursor 视频，按 precursor_start_time 的 end 秒裁剪 [0,end]，输出到 data/public_data/risk_segment/."""

import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

DB_PATH = "/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db"
SRC_ROOT = Path("/home/caiqingyuan/code/lifebench/data/public_data")
DST_ROOT = Path("/home/caiqingyuan/code/lifebench/data/public_data/risk_segment")

# 视频扩展名（用 .mp4 输出，ffmpeg 会自动转码）
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".ts"]


def parse_precursor_time(s: str) -> int | None:
    """从 '0,5' 或 '0,5' 或 '0, 5' 这种格式里取最后一个数字（end 秒）。"""
    if not s:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", s.replace("，", ","))
    if not nums:
        return None
    return int(float(nums[-1]))


def find_src(video_key: str) -> Path | None:
    """从 video_key 找原视频文件。"""
    p = SRC_ROOT / video_key
    if p.exists():
        return p
    for ext in VIDEO_EXTS:
        cand = p.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def get_ffmpeg_exe() -> str | None:
    """优先用 imageio_ffmpeg，回退到系统 ffmpeg。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    for cand in ("ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if shutil.which(cand):
            return cand
    return None


def trim_with_ffmpeg(src: Path, dst: Path, end_sec: int) -> bool:
    """用 ffmpeg 裁剪 [0, end_sec] 段，输出到 dst。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_exe()
    if ffmpeg_exe is None:
        print(f"  [ffmpeg not found] install imageio_ffmpeg or ffmpeg")
        return False
    # 先复制再 ffmpeg 重新编码（避免 -c copy 引起的时长不准）
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(src),
        "-t", str(end_sec),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return True
        print(f"  [ffmpeg fail] {src.name} -> {dst.name}: {result.stderr[-300:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [ffmpeg timeout] {src.name}")
        return False
    except FileNotFoundError:
        print(f"  [ffmpeg not found] {ffmpeg_exe}")
        return False


def main():
    import sys
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        """
        SELECT ha.video_key, v.dataset, ha.precursor_start_time
        FROM human_annotations ha
        JOIN videos v ON ha.video_key = v.video_key
        WHERE ha.precursor_start_time IS NOT NULL
          AND ha.precursor_start_time != ''
          AND v.dataset != 'generated_videos'
        """
    ).fetchall()
    db.close()

    print(f"Total precursor rows: {len(rows)}")

    # 同一视频可能多条 precursor 标注，取最大 end（保守裁剪）
    by_video: dict[str, tuple[str, int]] = {}
    for vk, ds, pt in rows:
        end = parse_precursor_time(pt)
        if end is None or end <= 0:
            continue
        if vk not in by_video or end > by_video[vk][1]:
            by_video[vk] = (ds, end)

    print(f"Unique precursor videos: {len(by_video)}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    success, fail, skip = 0, 0, 0
    fail_list = []
    items = sorted(by_video.items())
    if limit:
        items = items[:limit]
    for vk, (ds, end) in items:
        # 输出路径：risk_segment/<原 video_key 路径>/<stem>.mp4
        # 把 video_key 里的 dataset/xxx 转成 dst 路径，保留 dataset 一致
        rel = Path(vk)  # e.g. SmartHome-Bench/videos/smartbench_0209.mp4
        stem = rel.stem  # smartbench_0209
        dst = DST_ROOT / rel.parent / f"{stem}.mp4"

        # 优先检查已经裁剪过
        if dst.exists() and dst.stat().st_size > 0:
            skip += 1
            continue

        src = find_src(vk)
        if src is None:
            fail += 1
            fail_list.append((vk, "source not found"))
            continue

        if trim_with_ffmpeg(src, dst, end):
            success += 1
            if success % 20 == 0:
                print(f"  [{success} done] {vk} -> {dst.relative_to(DST_ROOT)} (0~{end}s)")
        else:
            fail += 1
            fail_list.append((vk, "ffmpeg failed"))

    print(f"\nDone. Success: {success}, Fail: {fail}, Skip: {skip}")
    if fail_list:
        print(f"\nFailures:")
        for vk, reason in fail_list[:20]:
            print(f"  {vk}: {reason}")


if __name__ == "__main__":
    main()

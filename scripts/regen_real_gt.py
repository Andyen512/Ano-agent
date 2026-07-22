#!/usr/bin/env python3
"""从 DB 重新生成 real_gt.txt，包含所有已标注的真实视频。"""

import sqlite3
import re
from collections import Counter
from pathlib import Path

DB_PATH = "/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db"
GT_OUTS = [
    Path("/home/caiqingyuan/code/lifebench/data/annotation/real_gt.txt"),
    Path("/data/caiqingyuan/Dataset/lifebench/data/annotation/real_gt.txt"),
]
RISK_SEGMENT_ROOT = Path("/home/caiqingyuan/code/lifebench/data/public_data/risk_segment")

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv"]

RISK_TYPE_RE = re.compile(r"\s*/\s*")


def fmt_time_segment(seg: str) -> str:
    """把 '0,15' 这种 risk_localization 格式转成 [Abnormal Result][0-15s]"""
    if not seg:
        return ""
    seg = seg.strip("[] ")
    nums = re.findall(r"\d+(?:\.\d+)?", seg)
    if not nums:
        return ""
    if len(nums) == 1:
        return f"[Abnormal Sign][{nums[0]}s]"
    return f"[Abnormal Result][{nums[0]}-{nums[1]}s]"


def build_path(level1: str, level2: str, level3: str) -> str:
    parts = [level1 or "", level2 or "", level3 or ""]
    return "/".join(parts) + "/"


def build_solution(person: str, hazard: str, prevention: str) -> str:
    pieces = []
    if person:
        pieces.append(f"Solution for the person: {person.strip()}")
    if hazard:
        pieces.append(f"Solution for the hazard source: {hazard.strip()}")
    if prevention:
        pieces.append(f"Follow-up prevention solution: {prevention.strip()}")
    return " ".join(pieces)


def fmt_video_id(key: str) -> str:
    """去掉扩展名 .mp4/.avi 等"""
    for ext in VIDEO_EXTS:
        if key.endswith(ext):
            return key[: -len(ext)]
    return key


def fmt_path_for_gt(key: str) -> str:
    """把 video_key 转成 path_text: dataset/scene/subject/risk/"""
    parts = key.split("/")
    if len(parts) < 4:
        return key
    return "/".join(parts[1:4]) + "/"


def status_from_annotation(risk: str, risk_subtype: str) -> str:
    if risk == "No":
        return "normal"
    if risk == "Yes" and risk_subtype in {"abnormal", "risk_only"}:
        return risk_subtype
    if risk == "Yes":
        return "abnormal"
    return "risk_only"


def build_timeline(risk: str, loc: str, desc: str) -> str:
    if risk == "Yes":
        timeline = ""
        if loc:
            timeline = fmt_time_segment(loc) + " "
        if desc:
            timeline += desc
        return timeline.strip()
    return "[Safe Action] " + (desc or "正常活动，无异常")


def make_line(video_id: str, key_for_path: str, status: str, row) -> str:
    (
        _key,
        risk,
        _risk_subtype,
        l1,
        l2,
        l3,
        desc,
        loc,
        sp,
        sh,
        spr,
    ) = row
    path = fmt_path_for_gt(key_for_path)
    timeline = build_timeline(risk, loc, desc)
    solution = build_solution(sp, sh, spr)
    return f"{video_id}\t{path}\t{status}\t{timeline}\t{solution}"


def risk_segment_annotation_key(rel: str, annotations_by_key: dict[str, tuple]) -> str | None:
    candidates = [rel]
    for ext in VIDEO_EXTS:
        if rel.endswith(ext):
            candidates.append(rel[: -len(ext)])
            break
    for candidate in candidates:
        if candidate in annotations_by_key:
            return candidate
    return None


def main():
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        """
        WITH latest AS (
            SELECT ha.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY ha.video_key
                       ORDER BY COALESCE(ha.updated_at, '') DESC, ha.id DESC
                   ) AS rn
            FROM human_annotations ha
            JOIN videos v ON ha.video_key = v.video_key
            WHERE v.dataset != 'generated_videos'
        )
        SELECT video_key, risk, COALESCE(risk_subtype, ''),
               ha.level1_scene, ha.level2_subject, ha.level3_risk_type,
               ha.description, ha.risk_localization,
               ha.solution_for_person, ha.solution_for_hazard_source, ha.solution_prevent_recurrence
        FROM latest ha
        WHERE rn = 1
        ORDER BY video_key
        """
    ).fetchall()
    db.close()

    annotations_by_key = {r[0]: r for r in rows}
    lines = []
    for r in rows:
        key, risk, risk_subtype, *_rest = r
        status = status_from_annotation(risk, risk_subtype)
        video_id = fmt_video_id(key)
        lines.append(make_line(video_id, key, status, r))

    missing_risk_segments = []
    risk_segment_count = 0
    for path in sorted(RISK_SEGMENT_ROOT.rglob("*.mp4")):
        rel = path.relative_to(RISK_SEGMENT_ROOT).as_posix()
        source_key = risk_segment_annotation_key(rel, annotations_by_key)
        if source_key is None:
            missing_risk_segments.append(rel)
            continue
        row = annotations_by_key[source_key]
        video_id = "risk_segment/" + fmt_video_id(rel)
        lines.append(make_line(video_id, source_key, "abnormal", row))
        risk_segment_count += 1

    if missing_risk_segments:
        print(f"Warning: {len(missing_risk_segments)} risk_segment files have no matching annotation")
        for rel in missing_risk_segments[:10]:
            print(f"  missing: {rel}")

    text = "\n".join(lines) + "\n"
    written_inodes = set()
    for out in GT_OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        inode = None
        if out.exists():
            stat = out.stat()
            inode = (stat.st_dev, stat.st_ino)
        if inode in written_inodes:
            continue
        out.write_text(text, encoding="utf-8")
        if out.exists():
            stat = out.stat()
            written_inodes.add((stat.st_dev, stat.st_ino))

    counts = Counter(line.split("\t")[2] for line in lines)
    print(f"Wrote {len(lines)} entries")
    for out in GT_OUTS:
        print(f"  {out}")
    print(f"status counts: {dict(counts)}")
    print(f"risk_segment appended as abnormal: {risk_segment_count}")


if __name__ == "__main__":
    main()

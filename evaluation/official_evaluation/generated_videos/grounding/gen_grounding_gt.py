#!/usr/bin/env python3
"""Generate grounding GT for generated_videos (time spans from descriptions)."""
import json, re
from pathlib import Path

INPUT_FILE = Path("/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_expanded_solutions.txt")
OUTPUT_DIR = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/generated_videos/grounding")
OUTPUT_JSON = OUTPUT_DIR / "grounding_gt.json"


def extract_time_spans(desc: str) -> list[list[float]]:
    if not desc:
        return []
    spans: list[list[float]] = []

    # "[Abnormal Sign][3-8s]" / "[Abnormal Result][5-8s]"
    for m in re.finditer(r"\[(?:Abnormal Sign|Abnormal Result)\]\s*\[(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*s?\]", desc):
        spans.append([float(m.group(1)), float(m.group(2))])

    # "[Normal Behavior][0-3s]" also consider as reference (not a risk span, but we include it for completeness)

    # Simple "2-8s：" or "2-8秒：" prefix
    if not spans:
        m = re.match(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*[s秒秒]?\s*[：:：]", desc)
        if m:
            spans.append([float(m.group(1)), float(m.group(2))])

    spans.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def extract_video_id(path: str) -> str:
    # path: data/generated_videos_20260703/generated_videos/.../file.mp4
    # -> generated_videos/.../file.mp4 (without .mp4)
    # Find "generated_videos/" after the data dir
    idx = path.find("/generated_videos/")
    if idx != -1:
        vid = path[idx + 1:]  # skip leading /
    else:
        vid = path
    # Remove .mp4 suffix for matching with real_videos convention
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    return vid


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        lines = f.readlines()

    hparts = header.rstrip("\n").split("\t")
    path_idx = hparts.index("video_path")
    desc_idx = hparts.index("description")
    vtype_idx = hparts.index("video_type")

    results = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(path_idx, desc_idx, vtype_idx):
            continue

        vpath = parts[path_idx]
        vtype = parts[vtype_idx]
        desc = parts[desc_idx] if desc_idx < len(parts) else ""

        risk_status = vtype.lower()  # abnormal / risk_only / normal

        spans = extract_time_spans(desc)
        start_time = spans[0][0] if spans else None
        end_time = spans[-1][-1] if spans else None
        duration = round(end_time - start_time, 1) if start_time is not None and end_time is not None else None

        results.append({
            "video_id": extract_video_id(vpath),
            "risk_status": risk_status,
            "time_spans": spans,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Grounding GT: {len(results)} entries -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
从 GT 标注提取 Grounding GT：
- time_spans: 异常时间区间列表
- start_time: 异常开始时间
- end_time: 异常结束时间
- duration: 异常持续时间
正则提取，不调 LLM。
"""

import json
import re
from pathlib import Path

GT_TSV = Path("/home/caiqingyuan/code/lifebench/data/annotation/real_gt_expanded_release.txt")
OUTPUT_JSON = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/real_videos/grounding/grounding_gt.json")


def extract_time_spans(timeline: str) -> list[list[float]]:
    if not timeline:
        return []

    spans: list[list[float]] = []

    # 单点格式: [Abnormal Sign][0s] 或 [0.9s]
    for m in re.finditer(r"\[(?:Abnormal Result|Abnormal Sign)\]\s*\[(\d+(?:\.\d+)?)\s*s?\]", timeline):
        t = float(m.group(1))
        spans.append([t, t])

    # 范围格式: [Abnormal Result][3-17s]
    for m in re.finditer(r"\[(?:Abnormal Result|Abnormal Sign)\]\s*\[(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*s?\]", timeline):
        spans.append([float(m.group(1)), float(m.group(2))])

    # 范围格式: [Abnormal Result][[3,17]]
    for m in re.finditer(r"\[(?:Abnormal Result|Abnormal Sign)\]\s*\[\[(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\]\]", timeline):
        spans.append([float(m.group(1)), float(m.group(2))])

    # 范围格式: [Abnormal Result][0，9]
    for m in re.finditer(r"\[(?:Abnormal Result|Abnormal Sign)\]\s*\[(\d+(?:\.\d+)?)\s*[，,]\s*(\d+(?:\.\d+)?)\s*\]", timeline):
        spans.append([float(m.group(1)), float(m.group(2))])

    spans.sort(key=lambda x: (x[0], x[1]))
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def parse_gt_line(line: str):
    parts = [p.strip() for p in line.split("\t")]
    if len(parts) < 3:
        return None
    s = parts[2].strip().lower().replace("-", "").replace("_", "")
    if s in ("normal", "noanomaly", "safe", "riskonly", "potential", "abnormal", "occurred", "anomalyoccurred"):
        return {"video_id": parts[0], "risk_status": parts[2], "timeline": parts[3] if len(parts) > 3 else ""}
    return {"video_id": parts[0], "risk_status": "unknown", "timeline": parts[2]}


def main():
    lines = GT_TSV.read_text(encoding="utf-8").splitlines()
    print(f"共 {len(lines)} 行")

    results = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parsed = parse_gt_line(line)
        if not parsed:
            continue

        spans = extract_time_spans(parsed["timeline"])
        start_time = spans[0][0] if spans else None
        end_time = spans[-1][-1] if spans else None
        duration = round(end_time - start_time, 1) if start_time is not None and end_time is not None else None

        results.append({
            "video_id": parsed["video_id"],
            "risk_status": parsed["risk_status"],
            "time_spans": spans,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，写入 {OUTPUT_JSON} ({len(results)} 条)")


if __name__ == "__main__":
    main()

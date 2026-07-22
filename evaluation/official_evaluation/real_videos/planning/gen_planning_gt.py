#!/usr/bin/env python3
"""
从 GT 标注提取 Planning GT：
- person_solution: 对人方案
- hazard_solution: 对危险源的方案
- prevention_solution: 预防/整体方案
直接从 solution 文本中按标签分割提取，不调 LLM。
"""

import json
import re
from pathlib import Path

GT_TSV = Path("/home/caiqingyuan/code/lifebench/data/annotation/real_gt_expanded_release.txt")
OUTPUT_JSON = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/real_videos/planning/planning_gt.json")

SOLUTION_LABELS: list[tuple[str, list[str]]] = [
    ("person_solution", [
        "Solution for the person:", "Solution for the person",
        "对人的解决方案：",
    ]),
    ("hazard_solution", [
        "Solution for the hazard source:", "Solution for the hazard source",
        "对危险源的解决方案：",
    ]),
    ("prevention_solution", [
        "Follow-up prevention solution:", "Follow-up prevention solution",
        "Overall solution:", "整体的解决方案：",
        "后续防止危险复发的解决方案：",
    ]),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_solution_sections(text: str) -> dict[str, str]:
    if not text:
        return {"person_solution": "", "hazard_solution": "", "prevention_solution": ""}

    positions: list[tuple[str, int, str]] = []
    for key, labels in SOLUTION_LABELS:
        best_pos = -1
        best_label = ""
        for label in labels:
            pos = text.find(label)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_label = label
        if best_pos != -1:
            positions.append((key, best_pos, best_label))

    if not positions:
        return {"person_solution": clean(text), "hazard_solution": "", "prevention_solution": ""}

    positions.sort(key=lambda x: x[1])
    sections = {}
    for i, (key, start_pos, label) in enumerate(positions):
        content_start = start_pos + len(label)
        if i + 1 < len(positions):
            content_end = positions[i + 1][1]
        else:
            content_end = len(text)
        sections[key] = clean(text[content_start:content_end])
    for key, _labels in SOLUTION_LABELS:
        sections.setdefault(key, "")
    return sections


def parse_gt_line(line: str):
    parts = [p.strip() for p in line.split("\t")]
    if len(parts) < 3:
        return None
    s = parts[2].strip().lower().replace("-", "").replace("_", "")
    if s in ("normal", "noanomaly", "safe", "riskonly", "potential", "abnormal", "occurred", "anomalyoccurred"):
        return {"video_id": parts[0], "risk_status": parts[2],
                "solution": parts[4] if len(parts) > 4 else ""}
    return {"video_id": parts[0], "risk_status": "unknown",
            "solution": parts[3] if len(parts) > 3 else ""}


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

        sections = parse_solution_sections(parsed["solution"])

        results.append({
            "video_id": parsed["video_id"],
            "risk_status": parsed["risk_status"],
            "person_solution": sections.get("person_solution", ""),
            "hazard_solution": sections.get("hazard_solution", ""),
            "prevention_solution": sections.get("prevention_solution", ""),
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，写入 {OUTPUT_JSON} ({len(results)} 条)")


if __name__ == "__main__":
    main()

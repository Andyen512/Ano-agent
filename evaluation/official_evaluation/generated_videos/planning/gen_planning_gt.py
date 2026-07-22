#!/usr/bin/env python3
"""Generate planning GT for generated_videos (solutions parsed from text)."""
import json, re
from pathlib import Path

INPUT_FILE = Path("/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_expanded_solutions.txt")
OUTPUT_DIR = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/generated_videos/planning")
OUTPUT_JSON = OUTPUT_DIR / "planning_gt.json"

SOLUTION_LABELS = [
    ("对人的解决方案：", "person_solution", ["对人的解决方案："]),
    ("对危险源的解决方案：", "hazard_solution", ["对危险源的解决方案："]),
    ("后续防止危险复发的解决方案：", "prevention_solution", ["后续防止危险复发的解决方案："]),
]


def parse_solutions(text: str) -> dict[str, str]:
    result = {"person_solution": "", "hazard_solution": "", "prevention_solution": ""}
    if not text:
        return result

    positions = []
    for key_name, field, labels in SOLUTION_LABELS:
        best_pos = -1
        best_label = ""
        for label in labels:
            pos = text.find(label)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
                best_label = label
        if best_pos != -1:
            positions.append((field, best_pos, best_label))

    if not positions:
        return result

    positions.sort(key=lambda x: x[1])
    for i, (field, start_pos, label) in enumerate(positions):
        content_start = start_pos + len(label)
        if i + 1 < len(positions):
            content_end = positions[i + 1][1]
        else:
            content_end = len(text)
        content = text[content_start:content_end].strip()
        result[field] = content

    return result


def extract_video_id(path: str) -> str:
    idx = path.find("/generated_videos/")
    if idx != -1:
        vid = path[idx + 1:]
    else:
        vid = path
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    return vid


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        lines = f.readlines()

    hparts = header.rstrip("\n").split("\t")
    path_idx = hparts.index("video_path")
    sol_idx = hparts.index("solutions")
    vtype_idx = hparts.index("video_type")

    results = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(path_idx, sol_idx, vtype_idx):
            continue

        vpath = parts[path_idx]
        vtype = parts[vtype_idx]
        sol = parts[sol_idx] if sol_idx < len(parts) else ""

        risk_status = vtype.lower()
        sections = parse_solutions(sol)

        results.append({
            "video_id": extract_video_id(vpath),
            "risk_status": risk_status,
            "person_solution": sections.get("person_solution", ""),
            "hazard_solution": sections.get("hazard_solution", ""),
            "prevention_solution": sections.get("prevention_solution", ""),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Planning GT: {len(results)} entries -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

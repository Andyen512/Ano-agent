#!/usr/bin/env python3
"""将 generated_videos 的 GT 标注迁移到 annotations 目录，按 risk_status / model 组织。"""
import json
from pathlib import Path
from collections import defaultdict

SRC_BASE = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/generated_videos")
DST_BASE = Path("/home/caiqingyuan/code/lifebench/data/public_data_release/annotations/generated_videos")

DIMENSIONS = ["cognition", "perception", "grounding", "planning"]


def extract_model(video_id: str) -> str:
    parts = video_id.split("/")
    if len(parts) >= 7:
        return parts[5]  # <scene>/<subject>/<risk_type>/<desc>/<model>/<status>/...
    return "unknown"


def extract_risk_status_from_video_id(video_id: str) -> str:
    parts = video_id.split("/")
    if len(parts) >= 7:
        return parts[6]  # <status>
    return "unknown"


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_cognition(entry: dict) -> dict:
    """Normalize cognition entry to match real_videos format."""
    if "risk_description" in entry:
        # abnormal or risk_only entry
        status = extract_risk_status_from_video_id(entry["video_id"])
        return {
            "video_id": entry["video_id"],
            "risk_description": entry["risk_description"],
            "consequence_understanding": entry.get("consequence_understanding", []),
            "causal_chain": entry.get("causal_chain", []),
            "causal_chain_text": entry.get("causal_chain_text", ""),
            "event_summary": entry.get("event_summary", ""),
            "risk_status": status,
        }
    else:
        # normal entry: convert video_description -> risk_description
        return {
            "video_id": entry["video_id"],
            "risk_description": entry["video_description"],
            "consequence_understanding": [],
            "causal_chain": [],
            "causal_chain_text": "",
            "event_summary": "",
            "risk_status": "normal",
        }


def main():
    for dim in DIMENSIONS:
        src = SRC_BASE / dim / f"{dim}_gt.json"
        if not src.exists():
            print(f"SKIP: {src} not found")
            continue

        data = load_json(src)
        print(f"{dim}: {len(data)} entries")

        # Group by (risk_status, model)
        groups = defaultdict(list)
        for entry in data:
            if dim == "cognition":
                entry = normalize_cognition(entry)
            status = entry.get("risk_status", "unknown")
            model = extract_model(entry["video_id"])
            groups[(status, model)].append(entry)

        for (status, model), entries in groups.items():
            dst = DST_BASE / status / model / f"{dim}_gt.json"
            entries.sort(key=lambda x: x["video_id"])
            save_json(entries, dst)
            print(f"  -> {status}/{model}/{dim}_gt.json ({len(entries)} entries)")

    print("\nDone!")


if __name__ == "__main__":
    main()

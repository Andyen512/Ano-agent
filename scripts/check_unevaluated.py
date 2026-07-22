#!/usr/bin/env python3
"""对比 DB 中已标注的真实视频 与 已评估视频，找出未评估的新增视频。"""

import json
import sqlite3
import os
from collections import defaultdict

EVAL_SCORES = "/home/caiqingyuan/code/lifebench/outputs/evaluation/real_videos/per_video_scores.json"
DB_PATH = "/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db"
OUTPUT_FILE = "/home/caiqingyuan/code/lifebench/outputs/evaluation/real_videos/unevaluated_videos.json"


def main():
    # 1. 从评估结果中提取已评估的视频ID
    print("Loading evaluated video IDs...")
    with open(EVAL_SCORES) as f:
        eval_data = json.load(f)
    evaluated_videos = {entry["video_id"] for entry in eval_data}
    print(f"  Evaluated videos: {len(evaluated_videos)}")

    # 2. 从 DB 中提取已标注的真实视频（排除 generated_videos）
    print("Loading annotated real video IDs from DB...")
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        """
        SELECT DISTINCT ha.video_key, v.dataset
        FROM human_annotations ha
        JOIN videos v ON ha.video_key = v.video_key
        WHERE v.dataset != 'generated_videos'
        """
    ).fetchall()

    annotated_videos = {r[0] for r in rows}
    video_dataset = {r[0]: r[1] for r in rows}
    print(f"  Annotated real videos (excl generated): {len(annotated_videos)}")

    # 3. 风险/无风险标注
    risk_rows = db.execute(
        """
        SELECT video_key, risk
        FROM human_annotations
        WHERE video_key IN (
            SELECT DISTINCT ha.video_key
            FROM human_annotations ha
            JOIN videos v ON ha.video_key = v.video_key
            WHERE v.dataset != 'generated_videos'
        )
        """
    ).fetchall()

    video_risk = defaultdict(set)
    for vk, risk in risk_rows:
        video_risk[vk].add(risk)

    db.close()

    # 4. 对比找差集
    unevaluated = annotated_videos - evaluated_videos
    overlap = annotated_videos & evaluated_videos
    eval_not_annotated = evaluated_videos - annotated_videos

    print(f"\n{'='*50}")
    print(f"Overlap (both annotated & evaluated): {len(overlap)}")
    print(f"Annotated but NOT evaluated:           {len(unevaluated)}")
    print(f"Evaluated but NOT annotated:           {len(eval_not_annotated)}")
    print(f"{'='*50}")

    # 5. 未评估视频的风险分布 + 数据集分布
    risk_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    dataset_risk = defaultdict(lambda: defaultdict(int))

    for v in sorted(unevaluated):
        ds = video_dataset.get(v, "unknown")
        risks = video_risk.get(v, set())
        dataset_counts[ds] += 1
        risk_label = "risk" if "Yes" in risks else "no_risk"
        if "Yes" in risks and "No" in risks:
            risk_label = "both"
        risk_counts[risk_label] += 1
        dataset_risk[ds][risk_label] += 1

    print(f"\nNot-evaluated risk breakdown:")
    for k in ["risk", "no_risk", "both"]:
        print(f"  {k}: {risk_counts[k]}")
    print(f"  Total: {sum(risk_counts.values())}")

    print(f"\nNot-evaluated by dataset:")
    for ds in sorted(dataset_counts.keys()):
        dr = dataset_risk[ds]
        ris = dr.get("risk", 0)
        nor = dr.get("no_risk", 0)
        both = dr.get("both", 0)
        print(f"  {ds}: {dataset_counts[ds]} (risk={ris}, no_risk={nor}, both={both})")

    # 6. 输出未评估视频列表
    output = {
        "evaluated_count": len(evaluated_videos),
        "annotated_count": len(annotated_videos),
        "unevaluated_count": len(unevaluated),
        "overlap_count": len(overlap),
        "risk_breakdown": dict(risk_counts),
        "by_dataset": {
            ds: {"total": cnt, "risk": dataset_risk[ds].get("risk", 0),
                 "no_risk": dataset_risk[ds].get("no_risk", 0),
                 "both": dataset_risk[ds].get("both", 0)}
            for ds, cnt in dataset_counts.items()
        },
        "video_ids": sorted(unevaluated),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull list saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

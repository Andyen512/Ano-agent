#!/usr/bin/env python3
"""Create model_predictions for all normal videos without one, loading original_prompt."""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")
PROMPTS_FILE = Path("/home/caiqingyuan/code/lifebench/data/prompts_0529.txt")


def load_prompts():
    prompts = {}
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                video_id = parts[0]
                prompt_text = parts[3]
                prompts[video_id] = prompt_text
    return prompts


def main():
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    conn = sqlite3.connect(DB_PATH)

    # Find normal videos without model_predictions
    rows = conn.execute("""
        SELECT v.video_key, v.video_id, v.title, v.file_path
        FROM videos v
        WHERE v.dataset = 'generated_videos'
          AND v.video_key LIKE '%/normal/%'
          AND v.video_key NOT IN (SELECT video_key FROM model_predictions)
    """).fetchall()
    print(f"Found {len(rows)} normal videos without model_predictions")

    inserted = 0
    no_prompt = 0
    for video_key, video_id, title, file_path in rows:
        original_prompt = prompts.get(video_id, "")
        if not original_prompt:
            no_prompt += 1

        # Extract scene/subject/risk_type from video_key
        parts = video_key.split("/")
        # generated_videos/scene/subject/risk_type/description/model/normal/file.mp4
        level1_scene = parts[1] if len(parts) > 1 else ""
        level2_subject = parts[2] if len(parts) > 2 else ""
        level3_risk_type = parts[3] if len(parts) > 3 else ""

        agents_data = [{
            "source": "pending_eval",
            "original_prompt": original_prompt,
            "video_type": "normal",
        }]

        conn.execute("""
            INSERT INTO model_predictions (
                video_key, source_output_root, summary_json, risk, level1_scene, level2_subject,
                level3_risk_type, description, risk_localization, solution_for_person,
                solution_for_hazard_source, solution_prevent_recurrence, yes_count, no_count,
                votes_cast, complete, majority_json, agents_json, updated_at
            ) VALUES (?, '', '', 'No', ?, ?, ?, '', '', '', '', '', 0, 0, 0, 0, '{}', ?, datetime('now'))
        """, (
            video_key,
            level1_scene,
            level2_subject,
            level3_risk_type,
            json.dumps(agents_data, ensure_ascii=False),
        ))
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Inserted {inserted} model_predictions records")
    print(f"No matching prompt: {no_prompt}")


if __name__ == "__main__":
    main()
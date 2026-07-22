#!/usr/bin/env python3
"""1) Clear human_annotations for normal generated_videos
   2) Load original_prompt from prompts file into agents_json"""
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
                video_id = parts[0]  # e.g. R000001_normal
                prompt_text = parts[3]
                prompts[video_id] = prompt_text
    return prompts


def main():
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts")

    conn = sqlite3.connect(DB_PATH)

    # 1) Clear human_annotations for normal generated_videos
    deleted = conn.execute("""
        DELETE FROM human_annotations
        WHERE video_key IN (
            SELECT video_key FROM videos
            WHERE dataset = 'generated_videos' AND video_key LIKE '%/normal/%'
        )
    """).rowcount
    conn.commit()
    print(f"Deleted {deleted} human_annotations for normal videos")

    # 2) Load original_prompt into agents_json for normal videos with model_predictions
    rows = conn.execute("""
        SELECT mp.video_key, mp.agents_json, v.video_id
        FROM model_predictions mp
        JOIN videos v ON v.video_key = mp.video_key
        WHERE mp.video_key LIKE 'generated_videos/%/normal/%'
    """).fetchall()

    updated = 0
    for video_key, agents_json_str, video_id in rows:
        if not agents_json_str:
            continue
        try:
            agents = json.loads(agents_json_str)
        except json.JSONDecodeError:
            continue

        original_prompt = prompts.get(video_id, "")
        if not original_prompt:
            continue

        # agents is a list of model dicts
        if isinstance(agents, list):
            if agents and "original_prompt" not in agents[0]:
                for a in agents:
                    a["original_prompt"] = original_prompt
                conn.execute(
                    "UPDATE model_predictions SET agents_json = ? WHERE video_key = ?",
                    (json.dumps(agents, ensure_ascii=False), video_key),
                )
                updated += 1
        elif isinstance(agents, dict):
            if "original_prompt" not in agents:
                agents["original_prompt"] = original_prompt
                conn.execute(
                    "UPDATE model_predictions SET agents_json = ? WHERE video_key = ?",
                    (json.dumps(agents, ensure_ascii=False), video_key),
                )
                updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} model_predictions with original_prompt")


if __name__ == "__main__":
    main()
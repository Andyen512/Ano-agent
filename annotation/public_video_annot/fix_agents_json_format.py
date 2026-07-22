#!/usr/bin/env python3
"""Fix agents_json format for normal videos: convert list to dict with original_prompt."""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT video_key, agents_json
        FROM model_predictions
        WHERE video_key LIKE 'generated_videos/%/normal/%'
    """).fetchall()

    updated = 0
    for video_key, agents_json_str in rows:
        if not agents_json_str:
            continue
        try:
            data = json.loads(agents_json_str)
        except json.JSONDecodeError:
            continue

        # Extract original_prompt
        original_prompt = ""
        original_prompt_zh = ""
        model_predictions = []

        if isinstance(data, list):
            # Current format: list of model predictions
            for item in data:
                if isinstance(item, dict):
                    if not original_prompt:
                        original_prompt = item.get("original_prompt", "")
                    model_predictions.append(item)
        elif isinstance(data, dict):
            # Already dict format
            original_prompt = data.get("original_prompt", "")
            original_prompt_zh = data.get("original_prompt_zh", "")
            model_predictions = data.get("model_predictions", [])

        # Build new dict format
        new_data = {
            "source": "normal_eval",
            "original_prompt": original_prompt,
            "original_prompt_zh": original_prompt_zh,
            "video_type": "normal",
            "model_predictions": model_predictions,
        }

        conn.execute(
            "UPDATE model_predictions SET agents_json = ? WHERE video_key = ?",
            (json.dumps(new_data, ensure_ascii=False), video_key),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} model_predictions records")


if __name__ == "__main__":
    main()
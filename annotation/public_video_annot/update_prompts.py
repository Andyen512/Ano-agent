#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

PROMPTS_FILE = Path("/data/caiqingyuan/Dataset/lifebench/data/prompts_0529.txt")
DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"


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
    rows = conn.execute(
        "SELECT video_key, agents_json FROM model_predictions WHERE video_key LIKE 'generated_videos/%'"
    ).fetchall()

    updated = 0
    for video_key, agents_json_str in rows:
        agents = json.loads(agents_json_str) if agents_json_str else {}
        if agents.get("original_prompt"):
            continue

        stem = Path(video_key).stem
        prompt = prompts.get(stem, "")
        if not prompt:
            continue

        agents["original_prompt"] = prompt
        conn.execute(
            "UPDATE model_predictions SET agents_json = ? WHERE video_key = ?",
            (json.dumps(agents, ensure_ascii=False), video_key),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} videos with prompts")


if __name__ == "__main__":
    main()

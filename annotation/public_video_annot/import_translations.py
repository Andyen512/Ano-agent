#!/usr/bin/env python3
"""Import translations from JSON file to database."""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/data/caiqingyuan/Dataset/lifebench/annotation/public_video_annot/data/app.db")
TRANSLATIONS_FILE = Path("/home/caiqingyuan/code/lifebench/data/generated_videos/translations.json")

def main():
    if not TRANSLATIONS_FILE.exists():
        print(f"Error: {TRANSLATIONS_FILE} not found")
        return 1

    with open(TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
        translations = json.load(f)

    print(f"Loading {len(translations)} translations...")
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    done = 0
    for t in translations:
        conn.execute(
            "UPDATE model_predictions SET agents_json = ? WHERE video_key = ?",
            (t["agents_json"], t["video_key"]),
        )
        done += 1
        if done % 1000 == 0:
            conn.commit()
            print(f"[{done}/{len(translations)}]", flush=True)

    conn.commit()
    conn.close()
    print(f"\nDone! Imported {done} translations")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

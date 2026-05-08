#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "app.db"
TRANSLATION_ROOT = Path("/data_4/liuyuan/lifebench/public_data_filter/batch_outputs/persistent_full")


def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def load_translations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Clear existing translations to reload fresh
    c.execute("DELETE FROM model_prediction_zh")
    print("Cleared existing translations")

    translation_files = sorted(TRANSLATION_ROOT.glob("strict_schema_chinese_text*.jsonl"))
    print(f"Found {len(translation_files)} translation files")

    total_loaded = 0
    total_skipped = 0
    total_missing = 0

    for filepath in translation_files:
        print(f"\nProcessing: {filepath.name}")
        loaded = 0
        skipped = 0
        missing = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                video_key = row.get("video_relpath", "")

                if not video_key:
                    continue

                existing = c.execute(
                    "SELECT video_key FROM videos WHERE video_key = ?",
                    (video_key,)
                ).fetchone()

                if not existing:
                    missing += 1
                    continue

                description_zh = row.get("Normal-video description or risk description zh", "")
                risk_localization_zh = row.get("Risk localization zh", "")
                solution_for_person_zh = row.get("Solutions-For person zh", "")
                solution_for_hazard_source_zh = row.get("Solutions-For hazard source zh", "")
                solution_prevent_recurrence_zh = row.get("Solutions-Prevent recurrence zh", "")

                # Only load if description actually contains Chinese
                if not contains_chinese(description_zh):
                    skipped += 1
                    continue

                c.execute(
                    """
                    INSERT INTO model_prediction_zh (
                        video_key, description, risk_localization,
                        solution_for_person, solution_for_hazard_source,
                        solution_prevent_recurrence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(video_key) DO UPDATE SET
                        description=excluded.description,
                        risk_localization=excluded.risk_localization,
                        solution_for_person=excluded.solution_for_person,
                        solution_for_hazard_source=excluded.solution_for_hazard_source,
                        solution_prevent_recurrence=excluded.solution_prevent_recurrence,
                        updated_at=datetime('now')
                    """,
                    (
                        video_key,
                        description_zh,
                        risk_localization_zh,
                        solution_for_person_zh,
                        solution_for_hazard_source_zh,
                        solution_prevent_recurrence_zh,
                    )
                )
                loaded += 1

        print(f"  Loaded: {loaded}, Skipped (no Chinese): {skipped}, Missing in DB: {missing}")
        total_loaded += loaded
        total_skipped += skipped
        total_missing += missing

    conn.commit()

    count = c.execute("SELECT COUNT(*) FROM model_prediction_zh").fetchone()[0]
    conn.close()

    print(f"\n=== Summary ===")
    print(f"Total loaded (with Chinese): {total_loaded}")
    print(f"Total skipped (no Chinese): {total_skipped}")
    print(f"Total missing (video not in DB): {total_missing}")
    print(f"Total translations in DB: {count}")


if __name__ == "__main__":
    load_translations()

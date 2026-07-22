#!/usr/bin/env python3
"""Import normal generated videos into the database."""
import sqlite3
from pathlib import Path

GEN_DATA_ROOT = Path("/home/caiqingyuan/code/lifebench/data/generated_videos")
DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")


def video_key_from_path(video_path: Path) -> str:
    """Generate video_key from absolute path."""
    try:
        rel = video_path.relative_to(GEN_DATA_ROOT)
        return f"generated_videos/{rel}"
    except ValueError:
        return f"generated_videos/{video_path.name}"


def extract_title(video_path: Path) -> str:
    """Extract title from video path: scene/risk_type/description."""
    try:
        parts = video_path.relative_to(GEN_DATA_ROOT).parts
        # parts: scene, subject, risk_type, description, model, normal, filename
        if len(parts) >= 4:
            scene = parts[0]
            risk_type = parts[2]
            description = parts[3]
            return f"{scene}/{risk_type}/{description}"
    except ValueError:
        pass
    return ""


def main():
    # Collect all normal video files
    normal_videos = list(GEN_DATA_ROOT.rglob("normal/*.mp4"))
    print(f"Found {len(normal_videos)} normal video files")
    
    conn = sqlite3.connect(DB_PATH)
    # Get existing video_keys for generated_videos
    existing = set(
        row[0]
        for row in conn.execute(
            "SELECT video_key FROM videos WHERE dataset = 'generated_videos'"
        ).fetchall()
    )
    print(f"Existing generated_videos records: {len(existing)}")
    
    inserted = 0
    for video_path in normal_videos:
        video_key = video_key_from_path(video_path)
        if video_key in existing:
            continue
        # Insert new record
        video_id = video_path.stem  # e.g., R000001_normal
        title = extract_title(video_path)
        file_path = str(video_path)
        conn.execute(
            """
            INSERT INTO videos (video_key, site, video_id, dataset, title, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (video_key, "generated", video_id, "generated_videos", title, file_path),
        )
        inserted += 1
    
    conn.commit()
    conn.close()
    print(f"Inserted {inserted} new normal video records")


if __name__ == "__main__":
    main()
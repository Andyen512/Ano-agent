#!/usr/bin/env python3
"""Create a directory with symlinks to normal videos that lack model_predictions."""
import os
import sqlite3
from pathlib import Path

DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")
TARGET_DIR = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/missing_normal_videos")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    # Get file paths of normal videos without model_predictions
    cursor = conn.execute("""
        SELECT v.file_path
        FROM videos v
        WHERE v.dataset = 'generated_videos'
          AND v.video_id LIKE '%normal'
          AND v.file_path LIKE '/home/caiqingyuan%'
          AND NOT EXISTS (
              SELECT 1 FROM model_predictions mp WHERE mp.video_key = v.video_key
          )
    """)
    rows = cursor.fetchall()
    print(f"Found {len(rows)} normal videos without model_predictions")
    
    created = 0
    for (file_path,) in rows:
        src = Path(file_path)
        if not src.exists():
            print(f"Warning: source file does not exist: {src}")
            continue
        # Create a symlink with a unique name to avoid collisions
        # Use the original filename, but prefix with a hash of the parent directory to ensure uniqueness
        import hashlib
        rel_hash = hashlib.md5(str(src.parent).encode()).hexdigest()[:8]
        link_name = f"{rel_hash}_{src.name}"
        link_path = TARGET_DIR / link_name
        if link_path.exists():
            continue
        os.symlink(src, link_path)
        created += 1
    
    conn.close()
    print(f"Created {created} symlinks in {TARGET_DIR}")


if __name__ == "__main__":
    main()
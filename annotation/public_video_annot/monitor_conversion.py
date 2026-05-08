import os
import sqlite3
import time
from pathlib import Path

DB_PATH = "/data_4/liuyuan/lifebench/annotation/public_video_annot/data/app.db"
SRC_DIR = Path("/data_4/liuyuan/lifebench/data/public_data/labimag/dataset")

print("Monitoring AVI -> MP4 conversion...")
print(f"Source: {SRC_DIR}")
print(f"Database: {DB_PATH}")
print("-" * 50)

converted_total = 0

while True:
    avi_files = list(SRC_DIR.rglob("*.avi"))
    mp4_files = list(SRC_DIR.rglob("*.mp4"))
    
    print(f"[{time.strftime('%H:%M:%S')}] AVI remaining: {len(avi_files)}, MP4: {len(mp4_files)}")
    
    if len(avi_files) == 0:
        print("\nAll AVI files converted!")
        
        # Final database update
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET file_path = REPLACE(file_path, '.avi', '.mp4') WHERE file_path LIKE '%.avi'")
        remaining = cursor.execute("SELECT COUNT(*) FROM videos WHERE file_path LIKE '%.avi'").fetchone()[0]
        conn.commit()
        conn.close()
        
        print(f"Database updated. Remaining AVI in DB: {remaining}")
        break
    
    # Update database for already converted files
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE videos SET file_path = REPLACE(file_path, '.avi', '.mp4') WHERE file_path LIKE '%.avi'")
    updated = cursor.rowcount
    conn.commit()
    conn.close()
    
    if updated > 0:
        print(f"  -> Database updated: {updated} paths")
        converted_total += updated
    
    time.sleep(10)

print(f"\nTotal database entries updated: {converted_total}")

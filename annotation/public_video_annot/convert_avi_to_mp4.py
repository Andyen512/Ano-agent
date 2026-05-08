import os
import subprocess
from pathlib import Path

src_dir = Path("/data_4/liuyuan/lifebench/data/public_data/labimag/dataset")

avi_files = list(src_dir.rglob("*.avi"))
print(f"Found {len(avi_files)} AVI files")

converted = 0
skipped = 0

for avi_path in avi_files:
    mp4_path = avi_path.with_suffix(".mp4")
    if mp4_path.exists():
        print(f"Skip (exists): {mp4_path.name}")
        skipped += 1
        continue
    
    print(f"Converting: {avi_path}")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(avi_path),
            "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
            str(mp4_path)
        ], check=True, capture_output=True)
        converted += 1
        print(f"  -> Done: {mp4_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"  -> Error: {e.stderr.decode()[:200] if e.stderr else e}")

print(f"\nDone! Converted: {converted}, Skipped: {skipped}")

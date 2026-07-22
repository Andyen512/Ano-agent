#!/usr/bin/env python3
"""批量更新视频时长数据"""
import sqlite3
import subprocess
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "app.db"

def get_video_duration(file_path):
    """使用 ffprobe 获取视频时长（秒）"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            duration = data.get("format", {}).get("duration")
            if duration:
                return int(float(duration))
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
    return None

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 获取所有视频
    rows = c.execute("SELECT video_key, file_path FROM videos WHERE duration_seconds IS NULL OR duration_seconds = 0").fetchall()
    total = len(rows)
    print(f"需要更新 {total} 个视频的时长")
    
    updated = 0
    failed = 0
    for i, (video_key, file_path) in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"进度: {i + 1}/{total}")
        
        path = Path(file_path)
        if not path.exists():
            failed += 1
            continue
        
        duration = get_video_duration(file_path)
        if duration is not None:
            c.execute("UPDATE videos SET duration_seconds = ? WHERE video_key = ?", (duration, video_key))
            updated += 1
        else:
            failed += 1
    
    conn.commit()
    conn.close()
    
    print(f"\n完成！成功更新: {updated}, 失败: {failed}")

if __name__ == "__main__":
    main()

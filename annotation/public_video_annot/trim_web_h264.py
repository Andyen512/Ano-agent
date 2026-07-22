#!/usr/bin/env python3
"""批量裁切 web_h264 视频并把片段添加到数据库"""
import sqlite3
import subprocess
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "app.db"
PUBLIC_DATA_ROOT = Path("/data_4/liuyuan/lifebench/data/public_data")
BATCH_TRIM_OUTPUT = PUBLIC_DATA_ROOT / "seg_video"

def parse_trim_segments(trim_str):
    segments = []
    trim_str = trim_str.replace('，', ',').replace('；', ';').replace('[', '').replace(']', '').rstrip(';').rstrip('；').rstrip(',')
    for seg in trim_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split(",")
        if len(parts) == 2:
            try:
                start = float(parts[0].strip())
                end = float(parts[1].strip())
                if start < end:
                    segments.append((start, end))
            except ValueError:
                continue
    return segments

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 获取需要裁切的 web_h264 视频
    rows = c.execute("""
        SELECT a.video_key, a.annotator, a.trim_segments, v.file_path
        FROM human_annotations a
        JOIN videos v ON v.video_key = a.video_key
        WHERE a.trim_segments IS NOT NULL AND a.trim_segments != ''
        AND (a.trimmed IS NULL OR a.trimmed = 0)
        AND v.dataset = 'web_h264'
    """).fetchall()
    
    print(f"共 {len(rows)} 个视频需要裁切")
    
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        ffmpeg_exe = "ffmpeg"
    
    success_count = 0
    fail_count = 0
    new_videos = []
    
    for row in rows:
        video_key = row["video_key"]
        file_path = Path(row["file_path"]).resolve()
        trim_segments_str = row["trim_segments"]
        
        if not file_path.exists():
            print(f"跳过 {video_key}: 文件不存在")
            fail_count += 1
            continue
        
        segments = parse_trim_segments(trim_segments_str)
        if not segments:
            print(f"跳过 {video_key}: 无有效裁切段")
            fail_count += 1
            continue
        
        # 创建输出目录
        relative_path = file_path.relative_to(PUBLIC_DATA_ROOT.resolve())
        output_subdir = BATCH_TRIM_OUTPUT / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        stem = file_path.stem
        suffix = file_path.suffix
        
        # 裁切每个片段
        for i, (start, end) in enumerate(segments):
            output_name = f"{stem}_seg{i+1}_{start:.1f}-{end:.1f}{suffix}"
            output_path = output_subdir / output_name
            
            try:
                cmd = [
                    ffmpeg_exe, "-y", "-i", str(file_path),
                    "-ss", str(start), "-to", str(end),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
                
                # 记录新视频信息
                new_video_key = f"{relative_path.parent}/{output_name}"
                new_file_path = str(output_path)
                new_duration = int(end - start)
                new_videos.append((new_video_key, new_file_path, row["video_key"], new_duration))
                
                print(f"  裁切成功: {output_name}")
            except Exception as e:
                print(f"  裁切失败: {output_name} - {e}")
        
        # 标记原视频为已裁切
        c.execute("UPDATE human_annotations SET trimmed = 1 WHERE video_key = ? AND annotator = ?",
                  (video_key, row["annotator"]))
        success_count += 1
    
    conn.commit()
    
    # 把裁切后的视频添加到数据库
    print(f"\n添加 {len(new_videos)} 个新视频到数据库...")
    
    for new_video_key, new_file_path, parent_key, duration in new_videos:
        # 检查是否已存在
        existing = c.execute("SELECT video_key FROM videos WHERE video_key = ?", (new_video_key,)).fetchone()
        if existing:
            print(f"  已存在: {new_video_key}")
            continue
        
        # 获取原视频的信息
        parent = c.execute("SELECT site, video_id, dataset FROM videos WHERE video_key = ?", (parent_key,)).fetchone()
        
        c.execute("""
            INSERT INTO videos (video_key, site, video_id, dataset, file_path, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (new_video_key, parent["site"] if parent else "", parent["video_id"] if parent else "", "web_h264_segments", new_file_path, duration))
        
        print(f"  添加: {new_video_key}")
    
    conn.commit()
    conn.close()
    
    print(f"\n完成！裁切成功: {success_count}, 失败: {fail_count}, 新视频: {len(new_videos)}")

if __name__ == "__main__":
    main()

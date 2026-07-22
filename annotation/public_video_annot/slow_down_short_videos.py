#!/usr/bin/env python3
"""对小于5秒的视频进行慢放扩充"""
import subprocess
from pathlib import Path

SEG_DIR = Path("/data_4/liuyuan/lifebench/data/public_data/seg_video/web_h264")
MIN_DURATION = 5.0

def get_duration(file_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(file_path)],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except:
        return None

def slow_down_video(input_path, output_path, speed_factor):
    """慢放视频，speed_factor < 1 表示慢放"""
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-filter:v', f'setpts={1/speed_factor}*PTS',
        '-filter:a', f'atempo={speed_factor}',
        '-an',  # 不处理音频（裁切视频通常没音频）
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0

def main():
    short_videos = []
    
    for f in SEG_DIR.glob('*.mp4'):
        duration = get_duration(f)
        if duration and duration < MIN_DURATION:
            short_videos.append((f, duration))
    
    print(f"找到 {len(short_videos)} 个小于 {MIN_DURATION} 秒的视频")
    
    success = 0
    failed = 0
    
    for file_path, duration in sorted(short_videos, key=lambda x: x[1]):
        # 计算慢放倍率，目标是5秒
        speed_factor = duration / MIN_DURATION
        
        tmp_path = file_path.with_suffix('.tmp.mp4')
        
        if slow_down_video(file_path, tmp_path, speed_factor):
            # 检查输出时长
            new_duration = get_duration(tmp_path)
            if new_duration and new_duration >= MIN_DURATION - 0.5:
                # 替换原文件
                tmp_path.rename(file_path)
                print(f"  {file_path.name}: {duration:.1f}s -> {new_duration:.1f}s")
                success += 1
            else:
                tmp_path.unlink(missing_ok=True)
                print(f"  {file_path.name}: 处理后时长不足 ({new_duration})")
                failed += 1
        else:
            tmp_path.unlink(missing_ok=True)
            print(f"  {file_path.name}: 处理失败")
            failed += 1
    
    print(f"\n完成！成功: {success}, 失败: {failed}")

if __name__ == "__main__":
    main()

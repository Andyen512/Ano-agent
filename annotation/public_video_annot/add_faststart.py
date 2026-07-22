#!/usr/bin/env python3
"""给视频添加 faststart，安全地原地替换"""
import subprocess
import shutil
import struct
import tempfile
import os
from pathlib import Path

DATASETS = [
    "/data_4/liuyuan/lifebench/data/public_data/Single_user_simple_event",
    "/data_4/liuyuan/lifebench/data/public_data/Multiple_users_compound_event",
]

def needs_faststart(file_path):
    found_mdat = False
    with open(file_path, 'rb') as f:
        for _ in range(10):
            pos = f.tell()
            header = f.read(8)
            if len(header) < 8:
                break
            size = struct.unpack('>I', header[:4])[0]
            box_type = header[4:8].decode('ascii', errors='replace')
            if box_type == 'mdat':
                found_mdat = True
            elif box_type == 'moov':
                return found_mdat
            if size >= 8:
                f.seek(pos + size)
            else:
                break
    return True

def add_faststart(file_path):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(tmp_fd)
    
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(file_path),
            '-c', 'copy', '-movflags', '+faststart',
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            return False
        
        tmp_size = Path(tmp_path).stat().st_size
        orig_size = file_path.stat().st_size
        if tmp_size < orig_size * 0.5 or tmp_size > orig_size * 1.5:
            return False
        
        if needs_faststart(Path(tmp_path)):
            return False
        
        backup_path = str(file_path) + '.bak'
        shutil.copy2(file_path, backup_path)
        shutil.move(tmp_path, str(file_path))
        
        if not needs_faststart(file_path):
            Path(backup_path).unlink()
            return True
        else:
            shutil.move(backup_path, str(file_path))
            return False
    except Exception as e:
        return False
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()

def main():
    all_files = []
    for ds in DATASETS:
        for ext in ('*.mp4', '*.MP4'):
            all_files.extend(Path(ds).rglob(ext))
    
    print(f"共找到 {len(all_files)} 个视频文件")
    
    need_fix = [f for f in all_files if needs_faststart(f)]
    
    print(f"需要添加 faststart: {len(need_fix)} 个")
    print(f"已含 faststart: {len(all_files) - len(need_fix)} 个")
    
    if not need_fix:
        print("无需处理")
        return
    
    success = 0
    failed = 0
    for i, f in enumerate(need_fix):
        if (i + 1) % 100 == 0 or (i + 1) == len(need_fix):
            print(f"进度: {i + 1}/{len(need_fix)} (成功:{success}, 失败:{failed})")
        if add_faststart(f):
            success += 1
        else:
            failed += 1
    
    print(f"\n完成！成功: {success}, 失败: {failed}")

if __name__ == "__main__":
    main()

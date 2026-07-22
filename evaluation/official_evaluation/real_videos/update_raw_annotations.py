#!/usr/bin/env python3
"""
更新 public_data_release/annotations/real_videos/ 下所有 perception_gt.json：
- normal 视频的 risk_type 置空
- abnormal/risk_only 从 DB 回填 level3_risk_type
"""
import json, sqlite3, os
from pathlib import Path

ANN_ROOT = Path("/home/caiqingyuan/code/lifebench/data/public_data_release/annotations/real_videos")
DB_PATH = "/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db"

def load_db_risk_types():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT video_key, level3_risk_type FROM human_annotations")
    db = {}
    for key, rt in c.fetchall():
        db[key] = rt or ""
        # also store without extension for matching
        for ext in (".mp4", ".avi", ".mov"):
            if key.endswith(ext):
                db[key[: -len(ext)]] = rt or ""
                break
    conn.close()
    return db


def main():
    db = load_db_risk_types()
    print(f"DB 加载 {len(db)} 条记录")

    updated_files = 0
    updated_entries_normal = 0
    updated_entries_risk = 0
    missing_db = 0

    for root, dirs, files in os.walk(ANN_ROOT):
        for fname in files:
            if fname != "perception_gt.json":
                continue
            path = Path(root) / fname
            data = json.loads(path.read_text(encoding="utf-8"))
            changed = False

            for item in data:
                vid = item["video_id"]
                status = item.get("risk_status", "")

                if status == "normal":
                    if item.get("risk_type", "") != "":
                        item["risk_type"] = ""
                        changed = True
                        updated_entries_normal += 1
                elif status in ("abnormal", "risk_only"):
                    rt = db.get(vid, "")
                    # also try with extensions
                    if not rt:
                        for ext in (".mp4", ".avi", ".mov"):
                            if db.get(vid + ext, ""):
                                rt = db[vid + ext]
                                break
                    if not rt and item.get("risk_type", "") == "":
                        missing_db += 1
                    if rt and item.get("risk_type", "") != rt:
                        item["risk_type"] = rt
                        changed = True
                        updated_entries_risk += 1

            if changed:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                updated_files += 1
                print(f"  更新: {path.relative_to(ANN_ROOT)}")

    print(f"\n完成: {updated_files} 个文件更新")
    print(f"  normal risk_type置空: {updated_entries_normal} 条")
    print(f"  abnormal/risk_only 回填: {updated_entries_risk} 条")
    print(f"  DB中未找到: {missing_db} 条")


if __name__ == "__main__":
    main()

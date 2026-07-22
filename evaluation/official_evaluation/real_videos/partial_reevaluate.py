#!/usr/bin/env python3
"""
部分重新评测：只重跑 GT label 发生变化的视频，然后与旧结果合并。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = Path(__file__).resolve().parent
ANN_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "annotations" / "real_videos"
GT_TXT = PROJECT_ROOT / "data" / "annotation" / "real_gt_expanded_release.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "real_videos"
PER_VIDEO_JSON = OUTPUT_DIR / "per_video_scores.json"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR.parent.parent))

# ── Step 1: 找出 label 变化的 video_id ──────────────────────────
print("Step 1: 找出 label 变化的 video_id ...")

new_status = {}
with open(GT_TXT) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            new_status[parts[0]] = parts[2]

with open(PER_VIDEO_JSON) as f:
    old_pv = json.load(f)

old_status = {}
for item in old_pv:
    vid = item["video_id"]
    if vid not in old_status:
        old_status[vid] = item["ground_truth"]["risk_status"]

changed_vids = set()
for vid, new_s in new_status.items():
    old_s = old_status.get(vid)
    if old_s and old_s != new_s:
        changed_vids.add(vid)

# 也包含之前从未被评测过的视频
evaluated_vids = set(old_status.keys())
all_vids = set(new_status.keys())
missing_vids = all_vids - evaluated_vids

# 从未评测的视频没有 prediction，跳过
target_vids = changed_vids
print(f"  已评测视频: {len(evaluated_vids)}")
print(f"  新标注视频: {len(all_vids)}")
print(f"  label 变化: {len(changed_vids)}")
print(f"  从未评测(跳过,无prediction): {len(missing_vids)}")
print(f"  需重评: {len(target_vids)}")

# ── Step 2: 过滤 GT JSON，只保留目标视频 ────────────────────────
print("\nStep 2: 过滤 GT JSON ...")

filtered_root = OUTPUT_DIR / "partial_reeval_gt"
if filtered_root.exists():
    shutil.rmtree(filtered_root)

# 读取所有 JSON，按 video_id 索引
all_gt = {}
for root, dirs, files in os.walk(ANN_ROOT):
    for fname in files:
        if not fname.endswith(".json"):
            continue
        fpath = Path(root) / fname
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            vid = entry.get("video_id", "")
            task = fname.replace("_gt.json", "")
            if vid not in all_gt:
                all_gt[vid] = {}
            all_gt[vid][task] = entry

# 过滤并写入临时 GT 目录
for vid in target_vids:
    if vid not in all_gt:
        continue
    vid_data = all_gt[vid]
    correct_label = new_status.get(vid, "")
    for task, entry in vid_data.items():
        entry["risk_status"] = correct_label
        dir_path = filtered_root / correct_label / "youtube2"
        dir_path.mkdir(parents=True, exist_ok=True)
        json_path = dir_path / f"{task}_gt.json"
        if json_path.exists():
            with open(json_path) as fh:
                existing = json.load(fh)
        else:
            existing = []
        existing.append(entry)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)

# 统计写了多少
total_filtered = sum(len(files) for _, _, files in os.walk(filtered_root))
print(f"  过滤后 GT 文件写入: {filtered_root}")

# ── Step 3: 跑 evaluate.py ──────────────────────────────────────
print("\nStep 3: 用过滤后的 GT 跑 evaluation ...")

tmp_output = OUTPUT_DIR / "partial_reeval_output"
if tmp_output.exists():
    shutil.rmtree(tmp_output)

cmd = [
    sys.executable,
    str(EVAL_DIR / "evaluate.py"),
    "--gt-dir", str(filtered_root),
    "--output-dir", str(tmp_output),
    "--judge-mode", "openai",
    "--judge-model", "qwen2.5:7b",
    "--judge-base-url", "http://localhost:11434/v1",
    "--judge-api-key", "ollama",
    "--workers", "4",
]
print(f"  运行: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
if result.returncode != 0:
    print(f"  evaluate.py 失败 (code={result.returncode})")
    sys.exit(1)

# ── Step 4: 合并结果 ────────────────────────────────────────────
print("\nStep 4: 合并新旧结果 ...")

# 读取新结果
new_pv_path = tmp_output / "per_video_scores.json"
with open(new_pv_path) as f:
    new_pv = json.load(f)

# 构建 {video_id: {model_id: entry}} 索引（新结果）
new_index = {}
for item in new_pv:
    vid = item["video_id"]
    mid = item["model_id"]
    new_index[(vid, mid)] = item

# 替换旧结果中对应 video_id 的条目
merged = []
replaced = 0
for item in old_pv:
    vid = item["video_id"]
    mid = item["model_id"]
    key = (vid, mid)
    if key in new_index:
        merged.append(new_index.pop(key))
        replaced += 1
    else:
        merged.append(item)

# 添加全新条目（之前完全没评测过的视频 - 这些可能没有 prediction）
for key, item in list(new_index.items()):
    merged.append(item)
    replaced += 1

merged.sort(key=lambda r: (r.get("model_id", ""), r.get("video_id", "")))

print(f"  替换/新增: {replaced} 条")
print(f"  合并后总数: {len(merged)} (原: {len(old_pv)})")

# 写回
with open(PER_VIDEO_JSON, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# ── Step 5: 重新生成 summary ────────────────────────────────────
print("\nStep 5: 重新生成 summary ...")

# 复用 evaluate.py 中的 summarize_by_model
from evaluation.official_evaluation.real_videos.evaluate import summarize_by_model, format_json, now_iso, write_json_atomic

model_summary = summarize_by_model(merged)

summary_payload = {
    "generated_at": now_iso(),
    "judge_mode": "mixed_openai",
    "gt_dir": str(ANN_ROOT),
    "predictions_root": str(PROJECT_ROOT / "data" / "public_data_release" / "prediction" / "real_videos"),
    "total_gt_entries": len(all_vids),
    "total_predictions_evaluated": len(merged),
    "model_summary": model_summary,
}
write_json_atomic(OUTPUT_DIR / "summary.json", summary_payload)

# ── 清理 ─────────────────────────────────────────────────────────
print("\n清理临时文件 ...")
shutil.rmtree(filtered_root, ignore_errors=True)
shutil.rmtree(tmp_output, ignore_errors=True)

print("\n完成！")

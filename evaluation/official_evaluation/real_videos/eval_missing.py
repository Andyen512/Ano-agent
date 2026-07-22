#!/usr/bin/env python3
"""
只评测之前因 discover 问题未匹配到的 prediction，然后与旧结果合并。
"""
import json, os, shutil, sys, tempfile, subprocess
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = Path(__file__).resolve().parent
ANN_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "annotations" / "real_videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "real_videos"
PER_VIDEO_JSON = OUTPUT_DIR / "per_video_scores.json"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR.parent.parent))

from evaluation.official_evaluation.real_videos.common_real import (
    discover_prediction_files, match_prediction_to_gt,
    _build_basename_index, load_real_video_gt,
)

# ── Step 1: 找到每个模型未评测的 prediction ──────────────────────
print("Step 1: 定位未评测的 prediction ...")

gt_by_video = load_real_video_gt(ANN_ROOT)
gt_basename_index = _build_basename_index(gt_by_video)
print(f"  GT: {len(gt_by_video)} 个视频")

with open(PER_VIDEO_JSON) as f:
    old_pv = json.load(f)

evaluated = defaultdict(set)
for item in old_pv:
    evaluated[item["model_id"]].add(item["video_id"])

payloads = discover_prediction_files()
print(f"  prediction: {len(payloads)} 个文件")

# 过滤出未评测的
new_payloads = []
for p in payloads:
    model_id = p.get("model_id", p.get("__source_model_dir__", "unknown"))
    source_path = p.get("__source_path__", "")
    filename_stem = Path(source_path).stem if source_path else ""
    vp = p.get("video_path", "")

    gt_vid = match_prediction_to_gt(filename_stem, vp, gt_by_video, gt_basename_index)
    if gt_vid and gt_vid not in evaluated.get(model_id, set()):
        new_payloads.append(p)

print(f"  新增待评测: {len(new_payloads)} 条")
if not new_payloads:
    print("  没有新增，退出")
    sys.exit(0)

# ── Step 2: 创建临时 prediction 目录 ────────────────────────────
print("\nStep 2: 创建临时 prediction 目录 ...")

tmp_pred_root = OUTPUT_DIR / "eval_missing_pred"
if tmp_pred_root.exists():
    shutil.rmtree(tmp_pred_root)

# 用 model_short_name 作为模型目录名（与 evaluate.py 兼容）
model_short_map = {}
for p in new_payloads:
    mid = p.get("model_id", p.get("__source_model_dir__", "unknown"))
    model_short_map[mid] = p.get("__source_model_dir__", mid.split("/")[-1])

model_new_videos = defaultdict(set)
for p in new_payloads:
    mid = p.get("model_id", p.get("__source_model_dir__", "unknown"))
    src = p["__source_path__"]
    model_short = model_short_map[mid]
    dst_dir = tmp_pred_root / model_short / "ts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(src).name
    if not dst.exists():
        shutil.copy2(src, dst)
    model_new_videos[mid].add(p["__source_model_id__" if "__source_model_id__" in p else "model_id"])

# 收集所有新的 GT video_id
new_gt_vids = set()
for p in new_payloads:
    mid = p.get("model_id", p.get("__source_model_dir__", "unknown"))
    source_path = p.get("__source_path__", "")
    filename_stem = Path(source_path).stem if source_path else ""
    vp = p.get("video_path", "")
    gt_vid = match_prediction_to_gt(filename_stem, vp, gt_by_video, gt_basename_index)
    if gt_vid:
        new_gt_vids.add(gt_vid)

# ── Step 3: 创建临时 GT 目录 ────────────────────────────────────
print("Step 3: 创建临时 GT 目录 ...")

tmp_gt_root = OUTPUT_DIR / "eval_missing_gt"
if tmp_gt_root.exists():
    shutil.rmtree(tmp_gt_root)

# 从原始 GT 中过滤出新视频的条目
all_gt_entries = {}
for root, dirs, files in os.walk(ANN_ROOT):
    for fname in files:
        if not fname.endswith("_gt.json"):
            continue
        task = fname.replace("_gt.json", "")
        fpath = Path(root) / fname
        with open(fpath, encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            vid = entry.get("video_id", "")
            if vid not in all_gt_entries:
                all_gt_entries[vid] = {}
            all_gt_entries[vid][task] = entry

# 获取视频的正确 label
new_status = {}
with open(PROJECT_ROOT / "data" / "annotation" / "real_gt_expanded_release.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            new_status[parts[0]] = parts[2]

for vid in new_gt_vids:
    if vid not in all_gt_entries:
        continue
    correct_label = new_status.get(vid, "")
    for task, entry in all_gt_entries[vid].items():
        entry["risk_status"] = correct_label
        dir_path = tmp_gt_root / correct_label / "youtube2"
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

# ── Step 4: 跑 evaluate.py ──────────────────────────────────────
print("Step 4: 用临时 GT + prediction 跑 evaluation ...")

tmp_output = OUTPUT_DIR / "eval_missing_output"
if tmp_output.exists():
    shutil.rmtree(tmp_output)

cmd = [
    sys.executable,
    str(EVAL_DIR / "evaluate.py"),
    "--gt-dir", str(tmp_gt_root),
    "--predictions-root", str(tmp_pred_root),
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

# ── Step 5: 合并结果 ────────────────────────────────────────────
print("\nStep 5: 合并新旧结果 ...")

with open(tmp_output / "per_video_scores.json") as f:
    new_pv = json.load(f)

print(f"  新结果: {len(new_pv)} 条")

# 按 (video_id, model_id) 索引新结果
new_index = {}
for item in new_pv:
    new_index[(item["video_id"], item["model_id"])] = item

# 替换/追加
merged = []
replaced = 0
for item in old_pv:
    key = (item["video_id"], item["model_id"])
    if key in new_index:
        merged.append(new_index.pop(key))
        replaced += 1
    else:
        merged.append(item)

# 添加新条目
for key, item in new_index.items():
    merged.append(item)
    replaced += 1

merged.sort(key=lambda r: (r.get("model_id", ""), r.get("video_id", "")))
print(f"  替换/新增: {replaced} 条")
print(f"  合并后总数: {len(merged)} (原: {len(old_pv)})")

with open(PER_VIDEO_JSON, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

# ── Step 6: 重新生成 summary ────────────────────────────────────
print("\nStep 6: 重新生成 summary ...")

from evaluation.official_evaluation.real_videos.evaluate import summarize_by_model, format_json, now_iso, write_json_atomic

model_summary = summarize_by_model(merged)
summary_payload = {
    "generated_at": now_iso(),
    "judge_mode": "mixed_openai",
    "gt_dir": str(ANN_ROOT),
    "predictions_root": str(PROJECT_ROOT / "data" / "public_data_release" / "prediction" / "real_videos"),
    "total_gt_entries": len(gt_by_video),
    "total_predictions_evaluated": len(merged),
    "model_summary": model_summary,
}
write_json_atomic(OUTPUT_DIR / "summary.json", summary_payload)

# ── 清理 ─────────────────────────────────────────────────────────
print("\n清理临时文件 ...")
shutil.rmtree(tmp_pred_root, ignore_errors=True)
shutil.rmtree(tmp_gt_root, ignore_errors=True)
shutil.rmtree(tmp_output, ignore_errors=True)

# ── 更新 FINAL_RESULTS.md ──────────────────────────────────────
print("更新 FINAL_RESULTS.md ...")
subprocess.run([sys.executable, str(EVAL_DIR / "gen_final_results_md.py")], cwd=str(PROJECT_ROOT))

print("\n完成！")

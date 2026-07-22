import sqlite3
import re
import os
import json
from pathlib import Path
from collections import defaultdict

APP_DIR = Path(__file__).resolve().parent
PROMPT_FILE = Path("/home/caiqingyuan/code/lifebench/data/annotation/gen_prompt_generated_all.txt")
OUTPUT_FILE = Path("/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release.txt")
DB_PATH = APP_DIR / "data" / "app.db"
VIDEO_ROOT = Path("/home/caiqingyuan/code/lifebench/data/generated_videos_20260703")
LIFEBENCH_ROOT = Path("/home/caiqingyuan/code/lifebench")

# Model name mapping: old DB model names -> new dataset model names
MODEL_MAP = {
    "veo3.1-fast": "veo3.1",
    "grok-imagine-video": "grok",
    "sora-2-chan1": "sora2",
}

# 1. Load prompt file into lookup dict
prompt_map = {}
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rid_type = parts[0]
        m = re.match(r"^(?:.*_)?(R\d+)_(.+)$", rid_type)
        if not m:
            continue
        r_number = m.group(1)
        vtype = m.group(2)
        prompt_map[(r_number, vtype)] = {
            "raw_id": rid_type,
            "category": parts[1] if len(parts) > 1 else "",
            "description": parts[2] if len(parts) > 2 else "",
            "solutions": parts[3] if len(parts) > 3 else "",
        }

print(f"Loaded {len(prompt_map)} prompt entries")

# 2. Load DB annotations: (R_number, vtype, old_model_name) -> risk decision
conn = sqlite3.connect(str(DB_PATH), timeout=30)
conn.row_factory = sqlite3.Row

db_rows = conn.execute("""
    WITH latest_annot AS (
      SELECT h.* FROM human_annotations h
      INNER JOIN (
        SELECT video_key, MAX(updated_at) AS max_ts FROM human_annotations GROUP BY video_key
      ) m ON h.video_key = m.video_key AND h.updated_at = m.max_ts
    )
    SELECT v.video_key, la.risk, la.annotator
    FROM videos v
    JOIN latest_annot la ON la.video_key = v.video_key
    WHERE v.dataset = 'generated_videos'
""").fetchall()

# Build: (R_number, vtype, new_model) -> {"risk": ..., "annotator": ...}
annot_map = {}
drop_model = set()
for row in db_rows:
    vk = row["video_key"]
    # Extract R_number, vtype, model from old path
    # Format: generated_videos/{scene}/{subject}/{risk_type}/{desc}/{old_model}/{type}/{RID}_{type}.mp4
    m_vk = re.search(r"/([^/]+)/(abnormal|risk_only|normal)/(?:.*/)?(R\d+)_(abnormal|risk_only|normal)\.mp4$", vk)
    if not m_vk:
        # Try simpler pattern
        m_vk = re.search(r"/([^/]+)/([^/]+)/(R\d+)_(abnormal|risk_only|normal)\.mp4$", vk)
        if not m_vk:
            continue
        old_model = m_vk.group(1)
        vtype = m_vk.group(2)
        r_number = m_vk.group(3)
    else:
        old_model = m_vk.group(1)
        vtype = m_vk.group(2)
        r_number = m_vk.group(3)
    
    new_model = MODEL_MAP.get(old_model)
    if new_model is None:
        continue  # skip models not in new dataset (e.g., sora-2-chan1 not mapped to anything)
    
    key = (r_number, vtype, new_model)
    annot_map[key] = {
        "risk": row["risk"],
        "annotator": row["annotator"],
    }

print(f"Loaded {len(annot_map)} DB annotations for {len(set((k[0],k[1]) for k in annot_map))} (R_number, type) pairs with known models")

conn.close()

# Also build a fallback: (R_number, vtype) -> majority risk across ALL old models
fallback_map = {}
for row in db_rows:
    vk = row["video_key"]
    m_vk = re.search(r"/(R\d+)_(abnormal|risk_only|normal)\.mp4$", vk)
    if not m_vk:
        continue
    r_number = m_vk.group(1)
    vtype = m_vk.group(2)
    key = (r_number, vtype)
    if key not in fallback_map:
        fallback_map[key] = []
    fallback_map[key].append(row["risk"])

from collections import Counter
for key, risks in fallback_map.items():
    c = Counter(risks)
    fallback_map[key] = c.most_common(1)[0][0]

# 3. Scan all video files in new dataset
video_files = []
for ext in ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm'):
    for f in VIDEO_ROOT.rglob(ext):
        video_files.append(f)

print(f"Found {len(video_files)} video files in dataset")

# 4. Process each video file
output_lines = []
stats = {
    "prompt_match": 0, "no_desc": 0,
    "annot_model_match": 0, "annot_fallback": 0, "unannotated": 0,
}

for vf in sorted(video_files):
    rel_to_lifebench = vf.relative_to(LIFEBENCH_ROOT)
    vf_str = str(rel_to_lifebench)
    
    # Extract R_number and video_type from filename
    m = re.search(r"(R\d+)_(abnormal|risk_only|normal)\.(mp4|avi|mkv|mov|webm)$", vf.name, re.IGNORECASE)
    if not m:
        continue
    
    r_number = m.group(1)
    vtype = m.group(2)
    
    # Extract path components
    # generated_videos_20260703/generated_videos/{scene}/{subject}/{risk_type}/{desc}/{model}/{type}/{filename}
    parts = vf_str.split("/")
    idx_gv = None
    for i, p in enumerate(parts):
        if p == "generated_videos":
            idx_gv = i
            break
    if idx_gv is not None and len(parts) >= idx_gv + 7:
        scene = parts[idx_gv + 1]
        subject = parts[idx_gv + 2]
        risk_type = parts[idx_gv + 3]
        desc = parts[idx_gv + 4]
        model = parts[idx_gv + 5]
    else:
        scene = subject = risk_type = desc = model = ""
    
    # Get description and solutions from prompt file
    prompt_key = (r_number, vtype)
    if prompt_key in prompt_map:
        pm = prompt_map[prompt_key]
        description = pm["description"]
        solutions = pm["solutions"]
        stats["prompt_match"] += 1
    else:
        description = ""
        solutions = ""
        stats["no_desc"] += 1
    
    # All videos in this curated dataset are kept
    risk_decision = "Yes"
    annotators = ""
    
    line = "\t".join([
        vf_str,
        r_number,
        vtype,
        scene,
        subject,
        risk_type,
        model,
        desc,
        description,
        solutions,
        risk_decision,
        annotators,
    ])
    output_lines.append(line)

print(f"\nStats:")
print(f"  prompt_match: {stats['prompt_match']}")
print(f"  no_desc: {stats['no_desc']}")
print(f"  annot_model_match: {stats['annot_model_match']}")
print(f"  annot_fallback: {stats['annot_fallback']}")
print(f"  unannotated: {stats['unannotated']}")
print(f"  Total output: {len(output_lines)}")

# 5. Write output
OUTPUT_DIR = OUTPUT_FILE.parent
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\t".join([
        "video_path", "R_number", "video_type", "scene", "subject",
        "risk_type", "model", "path_desc", "description", "solutions",
        "risk_decision", "annotators"
    ]) + "\n")
    for line in output_lines:
        f.write(line + "\n")

print(f"Output written to {OUTPUT_FILE}")

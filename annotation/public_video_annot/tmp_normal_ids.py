import json
from pathlib import Path

json_path = Path("/home/caiqingyuan/code/lifebench/outputs/evaluation/generated_videos/per_video_scores.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

normal_ids = set()
for entry in data:
    vid = entry.get('video_id', '')
    if vid.endswith('_normal'):
        normal_ids.add(vid)

print(f"Found {len(normal_ids)} distinct normal video_ids")
for vid in sorted(normal_ids)[:20]:
    print(vid)
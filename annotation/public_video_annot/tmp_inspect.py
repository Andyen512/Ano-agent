import json
from pathlib import Path

json_path = Path("/home/caiqingyuan/code/lifebench/outputs/evaluation/generated_videos/per_video_scores.json")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find first normal video entry
for entry in data:
    vid = entry.get('video_id', '')
    if vid.endswith('_normal'):
        print(json.dumps(entry, indent=2, ensure_ascii=False))
        break
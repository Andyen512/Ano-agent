#!/usr/bin/env python3
"""Import evaluation results for normal videos into model_predictions table."""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")
EVAL_JSON = Path("/home/caiqingyuan/code/lifebench/outputs/evaluation/generated_videos/per_video_scores.json")


def main():
    # Load evaluation data
    with open(EVAL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} evaluation entries")
    
    # Group by video_id (extract R*_normal pattern)
    import re
    grouped = defaultdict(list)
    for entry in data:
        vid = entry.get('video_id', '')
        if vid.endswith('_normal'):
            # Extract R*_normal part
            match = re.search(r'(R\d+_normal)', vid)
            if match:
                real_vid = match.group(1)
                grouped[real_vid].append(entry)
    print(f"Found {len(grouped)} normal videos with evaluation results")
    
    conn = sqlite3.connect(DB_PATH)
    # Get mapping from video_id to video_key for normal videos
    cursor = conn.execute(
        "SELECT video_id, video_key FROM videos WHERE dataset = 'generated_videos' AND video_id LIKE '%normal'"
    )
    id_to_key = dict(cursor.fetchall())
    print(f"Found {len(id_to_key)} normal videos in database")
    
    # Check existing model_predictions
    existing = set(
        row[0]
        for row in conn.execute(
            "SELECT video_key FROM model_predictions WHERE video_key LIKE 'generated_videos/%normal%'"
        ).fetchall()
    )
    print(f"Existing model_predictions for normal videos: {len(existing)}")
    
    inserted = 0
    for video_id, entries in grouped.items():
        video_key = id_to_key.get(video_id)
        if not video_key:
            print(f"Warning: video_id {video_id} not found in videos table")
            continue
        if video_key in existing:
            continue
        
        # Aggregate model predictions
        agents = []
        for entry in entries:
            model_id = entry.get('model_id', '')
            backend = entry.get('backend', '')
            structured = entry.get('structured_prediction', {})
            summary = structured.get('summary', {})
            metrics = entry.get('metrics', {})
            overall_score = metrics.get('overall_score', 0)
            
            agents.append({
                'model_id': model_id,
                'backend': backend,
                'risk_status': summary.get('risk_status', ''),
                'risk_type': summary.get('risk_type', ''),
                'video_description': summary.get('video_description', ''),
                'risk_description': summary.get('risk_description', ''),
                'solution': summary.get('solution', ''),
                'time_spans': summary.get('time_spans', []),
                'overall_score': overall_score,
                'metrics': metrics,
            })
        
        # Determine risk label (for normal videos, risk should be "No")
        risk = "No"
        # Extract scene, subject, risk_type from video_key path
        parts = video_key.split('/')
        # generated_videos/scene/subject/risk_type/description/model/normal/video.mp4
        level1_scene = parts[1] if len(parts) > 1 else ""
        level2_subject = parts[2] if len(parts) > 2 else ""
        level3_risk_type = parts[3] if len(parts) > 3 else ""
        description = parts[4] if len(parts) > 4 else ""
        
        # Use the first agent's video_description as description_text
        description_text = agents[0].get('video_description', '') if agents else ""
        
        # Insert into model_predictions
        conn.execute(
            """
            INSERT INTO model_predictions (
                video_key, source_output_root, summary_json, risk, level1_scene, level2_subject,
                level3_risk_type, description, risk_localization, solution_for_person,
                solution_for_hazard_source, solution_prevent_recurrence, yes_count, no_count,
                votes_cast, complete, majority_json, agents_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                video_key,
                str(EVAL_JSON.parent),
                "",
                risk,
                level1_scene,
                level2_subject,
                level3_risk_type,
                description_text,
                "None",
                "None",
                "None",
                "None",
                len(agents),
                0,
                len(agents),
                1,
                "{}",
                json.dumps(agents, ensure_ascii=False),
            ),
        )
        inserted += 1
    
    conn.commit()
    conn.close()
    print(f"Inserted {inserted} model_predictions records for normal videos")


if __name__ == "__main__":
    main()
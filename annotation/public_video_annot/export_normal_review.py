#!/usr/bin/env python3
"""Export normal videos with scores to CSV for manual review."""
import csv
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db")
OUTPUT_CSV = Path("/home/caiqingyuan/code/lifebench/annotation/public_video_annot/normal_videos_review.csv")


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT v.video_key, v.video_id, v.title, v.file_path,
               mp.risk, mp.level1_scene, mp.level2_subject, mp.level3_risk_type,
               mp.description, mp.agents_json
        FROM videos v
        LEFT JOIN model_predictions mp ON v.video_key = mp.video_key
        WHERE v.dataset = 'generated_videos' AND v.video_id LIKE '%normal'
        ORDER BY v.video_key
    """)
    
    rows = []
    for row in cursor:
        video_key, video_id, title, file_path, risk, scene, subject, risk_type, description, agents_json = row
        # Parse agents_json to get average overall score
        avg_score = 0
        if agents_json:
            try:
                agents = json.loads(agents_json)
                if isinstance(agents, list) and agents:
                    scores = [a.get('overall_score', 0) for a in agents if isinstance(a, dict)]
                    avg_score = sum(scores) / len(scores) if scores else 0
            except json.JSONDecodeError:
                pass
        rows.append({
            'video_key': video_key,
            'video_id': video_id,
            'title': title,
            'file_path': file_path,
            'risk': risk,
            'scene': scene,
            'subject': subject,
            'risk_type': risk_type,
            'description': description,
            'avg_score': round(avg_score, 2),
            'keep': '',  # For manual decision
        })
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Exported {len(rows)} normal videos to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
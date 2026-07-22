#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

EVAL_ROOTS = [
    Path("/data/caiqingyuan/Dataset/lifebench/outputs/normal_video_inference"),
    Path("/data/caiqingyuan/Dataset/lifebench/outputs/missing_score_inference"),
]
DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"


def load_all_results():
    results_by_video = defaultdict(list)
    for eval_root in EVAL_ROOTS:
        if not eval_root.exists():
            continue
        for model_dir in eval_root.iterdir():
            if not model_dir.is_dir() or model_dir.name in ("logs", "status", "summary.json"):
                continue
            for ts_dir in model_dir.iterdir():
                if not ts_dir.is_dir():
                    continue
                for jf in ts_dir.glob("*.json"):
                    try:
                        data = json.loads(jf.read_text())
                    except Exception:
                        continue
                    parsed = data.get("parsed_response") or {}
                    if not parsed:
                        continue
                    video_path = data.get("video_path", "")
                    if not video_path:
                        continue
                    try:
                        rel = Path(video_path).relative_to(Path("/data_4/liuyuan/lifebench/data/generated_videos"))
                        video_key = f"generated_videos/{rel}"
                    except ValueError:
                        try:
                            rel = Path(video_path).relative_to(Path("/data/caiqingyuan/Dataset/lifebench/data/generated_videos"))
                            video_key = f"generated_videos/{rel}"
                        except ValueError:
                            video_key = f"generated_videos/{Path(video_path).name}"
                    results_by_video[video_key].append({
                        "model_id": data.get("model_id", ""),
                        "backend": data.get("backend", ""),
                        "risk_status": parsed.get("risk_status", "normal"),
                        "risk_type": parsed.get("risk_type", ""),
                        "risk_description": parsed.get("risk_description", ""),
                        "video_description": parsed.get("video_description", ""),
                        "time_spans": parsed.get("time_spans", []),
                        "solution": parsed.get("solution", ""),
                    })
    return results_by_video


def compute_majority(results):
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["risk_status"]] += 1

    total = len(results)
    risk_count = status_counts.get("risk_only", 0) + status_counts.get("abnormal", 0)
    normal_count = status_counts.get("normal", 0)

    risk = "Yes" if risk_count > normal_count else "No"
    score = round(risk_count / total * 10, 2) if total > 0 else 0

    majority_vote = {
        "complete": True,
        "votes_cast": total,
        "risk_votes": {
            "normal": status_counts.get("normal", 0),
            "risk_only": status_counts.get("risk_only", 0),
            "abnormal": status_counts.get("abnormal", 0),
        },
        "final_decision": "accept" if risk == "Yes" else "reject",
        "avg_scores": {
            "avg_total_score": score,
        },
    }
    return risk, majority_vote


def main():
    results_by_video = load_all_results()
    print(f"Found results for {len(results_by_video)} videos")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    updated = 0
    skipped = 0

    for video_key, results in results_by_video.items():
        existing = conn.execute(
            "SELECT video_key FROM videos WHERE video_key = ?", (video_key,)
        ).fetchone()
        if not existing:
            skipped += 1
            continue

        risk, majority_vote = compute_majority(results)

        agents_data = {
            "source": "normal_video_inference",
            "avg_scores": majority_vote["avg_scores"],
            "model_decisions": [
                {"model": r["model_id"], "risk_status": r["risk_status"]}
                for r in results
            ],
            "risk_votes": majority_vote["risk_votes"],
        }

        conn.execute(
            """
            INSERT INTO model_predictions (
                video_key, source_output_root, summary_json, risk,
                yes_count, no_count, votes_cast, complete,
                majority_json, agents_json, updated_at
            ) VALUES (?, '', '', ?, ?, ?, ?, 1, ?, ?, datetime('now'))
            ON CONFLICT(video_key) DO UPDATE SET
                risk=excluded.risk,
                yes_count=excluded.yes_count,
                no_count=excluded.no_count,
                votes_cast=excluded.votes_cast,
                complete=1,
                majority_json=excluded.majority_json,
                agents_json=excluded.agents_json,
                updated_at=datetime('now')
            """,
            (
                video_key,
                risk,
                majority_vote["risk_votes"].get("risk_only", 0) + majority_vote["risk_votes"].get("abnormal", 0),
                majority_vote["risk_votes"].get("normal", 0),
                majority_vote["votes_cast"],
                json.dumps(majority_vote, ensure_ascii=False),
                json.dumps(agents_data, ensure_ascii=False),
            ),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated: {updated}, Skipped (not in DB): {skipped}")


if __name__ == "__main__":
    main()

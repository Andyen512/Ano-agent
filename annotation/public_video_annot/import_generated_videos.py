#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from app import APP_DIR, PROJECT_ROOT, init_db


GEN_DATA_ROOT = Path("/data_4/liuyuan/lifebench/data/generated_videos")
BATCH_OUTPUT_ROOT = PROJECT_ROOT / "public_data_filter" / "batch_outputs" / "gendata_20260529" / "reviews"
PROMPTS_FILE = PROJECT_ROOT / "data" / "prompts_0529.txt"


def load_prompts_file(prompts_path: Path) -> dict[str, str]:
    """Load prompts file and return a dict mapping video_id to prompt text."""
    prompts = {}
    if not prompts_path.exists():
        print(f"Warning: Prompts file not found: {prompts_path}")
        return prompts
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                video_id = parts[0]  # e.g., R000001_normal
                prompt_text = parts[3]  # The English description
                prompts[video_id] = prompt_text
    return prompts


def norm(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def video_key_from_gen_path(video_path: Path) -> str:
    """从生成视频路径生成 video_key"""
    try:
        rel = video_path.relative_to(GEN_DATA_ROOT)
        return f"generated_videos/{rel}"
    except ValueError:
        return f"generated_videos/{video_path.name}"


def load_gendata_results() -> list[dict[str, Any]]:
    """加载生成视频的评估结果"""
    results = []
    
    if not BATCH_OUTPUT_ROOT.exists():
        print(f"Batch output root not found: {BATCH_OUTPUT_ROOT}")
        return results
    
    # 加载 prompts 文件
    prompts_map = load_prompts_file(PROMPTS_FILE)
    print(f"Loaded {len(prompts_map)} prompts from {PROMPTS_FILE}")
    
    for review_summary_path in BATCH_OUTPUT_ROOT.rglob("review_summary.json"):
        try:
            data = json.loads(review_summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading {review_summary_path}: {e}")
            continue
        
        manifest = data.get("manifest", {})
        majority_vote = data.get("majority_vote", {})
        agent_results = data.get("agent_results", [])
        
        if not majority_vote.get("complete"):
            continue
        
        # 获取视频路径
        video_path_str = manifest.get("video_path", "")
        if not video_path_str:
            continue
        
        video_path = Path(video_path_str)
        video_key = video_key_from_gen_path(video_path)
        
        # 从路径提取元信息
        parts = video_path.relative_to(GEN_DATA_ROOT).parts
        # 结构: 场景/主体/风险类型/描述/模型/类型/视频文件
        scene = parts[0] if len(parts) > 0 else ""
        subject = parts[1] if len(parts) > 1 else ""
        risk_type = parts[2] if len(parts) > 2 else ""
        description = parts[3] if len(parts) > 3 else ""
        model_name = parts[4] if len(parts) > 4 else ""
        video_type = parts[5] if len(parts) > 5 else ""  # risk_only 或 abnormal
        
        # 获取原始 prompt
        video_stem = video_path.stem  # e.g., R000001_normal
        original_prompt = prompts_map.get(video_stem, "")
        
        # 获取平均分数
        avg_scores = majority_vote.get("avg_scores", {})
        avg_total_score = avg_scores.get("avg_total_score", 0)
        
        # 获取各模型的决策
        model_decisions = majority_vote.get("model_decisions", [])
        
        # 获取第一个成功 agent 的响应作为描述
        description_text = ""
        for agent in agent_results:
            if agent.get("returncode") == 0:
                response = agent.get("response", "")
                # 从响应中提取 explanation
                for line in response.split("\n"):
                    if line.lower().startswith("explanation:"):
                        description_text = line.split(":", 1)[1].strip()
                        break
                if description_text:
                    break
        
        results.append({
            "video_key": video_key,
            "video_path": str(video_path),
            "scene": scene,
            "subject": subject,
            "risk_type": risk_type,
            "description": description,
            "model_name": model_name,
            "video_type": video_type,
            "avg_total_score": avg_total_score,
            "avg_scores": avg_scores,
            "model_decisions": model_decisions,
            "majority_vote": majority_vote,
            "description_text": description_text,
            "original_prompt": original_prompt,
        })
    
    return results


def upsert_generated_video(conn: sqlite3.Connection, result: dict[str, Any]) -> None:
    """插入或更新生成视频记录"""
    video_key = result["video_key"]
    video_path = result["video_path"]

    existing_agents_json = conn.execute(
        "SELECT agents_json FROM model_predictions WHERE video_key = ?",
        (video_key,),
    ).fetchone()
    original_prompt_zh = ""
    if existing_agents_json and existing_agents_json[0]:
        try:
            original_prompt_zh = json.loads(existing_agents_json[0]).get("original_prompt_zh", "")
        except json.JSONDecodeError:
            original_prompt_zh = ""
    
    # 插入 videos 表
    conn.execute(
        """
        INSERT INTO videos (video_key, site, video_id, dataset, title, file_path)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_key) DO UPDATE SET
            title=excluded.title,
            file_path=excluded.file_path
        """,
        (video_key, "generated", video_key.split("/")[-1].replace(".mp4", ""), "generated_videos", 
         f"{result['scene']}/{result['risk_type']}/{result['description']}", video_path),
    )
    
    # 准备 model_predictions 数据
    majority_vote = result["majority_vote"]
    avg_scores = majority_vote.get("avg_scores", {})
    
    # 判定 risk: 根据 final_decision
    final_decision = majority_vote.get("final_decision", "unknown")
    risk = "Yes" if final_decision in ["accept", "weak_accept"] else "No"

    agents_data = {
        "source": "gendata_v2",
        "avg_scores": avg_scores,
        "model_decisions": result["model_decisions"],
        "video_type": result["video_type"],
        "model_name": result["model_name"],
        "original_prompt": result.get("original_prompt", ""),
    }
    if original_prompt_zh:
        agents_data["original_prompt_zh"] = original_prompt_zh
    
    conn.execute(
        """
        INSERT INTO model_predictions (
            video_key, source_output_root, summary_json, risk, level1_scene, level2_subject,
            level3_risk_type, description, risk_localization, solution_for_person,
            solution_for_hazard_source, solution_prevent_recurrence, yes_count, no_count,
            votes_cast, complete, majority_json, agents_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(video_key) DO UPDATE SET
            risk=excluded.risk,
            level1_scene=excluded.level1_scene,
            level2_subject=excluded.level2_subject,
            level3_risk_type=excluded.level3_risk_type,
            description=excluded.description,
            majority_json=excluded.majority_json,
            agents_json=excluded.agents_json,
            updated_at=datetime('now')
        """,
        (
            video_key,
            str(BATCH_OUTPUT_ROOT),
            str(BATCH_OUTPUT_ROOT),
            risk,
            result["scene"],
            result["subject"],
            result["risk_type"],
            result["description_text"],
            "None",
            "None",
            "None",
            "None",
            majority_vote.get("votes_cast", 0),
            0,
            majority_vote.get("votes_cast", 0),
            1 if majority_vote.get("complete") else 0,
            json.dumps(majority_vote, ensure_ascii=False),
            json.dumps(agents_data, ensure_ascii=False),
        ),
    )


def import_generated_videos(db_path: Path) -> dict[str, int]:
    """导入生成视频结果到数据库"""
    init_db()
    
    results = load_gendata_results()
    print(f"Found {len(results)} generated video results")
    
    conn = sqlite3.connect(db_path)
    imported = 0
    
    for result in results:
        try:
            upsert_generated_video(conn, result)
            imported += 1
        except Exception as e:
            print(f"Error importing {result['video_key']}: {e}")
    
    conn.commit()
    conn.close()
    
    return {
        "total_found": len(results),
        "imported": imported,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=APP_DIR / "data" / "app.db")
    args = parser.parse_args()
    
    stats = import_generated_videos(args.db.resolve())
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

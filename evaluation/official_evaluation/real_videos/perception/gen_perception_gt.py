#!/usr/bin/env python3
"""
从 GT 文件读取标注，调用 LLM 提取 risk_sources + abnormal_actions + affected_objects。
risk_type / scene / subject 从数据库 human_annotations 读取。
"""

import json
import re
import sqlite3
import threading
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

GT_TSV = Path("/home/caiqingyuan/code/lifebench/data/annotation/real_gt_expanded_release.txt")
OUTPUT_JSON = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/real_videos/perception/perception_gt.json")
DB_PATH = "/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db"

API_HOST = "www.lingganyaapi.com"
API_KEY = "sk-ke2RsoBH9l60EuXpKFLmCWTCuhT08B8jII9UkkHnaxWWWnRf"
MODEL = "gpt-5.6-luna"


def call_llm(prompt: str) -> dict:
    for attempt in range(3):
        try:
            resp = requests.post(f"https://{API_HOST}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
                timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        time.sleep(5)
    return {"choices": [{"message": {"content": "{}"}}]}


def build_prompt(timeline: str, solution: str) -> str:
    return (
        "你是视频安全标注助手。根据下面的描述，提取风险源、异常行为和受影响对象。\n\n"
        f"描述文本:\n{timeline}\n\n"
        f"解决方案:\n{solution}\n\n"
        "输出严格 JSON，不要任何额外文本:\n"
        "{\n"
        '  "risk_sources": ["风险源1", "风险源2"],\n'
        '  "abnormal_actions": ["异常动作1", "异常动作2"],\n'
        '  "affected_objects": ["受影响对象1", "受影响对象2"]\n'
        "}\n\n"
        "规则:\n"
        "1. risk_sources: 视频中的潜在风险源。注意：突然摔倒/晕倒/失去平衡等自身原因导致的异常，"
        '风险源应包括"自身身体健康状况（如头晕、突发疾病、失去平衡）"等自身因素\n'
        "2. abnormal_actions: 已发生或明显的异常行为\n"
        "3. affected_objects: 视频中受风险影响的对象或主体\n"
        "4. 只列出视频中明确存在的项，不要臆造\n"
        "5. 若某类为空，返回空数组 []\n"
        "6. 所有词汇用中文"
    )


def normalize_risk_status(s: str) -> str:
    s = s.strip().lower().replace("-", "").replace("_", "")
    if s in ("normal", "noanomaly", "safe"):
        return "normal"
    if s in ("riskonly", "risk_only", "potential"):
        return "risk_only"
    if s in ("abnormal", "occurred", "anomalyoccurred"):
        return "abnormal"
    return ""


def parse_gt_line(line: str):
    parts_raw = line.split("\t")
    parts = [p.strip() for p in parts_raw]
    if len(parts) < 3:
        return None
    video_id = parts[0]
    path_text = parts[1]
    is_status = normalize_risk_status(parts[2])
    if is_status:
        return {"video_id": video_id, "path_text": path_text, "risk_status": parts[2],
                "timeline": parts[3] if len(parts) > 3 else "",
                "solution": parts[4] if len(parts) > 4 else ""}
    return {"video_id": video_id, "path_text": path_text, "risk_status": "unknown",
            "timeline": parts[2], "solution": parts[3] if len(parts) > 3 else ""}


def load_db_info():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT video_key, level1_scene, level2_subject, level3_risk_type FROM human_annotations")
    info = {}
    for row in cursor.fetchall():
        key = row[0]
        # strip .mp4 suffix for matching
        plain = key.replace(".mp4", "")
        info[plain] = {"scene": row[1] or "", "subject": row[2] or "", "risk_type": row[3] or ""}
        info[key] = info[plain]
    conn.close()
    return info


def process_one(line: str, db_info: dict) -> dict:
    parsed = parse_gt_line(line)
    if not parsed:
        return None
    vid = parsed["video_id"]
    dinfo = db_info.get(vid, db_info.get(vid + ".mp4", db_info.get(vid + ".avi", {})))
    risk_type = dinfo.get("risk_type", "") if parsed["risk_status"] != "normal" else ""

    risk_sources = []
    abnormal_actions = []
    affected_objects = []

    if parsed["risk_status"] in ("risk_only", "abnormal", "unknown"):
        prompt = build_prompt(parsed["timeline"], parsed["solution"])
        try:
            resp = call_llm(prompt)
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            llm_out = json.loads(content) if isinstance(content, str) else content
            risk_sources = llm_out.get("risk_sources", [])
            abnormal_actions = llm_out.get("abnormal_actions", [])
            affected_objects = llm_out.get("affected_objects", [])
        except Exception as e:
            print(f"[{vid}] LLM 失败: {e}")

    return {"video_id": vid, "risk_status": parsed["risk_status"],
            "risk_type": risk_type, "scene": dinfo.get("scene", ""),
            "risk_sources": risk_sources, "abnormal_actions": abnormal_actions,
            "affected_objects": affected_objects}


def main():
    lines = GT_TSV.read_text(encoding="utf-8").splitlines()
    print(f"共 {len(lines)} 行")

    db_info = load_db_info()
    print(f"已加载 {len(db_info)} 条数据库信息")

    lock = threading.Lock()
    total = len(lines)
    done = [0]

    def progress_cb():
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  完成 {done[0]} / {total}")

    results = [None] * total
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, line in enumerate(lines):
            if not line.strip():
                with lock:
                    done[0] += 1
                continue
            futures[executor.submit(process_one, line, db_info)] = idx

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            progress_cb()

    results = [r for r in results if r is not None]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，写入 {OUTPUT_JSON} ({len(results)} 条)")


if __name__ == "__main__":
    main()

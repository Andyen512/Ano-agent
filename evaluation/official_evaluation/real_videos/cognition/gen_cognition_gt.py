#!/usr/bin/env python3
"""
从 GT 标注提取 Cognition GT：
- risk_description: 原始标注中的视频描述（timeline）
- consequence_understanding: LLM 从描述+方案中提取后果
- causal_chain: LLM 结合感知信息推理因果链
- causal_chain_text: 因果描述 + event_summary（事件总结追加在后面）
"""

import json
import re
import time
import requests
import sqlite3
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

GT_TSV = Path("/home/caiqingyuan/code/lifebench/data/annotation/real_gt_expanded_release.txt")
PERCEPTION_JSON = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/perception/perception_gt.json")
OUTPUT_JSON = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/real_videos/cognition/cognition_gt.json")
DB_PATH = "/home/caiqingyuan/code/lifebench/annotation/public_video_annot/data/app.db"

API_HOST = "www.lingganyaapi.com"
API_KEY = "sk-ke2RsoBH9l60EuXpKFLmCWTCuhT08B8jII9UkkHnaxWWWnRf"
MODEL = "gpt-5.6-luna"

scene_trans = {
    "living room": "客厅", "kitchen": "厨房", "bedroom": "卧室",
    "bathroom": "浴室", "dining room": "餐厅", "yard": "院子",
    "balcony": "阳台", "hallway": "走廊", "staircase": "楼梯",
    "entrance": "入口", "outdoor": "室外", "street": "街道",
    "study": "书房", "hospital corridor": "医院走廊",
}
subject_trans = {
    "child": "孩子", "young adult": "年轻人", "older adult": "老人",
    "middle-aged adult": "中年人", "adult": "成年人", "all": "多人",
    "tortoise": "乌龟", "infant": "婴儿",
}


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


def translate(text: str) -> str:
    for eng, chn in scene_trans.items():
        text = text.replace(eng, chn)
    for eng, chn in subject_trans.items():
        text = text.replace(eng, chn)
    return text


def parse_llm_response(resp: str) -> dict:
    try:
        outer = json.loads(resp)
        content = outer.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    except (json.JSONDecodeError, KeyError, IndexError):
        content = resp
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    return {}


def clean_description(text: str) -> str:
    return re.sub(r"^\[(?:Abnormal Result|Safe Action|Abnormal Sign)\](?:\[[^\]]*\])?\s*", "", text).strip()


def parse_gt_line(line: str):
    parts = [p.strip() for p in line.split("\t")]
    if len(parts) < 3:
        return None
    s = parts[2].strip().lower().replace("-", "").replace("_", "")
    if s in ("normal", "noanomaly", "safe", "riskonly", "potential", "abnormal", "occurred", "anomalyoccurred"):
        return {"video_id": parts[0], "risk_status": parts[2],
                "timeline": parts[3] if len(parts) > 3 else "",
                "solution": parts[4] if len(parts) > 4 else ""}
    return {"video_id": parts[0], "risk_status": "unknown",
            "timeline": parts[2], "solution": parts[3] if len(parts) > 3 else ""}


def load_json_by_id(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {item["video_id"]: item for item in data}
    return data


def build_prompt(timeline: str, solution: str, scene: str, subject: str,
                 risk_sources: list, abnormal_actions: list, affected_objects: list,
                 consequence: list, causal_text: str, is_abnormal: bool) -> str:
    tone = "已发生的异常事件" if is_abnormal else "潜在风险"
    return (
        "你是视频安全分析专家。根据下面的标注信息，提取后果、构建因果链，并生成完整的事件总结。\n\n"
        f"场景: {scene}\n"
        f"主体: {subject}\n"
        f"视频描述:\n{timeline}\n\n"
        f"解决方案:\n{solution}\n\n"
        f"已知风险源: {json.dumps(risk_sources, ensure_ascii=False)}\n"
        f"已知异常行为: {json.dumps(abnormal_actions, ensure_ascii=False)}\n"
        f"受影响对象: {json.dumps(affected_objects, ensure_ascii=False)}\n\n"
        f"标注类型: {tone}\n\n"
        "输出严格 JSON:\n"
        '{\n'
        '  "consequence_understanding": ["后果1", "后果2"],\n'
        '  "causal_chain": [{"source": "风险源", "action": "异常行为", "consequence": "后果"}],\n'
        '  "causal_chain_text": "一段自然的因果描述",\n'
        '  "event_summary": "完整的事件总结"\n'
        "}\n\n"
        "规则:\n"
        "1. consequence_understanding: 从描述和方案中提取该异常事件可能造成的具体后果\n"
        "2. causal_chain: 原子化的因果链数组，每条包含 source（风险源）、action（异常行为）、consequence（后果）\n"
        "3. causal_chain_text: 用一段自然的话描述完整的因果链，不要用固定模板\n"
        "4. event_summary: 用自然的中文叙述，覆盖场景、主体、风险源、异常行为、后果\n"
        f"5. 语气：{tone}，如果是潜在风险则用推测语气，已发生的异常则用陈述语气\n"
        "6. 所有内容必须来自标注信息，不要臆造\n"
        "7. 若 normal 视频无风险则所有字段为空\n"
        "8. 所有词汇用中文"
    )


def main():
    if not GT_TSV.exists():
        print(f"GT file not found: {GT_TSV}")
        return

    perception_data = load_json_by_id(PERCEPTION_JSON)
    print(f"已加载 {len(perception_data)} 条 perception GT")

    # 从数据库加载 scene/subject
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    db_info = {}
    lines = GT_TSV.read_text(encoding="utf-8").splitlines()
    for l in lines:
        if not l.strip():
            continue
        vid = l.split("\t")[0].strip()
        cursor.execute("SELECT level1_scene, level2_subject FROM human_annotations WHERE video_key=? OR video_key=?", (vid, vid + ".mp4"))
        row = cursor.fetchone()
        if row:
            db_info[vid] = {"scene": row[0] or "", "subject": row[1] or ""}
        else:
            parts = vid.split("/")
            db_info[vid] = {"scene": parts[0] if len(parts) >= 1 else "", "subject": parts[1] if len(parts) >= 2 else ""}
    conn.close()
    print(f"已加载 {len(db_info)} 条场景/主体信息")

    lock = threading.Lock()
    done = [0]

    def process_one(idx, line):
        if not line.strip():
            return None
        parsed = parse_gt_line(line)
        if not parsed:
            return None

        vid = parsed["video_id"]
        pdata = perception_data.get(vid, {})
        dinfo = db_info.get(vid, {"scene": "", "subject": ""})
        risk_desc = clean_description(parsed["timeline"])

        cons, chain, chain_text, event_summary = [], [], "", ""
        if parsed["risk_status"] not in ("normal",):
            prompt = build_prompt(parsed["timeline"], parsed["solution"],
                                  dinfo["scene"], dinfo["subject"],
                                  pdata.get("risk_sources", []),
                                  pdata.get("abnormal_actions", []),
                                  pdata.get("affected_objects", []),
                                  cons, chain_text, parsed["risk_status"] == "abnormal")
            try:
                resp = call_llm(prompt)
                content = resp["choices"][0]["message"]["content"]
                llm_out = json.loads(content)
                cons = llm_out.get("consequence_understanding", [])
                chain = llm_out.get("causal_chain", [])
                chain_text = llm_out.get("causal_chain_text", "")
                event_summary = translate(llm_out.get("event_summary", ""))
            except Exception as e:
                print(f"[{vid}] LLM 失败: {e}")

        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  完成 {done[0]} / {len(lines)}")

        return {"video_id": vid, "risk_description": risk_desc,
                "consequence_understanding": cons, "causal_chain": chain,
                "causal_chain_text": chain_text, "event_summary": event_summary}

    results = [None] * len(lines)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, line in enumerate(lines):
            futures[executor.submit(process_one, idx, line)] = idx
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    results = [r for r in results if r is not None]

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，写入 {OUTPUT_JSON} ({len(results)} 条)")


if __name__ == "__main__":
    main()

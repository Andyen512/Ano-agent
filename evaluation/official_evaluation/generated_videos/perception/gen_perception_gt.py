#!/usr/bin/env python3
"""Generate perception GT for generated_videos (LLM extracts risk_sources, abnormal_actions, affected_objects)."""
import json, re, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

INPUT_FILE = Path("/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_expanded_solutions.txt")
OUTPUT_DIR = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/generated_videos/perception")
OUTPUT_JSON = OUTPUT_DIR / "perception_gt.json"

API_HOST = "www.lingganyaapi.com"
API_KEY = "sk-ke2RsoBH9l60EuXpKFLmCWTCuhT08B8jII9UkkHnaxWWWnRf"
MODEL = "gpt-5.6-luna"

RISK_TYPE_MAP = {
    "高温火源": "heat/fire source",
    "跌倒失稳": "fall/instability",
    "碰撞砸伤": "collision/crush injury",
    "锐器危险": "sharp-object danger",
    "用电安全": "electrical safety",
    "中毒误食": "poisoning/accidental ingestion",
    "外部威胁": "interpersonal conflict",
    "碰撞伤害": "collision/crush injury",
}

SCENE_MAP = {
    "家中卫生间": "bathroom",
    "家中客厅": "living room",
    "家中庭院": "yard",
    "家庭卧室": "bedroom",
    "家庭厨房": "kitchen",
    "家庭餐厅": "dining room",
    "家用书房": "study",
    "家用阳台": "balcony",
}

def call_llm(prompt: str) -> dict:
    for attempt in range(3):
        try:
            resp = requests.post(
                f"https://{API_HOST}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()
            print(f"  [API] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"  [API] Exception: {e}")
        time.sleep(5)
    print(f"  [API] All retries exhausted")
    return {"choices": [{"message": {"content": "{}"}}]}


def build_prompt(desc: str, sol: str) -> str:
    return (
        "你是视频安全分析专家。根据视频描述和解决方案，提取风险源、异常行为和受影响对象。\n\n"
        f"视频描述:\n{desc}\n\n"
        f"解决方案:\n{sol}\n\n"
        "输出严格 JSON，不要任何额外文本:\n"
        "{\n"
        '  "risk_sources": ["风险源1", "风险源2"],\n'
        '  "abnormal_actions": ["异常动作1", "异常动作2"],\n'
        '  "affected_objects": ["受影响对象1", "受影响对象2"]\n'
        "}\n\n"
        "规则:\n"
        "1. risk_sources: 视频中的潜在风险源。注意：突然摔倒/晕倒/失去平衡等自身原因导致"
        "的异常，风险源应包括\"自身身体健康状况（如头晕、突发疾病、失去平衡）\"等自身因素\n"
        "2. abnormal_actions: 已发生或明显的异常行为\n"
        "3. affected_objects: 视频中受风险影响的对象或主体\n"
        "4. 只列出视频中明确存在的项，不要臆造\n"
        "5. 若某类为空，返回空数组 []\n"
        "6. 所有词汇用中文"
    )


def extract_video_id(path: str) -> str:
    idx = path.find("/generated_videos/")
    if idx != -1:
        vid = path[idx + 1:]
    else:
        vid = path
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    return vid


def process_line(line: str, header_parts: list[str]) -> dict | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 10:
        return None

    hmap = {h: i for i, h in enumerate(header_parts)}
    vpath = parts[hmap["video_path"]]
    vtype = parts[hmap["video_type"]]
    desc = parts[hmap["description"]] if hmap["description"] < len(parts) else ""
    sol = parts[hmap["solutions"]] if hmap["solutions"] < len(parts) else ""
    scene_cn = parts[hmap["scene"]] if hmap["scene"] < len(parts) else ""
    subject_cn = parts[hmap["subject"]] if hmap["subject"] < len(parts) else ""
    risk_type_cn = parts[hmap["risk_type"]] if hmap["risk_type"] < len(parts) else ""

    risk_status = vtype.lower()
    scene = SCENE_MAP.get(scene_cn, scene_cn)
    risk_type = RISK_TYPE_MAP.get(risk_type_cn, risk_type_cn)
    risk_sources = []
    abnormal_actions = []
    affected_objects = []

    if risk_status in ("abnormal", "risk_only"):
        prompt = build_prompt(desc, sol)
        try:
            resp = call_llm(prompt)
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            # Extract JSON from content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                llm_out = json.loads(json_match.group(0))
            else:
                llm_out = json.loads(content)
            risk_sources = llm_out.get("risk_sources", [])
            abnormal_actions = llm_out.get("abnormal_actions", [])
            affected_objects = llm_out.get("affected_objects", [])
        except Exception as e:
            print(f"[{extract_video_id(vpath)}] Perception LLM error: {e}")

    return {
        "video_id": extract_video_id(vpath),
        "risk_status": risk_status,
        "risk_type": risk_type,
        "scene": scene,
        "risk_sources": risk_sources,
        "abnormal_actions": abnormal_actions,
        "affected_objects": affected_objects,
    }


def quick_video_id(line: str, path_idx: int) -> str:
    parts = line.split("\t")
    return extract_video_id(parts[path_idx]) if path_idx < len(parts) else ""


def save_checkpoint(results: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    data = [results[k] for k in sorted(results.keys())]
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        lines = f.readlines()

    header_parts = header.split("\t")
    path_idx = header_parts.index("video_path")
    vtype_idx = header_parts.index("video_type")
    print(f"Total lines: {len(lines)}")

    # Resume from existing output
    existing_ids = set()
    if OUTPUT_JSON.exists():
        try:
            old = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
            existing_ids = {r["video_id"] for r in old}
            print(f"Resume: loaded {len(existing_ids)} existing entries")
        except:
            print(f"Failed to load existing output, starting fresh")

    risk_lines = []
    normal_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) <= vtype_idx:
            continue
        vtype = parts[vtype_idx].strip().lower()
        if vtype in ("abnormal", "risk_only"):
            risk_lines.append(line)
        else:
            normal_lines.append(line)

    pending_risk = [l for l in risk_lines if quick_video_id(l, path_idx) not in existing_ids]
    print(f"Need LLM (perception): {len(pending_risk)}/{len(risk_lines)}, Normal: {len(normal_lines)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if existing_ids:
        old = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        results = {r["video_id"]: r for r in old}
    else:
        results = {}

    # Process normal lines (no LLM) — only new ones
    new_normal = 0
    for line in normal_lines:
        vid = quick_video_id(line, path_idx)
        if vid and vid not in existing_ids:
            r = process_line(line, header_parts)
            if r:
                results[r["video_id"]] = r
                new_normal += 1
    if new_normal:
        print(f"  Added {new_normal} new normal entries")

    if not pending_risk:
        print("All risk lines already processed, skipping LLM.")
    else:
        lock = threading.Lock()
        done = [len(results)]
        total = len(lines)

        def process_and_report(line):
            r = process_line(line, header_parts)
            if r:
                vid = r["video_id"]
                with lock:
                    results[vid] = r
                    done[0] += 1
                    if done[0] % 100 == 0:
                        print(f"  {done[0]}/{total}")
                        save_checkpoint(results, OUTPUT_JSON)
            return r

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_and_report, l): l for l in pending_risk}
            for future in as_completed(futures):
                future.result()

    save_checkpoint(results, OUTPUT_JSON)
    print(f"Perception GT: {len(results)} entries -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

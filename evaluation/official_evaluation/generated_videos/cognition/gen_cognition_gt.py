#!/usr/bin/env python3
"""
生成 generated_videos 的 Cognition GT：
- risk_description: 直接提取 GT 中的 description 字段（已翻译的中文描述）
- consequence_understanding / causal_chain / causal_chain_text / event_summary: LLM 生成
参照 real_videos/cognition/gen_cognition_gt.py
"""
import json, re, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

INPUT_FILE = Path("/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_expanded_solutions.txt")
OUTPUT_DIR = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/generated_videos/cognition")
OUTPUT_JSON = OUTPUT_DIR / "cognition_gt.json"

API_HOST = "www.lingganyaapi.com"
API_KEY = "sk-ke2RsoBH9l60EuXpKFLmCWTCuhT08B8jII9UkkHnaxWWWnRf"
MODEL = "gpt-5.6-luna"

SCENE_CN_MAP = {
    "家中卫生间": "卫生间", "家中客厅": "客厅", "家中庭院": "庭院",
    "家庭卧室": "卧室", "家庭厨房": "厨房", "家庭餐厅": "餐厅",
    "家用书房": "书房", "家用阳台": "阳台",
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


def build_prompt(description: str, solution: str, scene: str, subject: str, is_abnormal: bool) -> str:
    tone = "已发生的异常事件" if is_abnormal else "潜在风险"
    return (
        "你是视频安全分析专家。根据下面的标注信息，提取后果、构建因果链，并生成完整的事件总结。\n\n"
        f"场景: {scene}\n"
        f"主体: {subject}\n"
        f"视频描述:\n{description}\n\n"
        f"解决方案:\n{solution}\n\n"
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
        "3. causal_chain_text: 用一段自然的话描述完整的因果链\n"
        "4. event_summary: 用自然的中文叙述，覆盖场景、主体、风险源、异常行为、后果\n"
        f"5. 语气：{tone}，如果是潜在风险则用推测语气，已发生的异常则用陈述语气\n"
        "6. 所有内容必须来自标注信息，不要臆造\n"
        "7. 所有词汇用中文"
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
    is_abnormal = (vtype == "abnormal")
    vid = extract_video_id(vpath)

    if risk_status in ("abnormal", "risk_only"):
        scene = SCENE_CN_MAP.get(scene_cn, scene_cn)
        prompt = build_prompt(desc, sol, scene, subject_cn, is_abnormal)
        consequence_understanding = []
        causal_chain = []
        causal_chain_text = ""
        event_summary = ""
        try:
            llm_out = parse_llm_response(json.dumps(call_llm(prompt)))
            consequence_understanding = llm_out.get("consequence_understanding", [])
            causal_chain = llm_out.get("causal_chain", [])
            causal_chain_text = llm_out.get("causal_chain_text", "")
            event_summary = llm_out.get("event_summary", "")
        except Exception as e:
            print(f"[{vid}] Cognition LLM error: {e}")
        return {
            "video_id": vid,
            "risk_description": desc,
            "consequence_understanding": consequence_understanding,
            "causal_chain": causal_chain,
            "causal_chain_text": causal_chain_text,
            "event_summary": event_summary,
        }
    else:
        # normal: video_description + risk_type
        return {
            "video_id": vid,
            "video_description": desc,
            "risk_type": risk_type_cn,
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
            for r in old:
                # Only treat as done if cognition fields are non-empty (for risk entries)
                if "risk_description" in r:
                    if r.get("consequence_understanding") or r.get("causal_chain") or r.get("event_summary"):
                        existing_ids.add(r["video_id"])
                else:
                    existing_ids.add(r["video_id"])
            print(f"Resume: loaded {len(existing_ids)} completed entries (skipping {len(old) - len(existing_ids)} empty-risk)")
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

    # Filter already-processed risk lines
    pending_risk = [l for l in risk_lines if quick_video_id(l, path_idx) not in existing_ids]
    print(f"Need LLM (cognition): {len(pending_risk)}/{len(risk_lines)}, Normal: {len(normal_lines)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for vid in existing_ids:
        pass  # will be loaded from file on first save, or we preload

    # Preload existing results
    # Load existing valid results
    results = {}
    if existing_ids:
        old = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        for r in old:
            vid = r["video_id"]
            if vid in existing_ids:
                results[vid] = r
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
    print(f"Cognition GT: {len(results)} entries -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

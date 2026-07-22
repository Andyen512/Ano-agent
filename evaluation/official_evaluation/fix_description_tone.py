#!/usr/bin/env python3
"""
修复描述语气：
- abnormal 描述 → 去掉推测性语言（用事实陈述已发生事件）
- risk_only 描述 → 加上推测性语言（用推测语气描述潜在风险）
"""

import json
import re
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

GT_FILE = "/home/caiqingyuan/code/lifebench/data/annotation/real_gt_expanded_release.txt"
PERCEPTION_JSON = Path("/home/caiqingyuan/code/lifebench/evaluation/official_evaluation/perception/perception_gt.json")
API_HOST = "www.lingganyaapi.com"
API_KEY = "sk-ke2RsoBH9l60EuXpKFLmCWTCuhT08B8jII9UkkHnaxWWWnRf"
MODEL = "gpt-5.6-luna"
WORKERS = 10

# ============================================================
# PROMPT 模板 - 请审核修改
# ============================================================

ABNORMAL_PROMPT = """你是视频描述改写助手。下面是一条视频描述，本应使用事实陈述语气描述一个已经发生的异常事件，
但当前描述中包含了推测性语言。请将其改写为纯事实陈述，去掉所有推测性词汇。

规则：
1. 去掉可能、存在风险、如果、一旦、容易等推测性词汇
2. 用已经发生的语气描述事件（如摔倒、起火、进入）
3. 不能增加原文中没有出现的信息
4. 保持时间前缀不变
5. 只输出改写后的完整描述文本

原始描述：
{desc}

改写后："""

RISK_ONLY_PROMPT = """你是视频描述改写助手。下面是一条视频描述，本应使用推测语气描述一个潜在风险场景，
但当前描述使用了事实陈述语气。请将其改写为推测性描述，突出潜在风险。

规则：
1. 添加可能、存在风险、容易、如果等推测性词汇
2. 描述潜在后果（如可能导致受伤、存在摔倒风险）
3. 不能增加原文中没有出现的信息
4. 保持时间前缀不变
5. 只输出改写后的完整描述文本

原始描述：
{desc}

改写后："""

# ============================================================

speculative_words = ['可能', '容易', '存在风险', '危险', '如果', '一旦', '不慎', '存在', '隐患', '小心', '注意', '缺乏', '没有', '会导致', '可能造成', '往往']

lock = threading.Lock()
progress = {"done": 0, "total": 0}

def call_llm(prompt):
    for attempt in range(3):
        try:
            r = requests.post(f"https://{API_HOST}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
                timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except:
            pass
        time.sleep(5)
    return ""

def clean_prefix(text):
    m = re.match(r"(\[(?:Abnormal Result|Abnormal Sign|Safe Action)\](?:\[[^\]]*\])?\s*)", text)
    return (m.group(1) if m else "", re.sub(r"^\[(?:Abnormal Result|Abnormal Sign|Safe Action)\](?:\[[^\]]*\])?\s*", "", text).strip())

def process_entry(vid, desc_text):
    prefix, content = clean_prefix(desc_text)
    spec_count = sum(1 for w in speculative_words if w in content)

    p = json.loads(PERCEPTION_JSON.read_text(encoding="utf-8"))
    p_by_id = {item["video_id"]: item["risk_status"] for item in p}
    status = p_by_id.get(vid, "")

    need_fix = False
    if status == "abnormal" and spec_count >= 2:
        prompt = ABNORMAL_PROMPT.format(desc=content)
        need_fix = True
    elif status == "risk_only" and spec_count == 0:
        prompt = RISK_ONLY_PROMPT.format(desc=content)
        need_fix = True

    if not need_fix:
        return None

    new_content = call_llm(prompt)
    if not new_content:
        return None

    new_desc = prefix + new_content
    with lock:
        progress["done"] += 1
        if progress["done"] % 50 == 0:
            print(f"  [{progress['done']}/{progress['total']}]")
    return (vid, new_desc)


def main():
    lines = Path(GT_FILE).read_text(encoding="utf-8").splitlines()
    gt_by_id = {l.split("\t")[0]: l for l in lines if l.strip()}

    p = json.loads(PERCEPTION_JSON.read_text(encoding="utf-8"))
    p_by_id = {item["video_id"]: item["risk_status"] for item in p}

    entries_to_fix = []
    for vid, line in gt_by_id.items():
        status = p_by_id.get(vid)
        if not status or status == "normal":
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        desc = parts[3]
        _, content = clean_prefix(desc)
        spec_count = sum(1 for w in speculative_words if w in content)
        if status == "abnormal" and spec_count >= 2:
            entries_to_fix.append((vid, desc))
        elif status == "risk_only" and spec_count == 0:
            entries_to_fix.append((vid, desc))

    print(f"需修复: {len(entries_to_fix)} 条")
    print(f"  abnormal (去掉推测): ~{sum(1 for v,d in entries_to_fix if p_by_id.get(v)=='abnormal')}")
    print(f"  risk_only (加上推测): ~{sum(1 for v,d in entries_to_fix if p_by_id.get(v)=='risk_only')}")
    progress["total"] = len(entries_to_fix)

    if not entries_to_fix:
        print("无需修复")
        return

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_entry, vid, desc): vid for vid, desc in entries_to_fix}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results[r[0]] = r[1]

    print(f"\nLLM 改写完成: {len(results)}/{len(entries_to_fix)}")

    # 写回 GT
    for i, line in enumerate(lines):
        vid = line.split("\t")[0]
        if vid in results:
            parts = line.split("\t")
            parts[3] = results[vid]
            lines[i] = "\t".join(parts)

    Path(GT_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已更新 {GT_FILE}")
    print(f"总行数: {len(lines)}")


if __name__ == "__main__":
    main()

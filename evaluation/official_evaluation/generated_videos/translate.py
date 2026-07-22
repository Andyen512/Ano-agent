#!/usr/bin/env python3
"""使用 DeepSeek API 翻译 generated_gt_release.txt 中的 description 和 solutions 字段。"""

import os
import time
import requests

# ============ 配置区 ============
API_KEY = "sk-b99ec07526524e9ebdf2c126dd8d5699"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 输入文件路径
INPUT_FILE = "/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release.txt"
# 输出文件路径（翻译后的结果保存到此）
OUTPUT_FILE = "/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_translated.txt"

# ============ 调用 DeepSeek API 翻译 ============

SYSTEM_PROMPT = (
    "你是一个专业的翻译助手。请将文本中的英文部分翻译成中文，保持原意准确、语言自然流畅。"
    "只返回翻译后的完整结果，不要添加任何解释或额外内容。"
    "严格保留以下内容原样不动：1) 时间戳标注（如 0-8s：）；2) 分隔符 |；"
    "3) 中文标签（如 对人的解决方案：、对危险源的解决方案：、后续防止危险复发的解决方案：）。"
    "仅翻译这些标记之后的英文描述内容。"
)


def translate_text(text: str, max_retries: int = 3) -> str:
    """调用 DeepSeek API 翻译单段文本"""
    if not text.strip():
        return text

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                result = data["choices"][0]["message"]["content"].strip()
                return result.replace("\n", " ")  # 避免换行破坏 TSV 格式
            else:
                print(f"  API 返回错误 (HTTP {resp.status_code}): {resp.text[:300]}")
        except Exception as e:
            print(f"  请求异常: {e}")

        if attempt < max_retries - 1:
            wait = 2 * (attempt + 1)
            print(f"  等待 {wait}s 后重试...")
            time.sleep(wait)

    print(f"  翻译失败，保留原文。")
    return text


def translate_description(text: str) -> str:
    """翻译 description 字段（一次调用，保留时间戳和 | 分隔符）"""
    if not text.strip():
        return text
    return translate_text(text)


def translate_solutions(text: str) -> str:
    """翻译 solutions 字段（一次调用，保留中文标签）"""
    if not text.strip():
        return text
    return translate_text(text)


def parse_line(line: str) -> dict:
    """将 TSV 行解析为字典"""
    fields = line.rstrip("\n").split("\t")
    names = [
        "video_path", "R_number", "video_type", "scene", "subject",
        "risk_type", "model", "path_desc", "description", "solutions",
        "risk_decision", "annotators",
    ]
    row = {}
    for i, name in enumerate(names):
        row[name] = fields[i] if i < len(fields) else ""
    return row


def row_to_line(row: dict) -> str:
    """将字典还原为 TSV 行"""
    names = [
        "video_path", "R_number", "video_type", "scene", "subject",
        "risk_type", "model", "path_desc", "description", "solutions",
        "risk_decision", "annotators",
    ]
    return "\t".join(row.get(name, "") for name in names)


def translate_single_row(row: dict) -> dict:
    """翻译单行数据的 description 和 solutions
    normal: 只翻译 description（solutions 为空）
    abnormal / risk_only: 翻译 description 和 solutions
    """
    video_type = row.get("video_type", "")
    desc = row.get("description", "")
    sol = row.get("solutions", "")

    print(f"  类型: {video_type}")

    if desc.strip():
        print(f"  原始 description: {desc[:120]}...")
        new_desc = translate_description(desc)
        print(f"  翻译 description: {new_desc[:120]}...")
        row["description"] = new_desc
    else:
        print(f"  description 为空，跳过")

    if video_type == "normal":
        print(f"  normal 类型无 solutions，跳过")
    elif sol.strip():
        print(f"  原始 solutions: {sol[:120]}...")
        new_sol = translate_solutions(sol)
        print(f"  翻译 solutions: {new_sol[:120]}...")
        row["solutions"] = new_sol
    else:
        print(f"  solutions 为空，跳过")

    return row


# ============ 测试：从文件中取第一条有内容的非 header 行 ============

def test_one():
    """从输入文件中各取一条 normal 和 abnormal/risk_only 数据进行翻译测试"""
    print("=" * 60)
    print("【测试模式】翻译一条 normal + 一条 abnormal/risk_only")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    normal_row = None
    risk_row = None

    for i, line in enumerate(lines[1:], start=2):
        row = parse_line(line)
        vt = row.get("video_type", "")
        if normal_row is None and vt == "normal" and row.get("description", "").strip():
            normal_row = (i, row)
        if risk_row is None and vt in ("abnormal", "risk_only") and row.get("description", "").strip() and row.get("solutions", "").strip():
            risk_row = (i, row)
        if normal_row and risk_row:
            break

    if normal_row:
        i, row = normal_row
        print(f"\n--- 第 {i} 行（{row['R_number']} - {row['video_type']}）---\n")
        translated = translate_single_row(row)
        print(f"\n翻译结果:")
        print(f"  Description:\n    {translated['description']}")
        print(f"  Solutions:\n    {translated['solutions']}")
    else:
        print("未找到 normal 行！")

    if risk_row:
        i, row = risk_row
        print(f"\n--- 第 {i} 行（{row['R_number']} - {row['video_type']}）---\n")
        translated = translate_single_row(row)
        print(f"\n翻译结果:")
        print(f"  Description:\n    {translated['description']}")
        print(f"  Solutions:\n    {translated['solutions']}")
    else:
        print("未找到 abnormal/risk_only 行！")


def translate_all():
    """翻译整个文件（支持断点续传）
    同一 R_number + video_type 下 3 个模型共用同一份 description 翻译结果。
    solutions 每个模型单独翻译。
    """
    print("=" * 60)
    print("【全量模式】翻译所有数据")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].rstrip("\n")
    total = len(lines) - 1
    print(f"共 {total} 行数据待处理\n")

    start_idx = 0
    desc_cache = {}

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output_lines = f.readlines()
        if len(output_lines) > 1 and output_lines[0].rstrip("\n") == header:
            start_idx = len(output_lines) - 1
            if start_idx > 0:
                print(f"检测到已翻译 {start_idx}/{total} 行，从第 {start_idx + 1} 行继续\n")
                for j in range(start_idx):
                    out_row = parse_line(output_lines[j + 1])
                    od = out_row.get("description", "").strip()
                    ovt = out_row.get("video_type", "")
                    orn = out_row.get("R_number", "")
                    if od and (orn, ovt) not in desc_cache:
                        desc_cache[(orn, ovt)] = od
        else:
            start_idx = 0
    else:
        start_idx = 0

    if start_idx == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(header + "\n")

    for i, line in enumerate(lines[1:], start=1):
        if i <= start_idx:
            continue

        row = parse_line(line)
        desc = row.get("description", "").strip()
        sol = row.get("solutions", "").strip()
        vt = row.get("video_type", "")
        rn = row.get("R_number", "")
        cache_key = (rn, vt)

        if desc or sol:
            print(f"[{i}/{total}] {rn} - {vt} ({row['model']})")
            if desc:
                if cache_key in desc_cache:
                    row["description"] = desc_cache[cache_key]
                    print(f"  description 复用缓存")
                else:
                    print(f"  翻译 description...")
                    translated = translate_description(desc)
                    desc_cache[cache_key] = translated
                    row["description"] = translated
            if vt == "normal":
                print(f"  normal 类型，跳过 solutions")
            elif sol:
                print(f"  翻译 solutions...")
                row["solutions"] = translate_solutions(sol)
        else:
            print(f"[{i}/{total}] 跳过 {rn} - {vt} ({row['model']}) (无内容)")

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(row_to_line(row) + "\n")

        time.sleep(0.1)  # 控制请求频率

    print(f"\n完成！结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        translate_all()
    else:
        test_one()

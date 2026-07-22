#!/usr/bin/env python3
"""提取 solutions 字段中的三个子解决方案，调用 DeepSeek API 进行润色扩充到约50字。"""

import os
import re
import time
import requests

# ============ 配置区 ============
API_KEY = "sk-b99ec07526524e9ebdf2c126dd8d5699"
API_URL = "https://api.deepseek.com/v1/chat/completions"

INPUT_FILE = "/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_translated.txt"
OUTPUT_FILE = "/home/caiqingyuan/code/lifebench/data/annotation/generated_gt_release_expanded_solutions.txt"

# ============ 解析 solutions 字段 ============

SOLUTION_LABELS = [
    "对人的解决方案：",
    "对危险源的解决方案：",
    "后续防止危险复发的解决方案：",
]


def parse_solutions(solutions_text: str) -> dict:
    """将 solutions 字段解析为三个子解决方案的字典"""
    result = {"对人": "", "对危险源": "", "后续": ""}
    if not solutions_text.strip():
        return result

    # 按标签拆分
    parts = {}
    remaining = solutions_text
    for i, label in enumerate(SOLUTION_LABELS):
        idx = remaining.find(label)
        if idx == -1:
            continue
        start = idx + len(label)
        end = remaining.find(SOLUTION_LABELS[i + 1]) if i + 1 < len(SOLUTION_LABELS) else len(remaining)
        content = remaining[start:end].strip()
        if i == 0:
            parts["对人"] = content
        elif i == 1:
            parts["对危险源"] = content
        elif i == 2:
            parts["后续"] = content

    return {k: parts.get(k, "") for k in ["对人", "对危险源", "后续"]}


def assemble_solutions(parts: dict) -> str:
    """将三个子解决方案重新组装为 solutions 字段"""
    segments = []
    if parts.get("对人"):
        segments.append("对人的解决方案：" + parts["对人"])
    if parts.get("对危险源"):
        segments.append("对危险源的解决方案：" + parts["对危险源"])
    if parts.get("后续"):
        segments.append("后续防止危险复发的解决方案：" + parts["后续"])
    return " ".join(segments)


# ============ 调用 DeepSeek API 扩充 ============

SYSTEM_PROMPT = (
    "你是一个专业的安全解决方案撰写助手。"
    "你需要根据视频描述和已有的三个子解决方案，"
    "对每个子解决方案进行润色扩充。"
    "目标字数：扩充到50字以上、60字以下为佳，只增不减。"
    "如果原始方案已经超过50字，则保持原长度或略微润色扩充，不要缩短。"
    "扩充时保留原始解决方案的核心意图和安全要点，不要凭空编造不相关的内容。"
    "结合视频描述中的具体场景元素（人物、物体、环境等），使方案更具针对性和实用性。"
    "只返回扩充后的三个解决方案，严格按以下格式输出，不要添加任何解释或额外内容：\n"
    "对人的解决方案：<扩充后的内容>\n"
    "对危险源的解决方案：<扩充后的内容>\n"
    "后续防止危险复发的解决方案：<扩充后的内容>"
)


def expand_solutions(description: str, parts: dict, max_retries: int = 3) -> dict:
    """调用 DeepSeek API 扩充三个子解决方案"""
    if not any(parts.values()):
        return parts

    user_prompt = (
        f"视频描述：\n{description}\n\n"
        f"原始解决方案：\n"
        f"对人的解决方案：{parts['对人']}\n"
        f"对危险源的解决方案：{parts['对危险源']}\n"
        f"后续防止危险复发的解决方案：{parts['后续']}\n\n"
        f"请将以上三个解决方案各扩充到50字以上（只增不减），保持原意，输出格式与输入一致。"
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        "thinking": {"type": "disabled"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                result = data["choices"][0]["message"]["content"].strip()
                return parse_solutions(result)
            else:
                print(f"  API 返回错误 (HTTP {resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"  请求异常: {e}")

        if attempt < max_retries - 1:
            wait = 2 * (attempt + 1)
            print(f"  等待 {wait}s 后重试...")
            time.sleep(wait)

    print(f"  扩充失败，保留原始方案。")
    return parts


# ============ 文件解析 ============

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


# ============ 主流程 ============

def test_one():
    """测试：取一条 non-normal 数据进行扩充测试"""
    print("=" * 60)
    print("【测试模式】扩充一条 abnormal/risk_only 的 solutions")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines[1:], start=2):
        row = parse_line(line)
        vt = row.get("video_type", "")
        sol = row.get("solutions", "").strip()
        desc = row.get("description", "")
        if vt in ("abnormal", "risk_only") and sol:
            print(f"\n--- 第 {i} 行（{row['R_number']} - {row['video_type']} - {row['model']}）---\n")
            print(f"  description: {desc[:150]}...")
            parts = parse_solutions(sol)
            print(f"  原始 solutions:")
            for key in ["对人", "对危险源", "后续"]:
                print(f"    {key}: {parts[key]}")
            expanded = expand_solutions(desc, parts)
            print(f"\n  扩充后 solutions:")
            for key in ["对人", "对危险源", "后续"]:
                print(f"    {key}: {expanded.get(key, '')}")
            print(f"\n  组装后: {assemble_solutions(expanded)}")
            break
    else:
        print("未找到 non-normal 行!")


def expand_all():
    """扩充整个文件的 solutions（支持断点续传）"""
    print("=" * 60)
    print("【全量模式】扩充所有 solutions")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header = lines[0].rstrip("\n")
    total = len(lines) - 1
    skip_count = 0
    expand_count = 0
    print(f"共 {total} 行数据待处理\n")

    # 断点续传
    start_idx = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            output_lines = f.readlines()
        if len(output_lines) > 1 and output_lines[0].rstrip("\n") == header:
            start_idx = len(output_lines) - 1
            # 统计已完成数
            for j in range(start_idx):
                out_row = parse_line(output_lines[j + 1])
                if out_row.get("video_type", "") != "normal" and out_row.get("solutions", "").strip():
                    expand_count += 1
                else:
                    skip_count += 1
            print(f"检测到已处理 {start_idx}/{total} 行（扩充 {expand_count}，跳过 {skip_count}），从第 {start_idx + 1} 行继续\n")
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
        vt = row.get("video_type", "")
        sol = row.get("solutions", "").strip()
        desc = row.get("description", "")

        if vt != "normal" and sol:
            expand_count += 1
            print(f"[{i}/{total}] {row['R_number']} - {vt} ({row['model']}) 扩充 solutions...")
            parts = parse_solutions(sol)
            original_len = {k: len(v) for k, v in parts.items()}
            expanded = expand_solutions(desc, parts)
            new_len = {k: len(v) for k, v in expanded.items()}
            print(f"  字数变化: "
                  f"对人 {original_len['对人']}→{new_len['对人']}, "
                  f"对危险源 {original_len['对危险源']}→{new_len['对危险源']}, "
                  f"后续 {original_len['后续']}→{new_len['后续']}")
            row["solutions"] = assemble_solutions(expanded)
        else:
            skip_count += 1
            if skip_count % 500 == 0:
                print(f"[{i}/{total}] 跳过 {row['R_number']} - {vt} ({row['model']}) (无需扩充)")

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(row_to_line(row) + "\n")

        if vt != "normal" and sol:
            time.sleep(0.1)

    print(f"\n完成！共扩充 {expand_count} 条，跳过 {skip_count} 条。结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        expand_all()
    else:
        test_one()

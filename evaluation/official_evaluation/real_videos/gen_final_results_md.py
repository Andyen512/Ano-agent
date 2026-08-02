#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path("/home/caiqingyuan/code/lifebench")
parser = argparse.ArgumentParser(description="Render detailed LifeBench real-video results.")
parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation" / "real_videos")
parser.add_argument("--output-file", type=Path)
parser.add_argument("--title", default="LifeBench Real-Videos 评测结果")
parser.add_argument("--evaluation-label", default="real_videos")
parser.add_argument("--gt-label", default="/data/.../annotations/real_videos/ (3000 条)")
parser.add_argument(
    "--excluded-label",
    default="video_llava_7b (输出空响应), mplug_owl3_7b (输出截断循环)",
)
args = parser.parse_args()
OUTPUT_DIR = args.output_dir.resolve()
OUTPUT_FILE = (args.output_file or (OUTPUT_DIR / "FINAL_RESULTS.md")).resolve()

pv = []
legacy_path = OUTPUT_DIR / "per_video_scores.json"
if legacy_path.exists():
    pv.extend(json.loads(legacy_path.read_text(encoding="utf-8")))

# Prefer per-model evaluations when they exist, replacing stale legacy rows.
model_score_files = sorted(OUTPUT_DIR.glob("*/per_video_scores.json"))
override_model_ids = set()
model_results_by_id = {}
for score_path in model_score_files:
    rows = json.loads(score_path.read_text(encoding="utf-8"))
    ids = {row.get("model_id") for row in rows if row.get("model_id")}
    override_model_ids.update(ids)
    for model_id in ids:
        model_results_by_id[model_id] = [row for row in rows if row.get("model_id") == model_id]
pv = [row for row in pv if row.get("model_id") not in override_model_ids]
for rows in model_results_by_id.values():
    pv.extend(rows)

groups = defaultdict(list)
for item in pv:
    groups[(item["model_id"], item["risk_status"])].append(item)

model_order = [
    "Qwen/Qwen3.5-9B",
    "gemini-3.1-pro",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "DAMO-NLP-SG/VideoLLaMA3-7B",
    "OpenGVLab/InternVL3_5-8B",
    "OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
    "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F",
    "omni-research/Tarsier2-7b-0115",
    "Vision-CAIR/MiniGPT4-Video",
    "MBZUAI/Video-ChatGPT-7B",
]

status_order = ["abnormal", "risk_only", "normal"]

def fmt(v, decimals=1):
    return f"{v:.{decimals}f}"

def short_name(mid):
    if mid == "omni-research/Tarsier2-7b-0115":
        return "Tarsier2-7B"
    return mid.split("/")[-1]

header = """# {title}

生成时间: `{time}`
评测模式: `mixed_openai` (LLM Judge: local Qwen3-8B)
数据集: `{evaluation_label}`
GT: `{gt_label}`
排除模型: `{excluded_label}`

## 权重说明
- **四维度平权** (各 0.25): Perception / Cognition / Temporal / Planning
- **Normal**: Perception = status_accuracy, Cognition = risk_desc, 总分 = 0.5xP + 0.5xC
- **非 Normal**: Perception 含 5 子指标 (各 0.20), Cognition 含 4 子指标 (各 0.25)
"""

label_map = {
    "abnormal": "异常事件 (abnormal)",
    "risk_only": "仅风险标签 (risk_only)",
    "normal": "正常视频 (normal)",
}

sections = []
for status in status_order:
    rows = []
    for mid in model_order:
        items = groups.get((mid, status), [])
        if not items:
            continue
        n = len(items)
        cat = {}
        m = {}
        metric_names = list(items[0]["metrics"].keys())
        for name in metric_names:
            m[name] = sum(r["metrics"][name] for r in items) / n
        for ck in ["perception", "cognition", "temporal_grounding", "intervention_planning"]:
            cat[ck] = sum(r["category_scores"].get(ck, 0) for r in items) / n
        overall = m.get("overall_score", 0)
        rows.append((n, overall, short_name(mid), m, cat))

    if not rows:
        continue
    rows.sort(key=lambda r: -r[1])

    lines = [f"## {label_map[status]}", ""]
    lines.append('<table border="1" cellspacing="0" cellpadding="4">')
    if status == "normal":
        hdrs = ["模型", "视频数", "总分", "status", "desc"]
    else:
        hdrs = ["模型","视频数","总分","Perception","status","type","source","action","object",
                "Cognition","desc","conseq","factual","causal","Temporal",
                "IoU","start","end","duration","Planning","person","hazard","overall"]
    lines.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in hdrs) + "</tr></thead>")
    lines.append("<tbody>")
    for n, overall, sname, m, cat in rows:
        if status == "normal":
            vals = [sname, str(n), fmt(overall), fmt(m["risk_status_accuracy"]),
                    fmt(m["risk_description_score"])]
        else:
            vals = [sname, str(n), fmt(overall), fmt(cat["perception"]),
                    fmt(m["risk_status_accuracy"]), fmt(m["risk_type_accuracy"]),
                    fmt(m["risk_source_recognition_score"]), fmt(m["abnormal_action_recognition_score"]),
                    fmt(m["affected_object_recognition_score"]), fmt(cat["cognition"]),
                    fmt(m["risk_description_score"]), fmt(m["consequence_understanding_score"]),
                     fmt(m["factual_boundary_error_score"]), fmt(m["causal_chain_understanding_score"]),
                     fmt(cat["temporal_grounding"]), fmt(m["time_iou_score"]),
                     fmt(m["start_time_score"]), fmt(m["end_time_score"]), fmt(m["duration_score"]),
                     fmt(cat["intervention_planning"]),
                    fmt(m["person_solution_score"]), fmt(m["hazard_solution_score"]),
                    fmt(m["overall_solution_score"])]
        lines.append("<tr><td>" + "</td><td>".join(vals) + "</td></tr>")
    lines.append("</tbody>")
    lines.append("</table>")
    lines.append("")
    sections.append("\n".join(lines))

content = header.format(
    time=datetime.now(timezone.utc).isoformat(),
    title=args.title,
    evaluation_label=args.evaluation_label,
    gt_label=args.gt_label,
    excluded_label=args.excluded_label,
)
content += "\n\n".join(sections) + "\n"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(content, encoding="utf-8")
print(f"{OUTPUT_FILE} updated")

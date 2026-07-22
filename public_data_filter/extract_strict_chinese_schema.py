#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_outputs" / "persistent_full"
DEFAULT_REVIEWS_ROOT = DEFAULT_OUTPUT_ROOT / "reviews"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen__Qwen3-8B-Base"
DEFAULT_OUTPUT_STEM = "strict_english_schema"

SCENES = ("dining room", "kitchen", "study", "balcony", "living room", "bathroom", "yard")
SUBJECTS = ("child", "older adult", "young adult", "middle-aged adult")
RISK_TYPES = (
    "fall/instability",
    "heat/fire source",
    "collision/crush injury",
    "sharp-object danger",
    "electrical safety",
    "poisoning/accidental ingestion",
    "interpersonal conflict",
)
NO_RISK_VALUE = "No"
YES_RISK_VALUE = "Yes"
NO_RISK_TYPE = "None"
NO_INTERVAL_VALUE = "None"
UNKNOWN_INTERVAL_VALUE = "Unknown"
NO_SOLUTION_VALUE = "None"

SCENE_MAP = {
    "dining room": "dining room",
    "餐厅": "dining room",
    "kitchen": "kitchen",
    "厨房": "kitchen",
    "study": "study",
    "书房": "study",
    "balcony": "balcony",
    "阳台": "balcony",
    "living room": "living room",
    "客厅": "living room",
    "bathroom": "bathroom",
    "toilet": "bathroom",
    "restroom": "bathroom",
    "卫生间": "bathroom",
    "yard": "yard",
    "courtyard": "yard",
    "garden": "yard",
    "庭院": "yard",
}

SUBJECT_MAP = {
    "child": "child",
    "children": "child",
    "kid": "child",
    "儿童": "child",
    "older adult": "older adult",
    "elderly": "older adult",
    "senior": "older adult",
    "old adult": "older adult",
    "老年人": "older adult",
    "young adult": "young adult",
    "adult": "young adult",
    "年轻人": "young adult",
    "middle-aged adult": "middle-aged adult",
    "middle aged adult": "middle-aged adult",
    "middle-aged": "middle-aged adult",
    "中年人": "middle-aged adult",
}

RISK_TYPE_MAP = {
    "fall/instability": "fall/instability",
    "fall": "fall/instability",
    "instability": "fall/instability",
    "跌倒失稳": "fall/instability",
    "高温火源": "heat/fire source",
    "heat/fire source": "heat/fire source",
    "heat": "heat/fire source",
    "fire": "heat/fire source",
    "burn": "heat/fire source",
    "collision/crush injury": "collision/crush injury",
    "collision": "collision/crush injury",
    "crush injury": "collision/crush injury",
    "碰撞砸伤": "collision/crush injury",
    "sharp-object danger": "sharp-object danger",
    "sharp object danger": "sharp-object danger",
    "sharp": "sharp-object danger",
    "cut": "sharp-object danger",
    "锐器危险": "sharp-object danger",
    "electrical safety": "electrical safety",
    "electric": "electrical safety",
    "electrical": "electrical safety",
    "用电安全": "electrical safety",
    "poisoning/accidental ingestion": "poisoning/accidental ingestion",
    "poisoning": "poisoning/accidental ingestion",
    "accidental ingestion": "poisoning/accidental ingestion",
    "ingestion": "poisoning/accidental ingestion",
    "中毒误食": "poisoning/accidental ingestion",
    "interpersonal conflict": "interpersonal conflict",
    "conflict": "interpersonal conflict",
    "fight": "interpersonal conflict",
    "人际冲突": "interpersonal conflict",
}


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use a text language model to re-extract existing five-agent review outputs into "
            "a strict Chinese risk schema, with validation and resumable JSONL output."
        )
    )
    parser.add_argument("--reviews-root", type=Path, default=DEFAULT_REVIEWS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the workload into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index for the current worker.")
    parser.add_argument("--merge-shards", action="store_true", help="Merge per-shard outputs back into the base output stem.")
    parser.add_argument("--no-llm", action="store_true", help="Only use deterministic parser/normalizer fallback.")
    parser.add_argument("--strict-llm", action="store_true", help="Fail a record if LLM output is invalid instead of using fallback.")
    parser.add_argument("--wait-for-model", action="store_true", help="Wait until all model shards referenced by the index exist.")
    parser.add_argument("--wait-timeout-seconds", type=int, default=0, help="0 means wait forever when --wait-for-model is set.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shard_output_stem(base_stem: str, shard_index: int, num_shards: int) -> str:
    if num_shards <= 1:
        return base_stem
    return f"{base_stem}.shard{shard_index:02d}of{num_shards:02d}"


def output_paths(output_root: Path, output_stem: str) -> tuple[Path, Path, Path]:
    return (
        output_root / f"{output_stem}.jsonl",
        output_root / f"{output_stem}.json",
        output_root / f"{output_stem}.csv",
    )


def shard_summaries(summary_paths: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    if num_shards <= 1:
        return summary_paths
    return summary_paths[shard_index::num_shards]


def discover_review_summaries(reviews_root: Path) -> list[Path]:
    paths: list[Path] = []
    for review_dir in sorted(path for path in reviews_root.rglob("*") if path.is_dir()):
        repaired = review_dir / "review_summary.repaired.json"
        original = review_dir / "review_summary.json"
        if repaired.exists():
            paths.append(repaired)
        elif original.exists():
            paths.append(original)
    return paths


def video_relpath_from_summary(summary_path: Path, payload: dict[str, Any]) -> str:
    video_path = str(payload.get("manifest", {}).get("video_path", "")).strip()
    public_root = PROJECT_ROOT / "data" / "public_data"
    if video_path:
        try:
            return str(Path(video_path).resolve().relative_to(public_root.resolve()))
        except ValueError:
            return video_path
    parts = summary_path.parts
    if "reviews" in parts:
        idx = parts.index("reviews")
        return str(Path(*parts[idx + 1 :]).with_suffix(".mp4"))
    return str(summary_path)


def normalize_none(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "n/a", "na", "null", "unknown"}:
        return ""
    return text


def normalize_by_map(value: Any, mapping: dict[str, str], allowed: tuple[str, ...], default: str) -> str:
    text = normalize_none(value)
    if not text:
        return default
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if text in allowed:
        return text
    if lowered in mapping:
        return mapping[lowered]
    for key, mapped in mapping.items():
        if key and key in lowered:
            return mapped
    for item in allowed:
        if item in text:
            return item
    return default


def normalize_risk(value: Any, fallback: str = "无") -> str:
    text = str(value or "").strip().lower()
    compact = re.sub(r"[\s:：,，。.;；]+", "", text)
    if compact in {"有", "yes", "true", "risk", "risky", "存在", "有风险"}:
        return YES_RISK_VALUE
    if compact in {"无", "no", "false", "normal", "safe", "none", "无风险"}:
        return NO_RISK_VALUE
    if text.startswith("yes") or "有风险" in text or "存在风险" in text:
        return YES_RISK_VALUE
    if text.startswith("no") or "无风险" in text or "不存在风险" in text:
        return NO_RISK_VALUE
    return fallback


def normalize_interval(value: Any, has_risk: bool) -> str:
    if not has_risk:
        return NO_INTERVAL_VALUE
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "n/a", "na", "null"}:
        return UNKNOWN_INTERVAL_VALUE
    if "未知" in text or "unknown" in text.lower():
        return UNKNOWN_INTERVAL_VALUE
    intervals: list[str] = []
    for start, end in re.findall(r"\[?\s*(\d+(?:\.\d+)?)\s*[,，-]\s*(\d+(?:\.\d+)?)\s*\]?", text):
        left = int(round(float(start)))
        right = int(round(float(end)))
        if right < left:
            left, right = right, left
        intervals.append(f"[{left},{right}]")
    if intervals:
        return ";".join(intervals)
    single = re.search(r"(\d+(?:\.\d+)?)", text)
    if single:
        second = int(round(float(single.group(1))))
        return f"[{second},{second}]"
    return UNKNOWN_INTERVAL_VALUE


def most_common(values: list[str]) -> str:
    cleaned = [value for value in values if normalize_none(value)]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def first_nonempty(values: list[str]) -> str:
    for value in values:
        cleaned = normalize_none(value)
        if cleaned:
            return cleaned
    return ""


def agent_parsed(payload: dict[str, Any], keep: bool | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("agent_results", []):
        parsed = item.get("parsed_decision", {}) or {}
        if keep is not None and parsed.get("keep") is not keep:
            continue
        rows.append(parsed)
    return rows


def fallback_schema(payload: dict[str, Any]) -> dict[str, Any]:
    majority = payload.get("majority_vote", {}) or {}
    has_risk = majority.get("keep") is True
    risk_value = YES_RISK_VALUE if has_risk else NO_RISK_VALUE
    selected = agent_parsed(payload, keep=has_risk)
    all_agents = agent_parsed(payload)

    if has_risk:
        scene_raw = majority.get("level1_scene_top") or most_common([item.get("level1_scene", "") for item in selected])
        subject_raw = majority.get("level2_subject_top") or most_common([item.get("level2_subject", "") for item in selected])
        risk_type_raw = majority.get("level3_risk_type_top") or most_common([item.get("level3_risk_type", "") for item in selected])
        description = first_nonempty([item.get("risk_description", "") for item in selected])
        interval = first_nonempty([item.get("risk_time_interval", "") for item in selected])
        solution_person = first_nonempty([item.get("solution_for_person", "") for item in selected])
        solution_hazard = first_nonempty([item.get("solution_for_hazard_source", "") for item in selected])
        solution_recurrence = first_nonempty([item.get("solution_to_prevent_recurrence", "") for item in selected])
    else:
        scene_raw = majority.get("normal_level1_scene_top") or most_common([item.get("level1_scene", "") for item in selected or all_agents])
        subject_raw = majority.get("normal_level2_subject_top") or most_common([item.get("level2_subject", "") for item in selected or all_agents])
        risk_type_raw = NO_RISK_TYPE
        descriptions = majority.get("normal_video_descriptions") or []
        description = first_nonempty([*descriptions, *[item.get("normal_video_description", "") for item in selected or all_agents]])
        interval = NO_INTERVAL_VALUE
        solution_person = NO_SOLUTION_VALUE
        solution_hazard = NO_SOLUTION_VALUE
        solution_recurrence = NO_SOLUTION_VALUE

    scene = normalize_by_map(scene_raw, SCENE_MAP, SCENES, "living room")
    subject = normalize_by_map(subject_raw, SUBJECT_MAP, SUBJECTS, "young adult")
    risk_type = (
        normalize_by_map(risk_type_raw, RISK_TYPE_MAP, RISK_TYPES, "fall/instability")
        if has_risk
        else NO_RISK_TYPE
    )
    description = description or (
        "The video contains a household safety risk that needs attention."
        if has_risk
        else "The video shows ordinary household activity without a clear visible risk."
    )

    return {
        "Risk": risk_value,
        "Video content labels": {
            "Level 1 scene": scene,
            "Level 2 subject": subject,
            "Level 3 risk type": risk_type,
        },
        "Normal-video description or risk description": description,
        "Risk localization": normalize_interval(interval, has_risk),
        "Solutions": {
            "For person": normalize_none(solution_person)
            or ("Stop the dangerous behavior and assist the person or caregiver immediately." if has_risk else NO_SOLUTION_VALUE),
            "For hazard source": normalize_none(solution_hazard)
            or ("Remove, isolate, or control the hazard source in the environment." if has_risk else NO_SOLUTION_VALUE),
            "Prevent recurrence": normalize_none(solution_recurrence)
            or ("Improve the environment setup and add reminders, supervision, or preventive safeguards." if has_risk else NO_SOLUTION_VALUE),
        },
    }


def normalize_schema(candidate: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    risk = normalize_risk(candidate.get("Risk") or candidate.get("风险"), fallback=fallback["Risk"])
    has_risk = risk == YES_RISK_VALUE
    labels = candidate.get("Video content labels") or candidate.get("视频内容标注") or {}
    if not isinstance(labels, dict):
        labels = {}
    fallback_labels = fallback["Video content labels"]

    scene = normalize_by_map(
        labels.get("Level 1 scene") or labels.get("Level 1 场景") or labels.get("Level 1") or labels.get("scene") or labels.get("场景"),
        SCENE_MAP,
        SCENES,
        fallback_labels["Level 1 scene"],
    )
    subject = normalize_by_map(
        labels.get("Level 2 subject") or labels.get("Level 2 主体") or labels.get("Level 2") or labels.get("subject") or labels.get("主体"),
        SUBJECT_MAP,
        SUBJECTS,
        fallback_labels["Level 2 subject"],
    )
    risk_type = NO_RISK_TYPE
    if has_risk:
        risk_type = normalize_by_map(
            labels.get("Level 3 risk type")
            or labels.get("Level 3 风险类型")
            or labels.get("Level 3")
            or labels.get("risk type")
            or labels.get("风险类型"),
            RISK_TYPE_MAP,
            RISK_TYPES,
            fallback_labels["Level 3 risk type"] if fallback_labels["Level 3 risk type"] != NO_RISK_TYPE else "fall/instability",
        )

    description = normalize_none(
        candidate.get("Normal-video description or risk description") or candidate.get("正常视频描述或风险描述")
    ) or fallback["Normal-video description or risk description"]
    interval = normalize_interval(candidate.get("Risk localization") or candidate.get("风险定位") or fallback["Risk localization"], has_risk)
    solution = candidate.get("Solutions") or candidate.get("解决方案") or {}
    if not isinstance(solution, dict):
        solution = {}
    fallback_solution = fallback["Solutions"]
    if has_risk:
        solution_person = normalize_none(solution.get("For person") or solution.get("对人")) or fallback_solution["For person"]
        solution_hazard = normalize_none(solution.get("For hazard source") or solution.get("对危险源")) or fallback_solution["For hazard source"]
        solution_recurrence = normalize_none(solution.get("Prevent recurrence") or solution.get("防止危险复发")) or fallback_solution["Prevent recurrence"]
    else:
        solution_person = NO_SOLUTION_VALUE
        solution_hazard = NO_SOLUTION_VALUE
        solution_recurrence = NO_SOLUTION_VALUE
        interval = NO_INTERVAL_VALUE

    return {
        "Risk": risk,
        "Video content labels": {
            "Level 1 scene": scene,
            "Level 2 subject": subject,
            "Level 3 risk type": risk_type,
        },
        "Normal-video description or risk description": description,
        "Risk localization": interval,
        "Solutions": {
            "For person": solution_person,
            "For hazard source": solution_hazard,
            "Prevent recurrence": solution_recurrence,
        },
    }


def validate_schema(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    risk = item.get("Risk")
    labels = item.get("Video content labels", {}) or {}
    solution = item.get("Solutions", {}) or {}
    if risk not in {YES_RISK_VALUE, NO_RISK_VALUE}:
        errors.append("Risk must be Yes or No")
    if labels.get("Level 1 scene") not in SCENES:
        errors.append("Level 1 scene is outside allowed taxonomy")
    if labels.get("Level 2 subject") not in SUBJECTS:
        errors.append("Level 2 subject is outside allowed taxonomy")
    if risk == YES_RISK_VALUE and labels.get("Level 3 risk type") not in RISK_TYPES:
        errors.append("Level 3 risk type is outside allowed taxonomy")
    if risk == NO_RISK_VALUE and labels.get("Level 3 risk type") != NO_RISK_TYPE:
        errors.append("No-risk records must use Level 3 risk type=None")
    if not normalize_none(item.get("Normal-video description or risk description")):
        errors.append("Description is empty")
    interval = str(item.get("Risk localization", "")).strip()
    if risk == YES_RISK_VALUE and not (interval == UNKNOWN_INTERVAL_VALUE or re.fullmatch(r"\[\d+,\d+\](;\[\d+,\d+\])*", interval)):
        errors.append("Risk interval must be integer intervals like [1,3] or Unknown")
    if risk == NO_RISK_VALUE and interval != NO_INTERVAL_VALUE:
        errors.append("No-risk records must use Risk localization=None")
    for key in ("For person", "For hazard source", "Prevent recurrence"):
        if not normalize_none(solution.get(key)):
            errors.append(f"Solutions.{key} is empty")
    return errors


def compact_source_for_llm(payload: dict[str, Any]) -> dict[str, Any]:
    agents: list[dict[str, Any]] = []
    for item in payload.get("agent_results", []):
        parsed = item.get("parsed_decision", {}) or {}
        agents.append(
            {
                "perspective": item.get("perspective", {}).get("name"),
                "model": item.get("model", {}).get("model_id"),
                "risk_presence": parsed.get("risk_presence"),
                "level1_scene": parsed.get("level1_scene"),
                "level2_subject": parsed.get("level2_subject"),
                "level3_risk_type": parsed.get("level3_risk_type"),
                "risk_description": parsed.get("risk_description"),
                "risk_time_interval": parsed.get("risk_time_interval"),
                "solution_for_person": parsed.get("solution_for_person"),
                "solution_for_hazard_source": parsed.get("solution_for_hazard_source"),
                "solution_to_prevent_recurrence": parsed.get("solution_to_prevent_recurrence"),
                "normal_video_description": parsed.get("normal_video_description"),
            }
        )
    return {
        "majority_vote": payload.get("majority_vote", {}),
        "agents": agents,
    }


def build_extraction_prompt(payload: dict[str, Any], fallback: dict[str, Any]) -> str:
    source = compact_source_for_llm(payload)
    return (
        "You are a strict information extraction model.\n"
        "Based on the outputs of five video-risk review agents, produce one normalized English JSON object.\n"
        "Only extract, normalize, and briefly summarize. Do not add unsupported facts.\n"
        "The final risk field must follow the majority-vote conclusion in majority_vote.risk_presence / keep.\n\n"
        "The allowed taxonomy is strictly limited to:\n"
        f"Level 1 scene: {' / '.join(SCENES)}\n"
        f"Level 2 subject: {' / '.join(SUBJECTS)}\n"
        f"Level 3 risk type: {' / '.join(RISK_TYPES)} / {NO_RISK_TYPE}. If Risk is {NO_RISK_VALUE}, then Level 3 risk type must be {NO_RISK_TYPE}.\n\n"
        "Output requirements:\n"
        "- Output JSON only. No markdown. No explanation.\n"
        "- The JSON keys must exactly match the structure below.\n"
        "- All free-text fields must be written in English.\n"
        "- The short summary field should be 1-3 English sentences.\n"
        f"- For no-risk videos, Risk localization must be {NO_INTERVAL_VALUE} and all three solution fields must be {NO_SOLUTION_VALUE}.\n"
        f"- For risky videos, Risk localization should be integer-second intervals like [1,3]; use semicolons for multiple intervals; if the source has no reliable time evidence, use {UNKNOWN_INTERVAL_VALUE}.\n\n"
        "The JSON must exactly follow this structure:\n"
        "{\n"
        f'  "Risk": "{YES_RISK_VALUE}/{NO_RISK_VALUE}",\n'
        '  "Video content labels": {\n'
        '    "Level 1 scene": "dining room/kitchen/study/balcony/living room/bathroom/yard",\n'
        '    "Level 2 subject": "child/older adult/young adult/middle-aged adult",\n'
        '    "Level 3 risk type": "fall/instability/heat/fire source/collision/crush injury/sharp-object danger/electrical safety/poisoning/accidental ingestion/interpersonal conflict/None"\n'
        "  },\n"
        '  "Normal-video description or risk description": "Summarize the video or risk in 1-3 English sentences",\n'
        '  "Risk localization": "For risky videos, use integer-second intervals like [1,3], semicolon-separated if multiple; if evidence is missing, use Unknown; for no-risk videos, use None",\n'
        '  "Solutions": {\n'
        '    "For person": "For risky videos, give a concrete action; for no-risk videos, use None",\n'
        '    "For hazard source": "For risky videos, give a concrete action; for no-risk videos, use None",\n'
        '    "Prevent recurrence": "For risky videos, give a prevention measure; for no-risk videos, use None"\n'
        "  }\n"
        "}\n\n"
        "A deterministic fallback draft is provided below. Only revise it when the agent evidence clearly supports a more accurate normalized result:\n"
        f"{json.dumps(fallback, ensure_ascii=False)}\n\n"
        "The five agent outputs and the majority-vote source are:\n"
        f"{json.dumps(source, ensure_ascii=False)}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(cleaned[start : end + 1])


class QwenExtractor:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import logging as hf_logging
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_logging.set_verbosity_error()
        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.max_new_tokens = args.max_new_tokens

    def render_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        rendered = self.render_prompt(prompt)
        inputs = self.tokenizer([rendered], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return extract_json_object(response)


def model_ready(model_path: Path) -> tuple[bool, str]:
    index_path = model_path / "model.safetensors.index.json"
    config_path = model_path / "config.json"
    tokenizer_path = model_path / "tokenizer.json"
    if not config_path.exists():
        return False, f"missing {config_path}"
    if not tokenizer_path.exists():
        return False, f"missing {tokenizer_path}"
    if index_path.exists():
        try:
            index = read_json(index_path)
        except json.JSONDecodeError as exc:
            return False, f"invalid index json: {exc}"
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (model_path / name).exists()]
        if missing:
            return False, f"missing {len(missing)} shard(s), first: {missing[0]}"
        for name in shards:
            path = model_path / name
            try:
                with path.open("rb") as handle:
                    header_size_bytes = handle.read(8)
                    header_start = handle.read(1)
            except OSError as exc:
                return False, f"cannot read {name}: {exc}"
            if len(header_size_bytes) != 8 or header_start != b"{":
                return False, f"invalid safetensors header in {name}"
    elif not any(model_path.glob("*.safetensors")) and not (model_path / "pytorch_model.bin").exists():
        return False, "missing model weight files"
    return True, "ready"


def wait_for_model_if_needed(args: argparse.Namespace) -> None:
    if args.no_llm:
        return
    started = time.time()
    while True:
        ready, detail = model_ready(args.model_path)
        if ready:
            print(f"[model] {args.model_path} is ready", flush=True)
            return
        if not args.wait_for_model:
            raise FileNotFoundError(f"Model is not ready: {detail}. Use --wait-for-model to wait.")
        elapsed = time.time() - started
        if args.wait_timeout_seconds and elapsed >= args.wait_timeout_seconds:
            raise TimeoutError(f"Timed out waiting for model after {elapsed:.0f}s: {detail}")
        print(f"[model] waiting for {args.model_path}: {detail}", flush=True)
        time.sleep(max(1, args.poll_seconds))


def existing_jsonl_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows[str(item.get("video_relpath"))] = item
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "processed_videos": len(rows),
        "risk_counts": dict(Counter(row["Risk"] for row in rows)),
        "method_counts": dict(Counter(row["extraction_method"] for row in rows)),
        "rows_with_validation_errors": sum(bool(row["validation_errors"]) for row in rows),
    }


def build_output_payload(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    effective_output_stem: str,
    shard_index: int | None,
    merged_from_shards: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "manifest": {
            "generated_at": utc_tag(),
            "reviews_root": str(args.reviews_root),
            "output_root": str(args.output_root),
            "output_stem": args.output_stem,
            "effective_output_stem": effective_output_stem,
            "model_path": None if args.no_llm else str(args.model_path),
            "num_shards": args.num_shards,
            "shard_index": shard_index,
            "merged_from_shards": merged_from_shards or [],
            "schema": {
                "Risk": [YES_RISK_VALUE, NO_RISK_VALUE],
                "Level 1 scene": list(SCENES),
                "Level 2 subject": list(SUBJECTS),
                "Level 3 risk type": [*RISK_TYPES, NO_RISK_TYPE],
                "Risk localization": f"Integer intervals like [start,end]; use semicolons for multiple; no-risk is {NO_INTERVAL_VALUE}; missing evidence is {UNKNOWN_INTERVAL_VALUE}",
            },
        },
        "summary": summarize_rows(rows),
        "results": rows,
    }


def formatted_text(schema: dict[str, Any]) -> str:
    labels = schema["Video content labels"]
    solution = schema["Solutions"]
    return "\n".join(
        [
            f"Risk: {schema['Risk']}",
            "Video content labels:",
            f"Level 1: scene: {labels['Level 1 scene']}",
            f"Level 2: subject: {labels['Level 2 subject']}",
            f"Level 3: risk type: {labels['Level 3 risk type']}",
            f"Normal-video description or risk description: {schema['Normal-video description or risk description']}",
            f"Risk localization: {schema['Risk localization']}",
            "Solutions:",
            f"For person: {solution['For person']}",
            f"For hazard source: {solution['For hazard source']}",
            f"Prevent recurrence: {solution['Prevent recurrence']}",
        ]
    )


def build_row(
    summary_path: Path,
    payload: dict[str, Any],
    schema: dict[str, Any],
    method: str,
    errors: list[str],
) -> dict[str, Any]:
    labels = schema["Video content labels"]
    solution = schema["Solutions"]
    video_path = str(payload.get("manifest", {}).get("video_path", "")).strip()
    return {
        "video_relpath": video_relpath_from_summary(summary_path, payload),
        "video_path": video_path,
        "summary_json": str(summary_path),
        "Risk": schema["Risk"],
        "Level 1 scene": labels["Level 1 scene"],
        "Level 2 subject": labels["Level 2 subject"],
        "Level 3 risk type": labels["Level 3 risk type"],
        "Normal-video description or risk description": schema["Normal-video description or risk description"],
        "Risk localization": schema["Risk localization"],
        "Solutions-For person": solution["For person"],
        "Solutions-For hazard source": solution["For hazard source"],
        "Solutions-Prevent recurrence": solution["Prevent recurrence"],
        "formatted_text": formatted_text(schema),
        "extraction_method": method,
        "validation_errors": errors,
        "source_majority_vote": payload.get("majority_vote", {}),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "video_relpath",
        "video_path",
        "Risk",
        "Level 1 scene",
        "Level 2 subject",
        "Level 3 risk type",
        "Normal-video description or risk description",
        "Risk localization",
        "Solutions-For person",
        "Solutions-For hazard source",
        "Solutions-Prevent recurrence",
        "formatted_text",
        "extraction_method",
        "validation_errors",
        "summary_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def process_one(
    summary_path: Path,
    extractor: QwenExtractor | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = read_json(summary_path)
    fallback = fallback_schema(payload)
    method = "fallback"
    errors: list[str] = []
    schema = fallback

    if extractor is not None:
        prompt = build_extraction_prompt(payload, fallback)
        try:
            candidate = extractor.generate_json(prompt)
            schema = normalize_schema(candidate, fallback)
            errors = validate_schema(schema)
            method = "llm"
            if errors and args.strict_llm:
                raise ValueError("; ".join(errors))
            if errors:
                schema = fallback
                method = "fallback_after_invalid_llm"
                errors = validate_schema(schema)
        except Exception as exc:
            if args.strict_llm:
                raise
            schema = fallback
            method = f"fallback_after_llm_error:{type(exc).__name__}"
            errors = validate_schema(schema)

    return build_row(summary_path, payload, schema, method, errors)


def merge_shard_outputs(args: argparse.Namespace) -> int:
    rows_by_video: dict[str, dict[str, Any]] = {}
    merged_from_shards: list[str] = []
    missing_files: list[str] = []
    for shard_index in range(args.num_shards):
        stem = shard_output_stem(args.output_stem, shard_index, args.num_shards)
        jsonl_path = args.output_root / f"{stem}.jsonl"
        if not jsonl_path.exists():
            missing_files.append(str(jsonl_path))
            continue
        merged_from_shards.append(str(jsonl_path))
        rows_by_video.update(existing_jsonl_rows(jsonl_path))
    if missing_files:
        raise FileNotFoundError(
            "Cannot merge shard outputs because some shard jsonl files are missing: "
            + ", ".join(missing_files[:3])
            + (" ..." if len(missing_files) > 3 else "")
        )

    rows = [rows_by_video[key] for key in sorted(rows_by_video)]
    output_jsonl, output_json, output_csv = output_paths(args.output_root, args.output_stem)
    payload = build_output_payload(
        rows,
        args,
        effective_output_stem=args.output_stem,
        shard_index=None,
        merged_from_shards=merged_from_shards,
    )
    write_jsonl(output_jsonl, rows)
    write_json(output_json, payload)
    write_csv(output_csv, rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(str(output_json), flush=True)
    print(str(output_csv), flush=True)
    print(str(output_jsonl), flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    args.reviews_root = args.reviews_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()

    if not args.reviews_root.exists():
        raise FileNotFoundError(f"reviews root does not exist: {args.reviews_root}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    if args.merge_shards:
        return merge_shard_outputs(args)

    wait_for_model_if_needed(args)
    extractor = None if args.no_llm else QwenExtractor(args)
    effective_output_stem = shard_output_stem(args.output_stem, args.shard_index, args.num_shards)
    output_jsonl, output_json, output_csv = output_paths(args.output_root, effective_output_stem)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = discover_review_summaries(args.reviews_root)
    if args.limit is not None:
        summaries = summaries[: max(0, args.limit)]
    summaries = shard_summaries(summaries, args.num_shards, args.shard_index)

    rows_by_video = existing_jsonl_rows(output_jsonl) if args.resume else {}
    if args.resume and args.num_shards > 1 and not rows_by_video:
        base_jsonl = args.output_root / f"{args.output_stem}.jsonl"
        if base_jsonl.exists():
            base_rows = existing_jsonl_rows(base_jsonl)
            if base_rows:
                seed_rows: dict[str, dict[str, Any]] = {}
                for summary_path in summaries:
                    payload = read_json(summary_path)
                    relpath = video_relpath_from_summary(summary_path, payload)
                    row = base_rows.get(relpath)
                    if row is not None:
                        seed_rows[relpath] = row
                if seed_rows:
                    rows_by_video.update(seed_rows)
                    write_jsonl(output_jsonl, [seed_rows[key] for key in sorted(seed_rows)])

    with output_jsonl.open("a", encoding="utf-8") as jsonl_handle:
        for index, summary_path in enumerate(summaries, start=1):
            payload = read_json(summary_path)
            relpath = video_relpath_from_summary(summary_path, payload)
            if args.resume and relpath in rows_by_video:
                continue
            row = process_one(summary_path, extractor, args)
            rows_by_video[row["video_relpath"]] = row
            jsonl_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_handle.flush()
            if index == 1 or index % 20 == 0:
                if args.num_shards > 1:
                    print(
                        f"[shard {args.shard_index + 1}/{args.num_shards}] "
                        f"[{index}/{len(summaries)}] {row['video_relpath']} -> "
                        f"{row['Risk']} ({row['extraction_method']})",
                        flush=True,
                    )
                else:
                    print(f"[{index}/{len(summaries)}] {row['video_relpath']} -> {row['Risk']} ({row['extraction_method']})", flush=True)

    rows = [rows_by_video[key] for key in sorted(rows_by_video)]
    payload = build_output_payload(
        rows,
        args,
        effective_output_stem=effective_output_stem,
        shard_index=args.shard_index if args.num_shards > 1 else None,
    )
    write_json(output_json, payload)
    write_csv(output_csv, rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(str(output_json), flush=True)
    print(str(output_csv), flush=True)
    print(str(output_jsonl), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

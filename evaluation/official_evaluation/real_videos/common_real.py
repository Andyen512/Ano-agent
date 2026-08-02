from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(EVALUATION_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.common import (
    RISK_STATUS_NO_ANOMALY,
    RISK_STATUS_OCCURRED,
    RISK_STATUS_POTENTIAL,
    compact_text,
    normalize_risk_status_label,
    normalize_risk_type_label,
    normalize_text,
    parse_prediction_solution_sections,
)

RISK_TYPE_CN_MAP: dict[str, str] = {
    # fall/instability
    "跌倒失稳": "fall_instability",
    "跌倒": "fall_instability",
    "摔倒": "fall_instability",
    # animal attack/biosecurity risk
    "动物攻击": "animal attack/biosecurity",
    "动物袭击": "animal attack/biosecurity",
    # heat/fire source
    "高温火源": "heat_source",
    "起火冒烟": "heat_source",
    "火灾": "heat_source",
    "起火": "heat_source",
    "爆炸": "heat_source",
    "氢气球爆炸": "heat_source",
    # collision/crush injury
    "碰撞砸伤": "collision_or_struck",
    "夹伤": "collision_or_struck",
    "碰撞": "collision_or_struck",
    "高空坠物": "collision_or_struck",
    "车辆异常": "collision_or_struck",
    "车辆失控": "collision_or_struck",
    # sharp-object danger
    "锐器危险": "sharp_object",
    "锐器": "sharp_object",
    # electrical safety
    "用电安全": "electrical_safety",
    "电气安全": "electrical_safety",
    "电气": "electrical_safety",
    # poisoning/accidental ingestion
    "异物卡喉": "poisoning/accidental ingestion",
    "异物窒息": "poisoning/accidental ingestion",
    "儿童窒息": "poisoning/accidental ingestion",
    "窒息": "poisoning/accidental ingestion",
    "一氧化碳中毒": "poisoning/accidental ingestion",
    "中毒": "poisoning/accidental ingestion",
    # stranger theft
    "盗窃": "stranger_theft",
    "偷窃": "stranger_theft",
    "盗窃风险": "stranger_theft",
    "盗窃行为": "stranger_theft",
    "盗窃抢劫": "stranger_theft",
    "伪装盗窃": "stranger_theft",
    "抢劫": "stranger_theft",
    "未经授权使用": "stranger_theft",
    # interpersonal conflict
    "入侵盗窃": "interpersonal conflict",
    "非法入侵": "interpersonal conflict",
    "入侵": "interpersonal conflict",
    "入侵行为": "interpersonal conflict",
    "暴力入侵": "interpersonal conflict",
    "异常入侵": "interpersonal conflict",
    "陌生人入侵": "interpersonal conflict",
    "隐私侵犯": "interpersonal conflict",
    "暴力冲突": "interpersonal conflict",
    "异常行为": "interpersonal conflict",
    "危险驾驶": "collision_or_struck",
    "溺水": "interpersonal conflict",
    "溺水风险": "interpersonal conflict",
    "安全": "interpersonal conflict",
    "雷电灾害": "interpersonal conflict",
    "交通事故": "collision_or_struck",
}


def normalize_risk_type_cn(label: str) -> str:
    if not label:
        return ""
    clean = label.strip()
    if clean in RISK_TYPE_CN_MAP:
        return RISK_TYPE_CN_MAP[clean]
    mapped = normalize_risk_type_label(clean)
    if mapped:
        return mapped
    return ""

OFFICIAL_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_GT_DIR = OFFICIAL_EVAL_DIR
DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "prediction" / "real_videos"


@dataclass
class RealVideoGT:
    video_id: str
    risk_status: str

    perception: dict[str, Any] = field(default_factory=dict)
    cognition: dict[str, Any] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_type(self) -> str:
        return self.perception.get("risk_type", "")

    @property
    def scene(self) -> str:
        return self.perception.get("scene", "")

    @property
    def risk_sources(self) -> tuple[str, ...]:
        return tuple(self.perception.get("risk_sources", []))

    @property
    def abnormal_actions(self) -> tuple[str, ...]:
        return tuple(self.perception.get("abnormal_actions", []))

    @property
    def affected_objects(self) -> tuple[str, ...]:
        return tuple(self.perception.get("affected_objects", []))

    @property
    def risk_description(self) -> str:
        return self.cognition.get("risk_description", "")

    @property
    def consequence_understanding(self) -> tuple[str, ...]:
        return tuple(self.cognition.get("consequence_understanding", []))

    @property
    def causal_chain(self) -> tuple[dict[str, str], ...]:
        return tuple(self.cognition.get("causal_chain", []))

    @property
    def causal_chain_text(self) -> str:
        return self.cognition.get("causal_chain_text", "")

    @property
    def event_summary(self) -> str:
        return self.cognition.get("event_summary", "")

    @property
    def time_spans(self) -> list[list[float]]:
        return self.grounding.get("time_spans", [])

    @property
    def start_time(self) -> float | None:
        return self.grounding.get("start_time")

    @property
    def end_time(self) -> float | None:
        return self.grounding.get("end_time")

    @property
    def duration(self) -> float | None:
        return self.grounding.get("duration")

    @property
    def person_solution(self) -> str:
        return self.planning.get("person_solution", "")

    @property
    def hazard_solution(self) -> str:
        return self.planning.get("hazard_solution", "")

    @property
    def prevention_solution(self) -> str:
        return self.planning.get("prevention_solution", "")

    @property
    def solution_sections(self) -> dict[str, str]:
        return {
            "person_solution": self.person_solution,
            "hazard_solution": self.hazard_solution,
            "prevention_solution": self.prevention_solution,
        }

    def as_summary_gt(self) -> dict[str, Any]:
        return {
            "risk_status": self.risk_status,
            "risk_type": self.risk_type,
            "risk_description": self.risk_description,
            "risk_sources": list(self.risk_sources),
            "abnormal_actions": list(self.abnormal_actions),
            "affected_objects": list(self.affected_objects),
            "consequences": list(self.consequence_understanding),
            "causal_chain": list(self.causal_chain),
            "solution_sections": self.solution_sections,
        }


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def load_real_video_gt(gt_dir: Path | None = None) -> dict[str, RealVideoGT]:
    gt_dir = gt_dir or DEFAULT_GT_DIR
    dimension_files = {
        "perception": gt_dir / "perception" / "perception_gt.json",
        "cognition": gt_dir / "cognition" / "cognition_gt.json",
        "grounding": gt_dir / "grounding" / "grounding_gt.json",
        "planning": gt_dir / "planning" / "planning_gt.json",
    }

    use_legacy = all(p.exists() for p in dimension_files.values())

    entries: dict[str, dict[str, dict[str, Any]]] = {}

    if use_legacy:
        for dim, path in dimension_files.items():
            for item in _load_json_array(path):
                video_id = item["video_id"]
                if video_id not in entries:
                    entries[video_id] = {}
                entries[video_id][dim] = item
    else:
        for json_path in sorted(gt_dir.rglob("perception_gt.json")):
            source_dir = json_path.parent
            for dim, fname in [
                ("perception", "perception_gt.json"),
                ("cognition", "cognition_gt.json"),
                ("grounding", "grounding_gt.json"),
                ("planning", "planning_gt.json"),
            ]:
                dim_path = source_dir / fname
                if not dim_path.exists():
                    continue
                for item in _load_json_array(dim_path):
                    video_id = item["video_id"]
                    if video_id not in entries:
                        entries[video_id] = {}
                    entries[video_id][dim] = item

    if not entries:
        raise FileNotFoundError(f"No GT files found in {gt_dir}")

    result: dict[str, RealVideoGT] = {}
    for video_id, dims in entries.items():
        perception = dims.get("perception", {})
        cognition = dims.get("cognition", {})
        grounding = dims.get("grounding", {})
        planning = dims.get("planning", {})

        risk_status = normalize_risk_status_label(
            perception.get("risk_status")
            or grounding.get("risk_status")
            or planning.get("risk_status")
            or ""
        )
        if not risk_status:
            continue

        result[video_id] = RealVideoGT(
            video_id=video_id,
            risk_status=risk_status,
            perception={k: v for k, v in perception.items() if k != "video_id"},
            cognition={k: v for k, v in cognition.items() if k != "video_id"},
            grounding={k: v for k, v in grounding.items() if k != "video_id"},
            planning={k: v for k, v in planning.items() if k != "video_id"},
        )

    return result


def basename_from_video_id(video_id: str) -> str:
    return Path(video_id).name


def match_prediction_to_gt(
    prediction_filename_stem: str,
    prediction_video_path: str | None,
    gt_by_video_id: dict[str, RealVideoGT],
    gt_by_basename: dict[str, list[str]] | None = None,
) -> str | None:
    if prediction_filename_stem in gt_by_video_id:
        return prediction_filename_stem

    # Generated videos are stored as
    # generated_videos/<status>/<scene>/<subject>/<risk>/<description>/<generator>/<file>,
    # while their released GT uses the status component immediately before the file.
    # Resolve that layout from the full prediction path before falling back to an
    # ambiguous basename match.
    if prediction_video_path:
        path_parts = Path(prediction_video_path).parts
        try:
            generated_index = path_parts.index("generated_videos")
        except ValueError:
            generated_index = -1
        if generated_index >= 0:
            generated_parts = list(path_parts[generated_index + 1:])
            if len(generated_parts) >= 3 and generated_parts[0] in {
                "normal", "risk_only", "abnormal",
            }:
                status = generated_parts[0]
                generated_id = "/".join(
                    [
                        "generated_videos",
                        *generated_parts[1:-1],
                        status,
                        generated_parts[-1].removesuffix(Path(generated_parts[-1]).suffix),
                    ]
                )
                if generated_id in gt_by_video_id:
                    return generated_id

                # Some generated GT entries use a different description path
                # while retaining the same generator, status, and file name.
                # Use that combination only when it identifies one GT entry.
                basename_index = gt_by_basename or _build_basename_index(gt_by_video_id)
                generator = generated_parts[-2]
                filename = generated_parts[-1].removesuffix(Path(generated_parts[-1]).suffix)
                candidates = [
                    candidate
                    for candidate in basename_index.get(filename, [])
                    if candidate.endswith(f"/{generator}/{status}/{filename}")
                ]
                if len(candidates) == 1:
                    return candidates[0]

    if gt_by_basename is None:
        gt_by_basename = _build_basename_index(gt_by_video_id)

    # Prefer the most specific path when datasets share the same basename
    # (for example, SmartHome-Bench and risk_segment variants).
    candidates = sorted(
        gt_by_basename.get(prediction_filename_stem, []),
        key=len,
        reverse=True,
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and prediction_video_path:
        for c in candidates:
            if c.replace("/", os.sep) in prediction_video_path or c in prediction_video_path:
                return c

    # fallback: try to extract video_id from video_path
    if prediction_video_path:
        # 去掉扩展名
        vp_no_ext = prediction_video_path
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"):
            if vp_no_ext.lower().endswith(ext):
                vp_no_ext = vp_no_ext[:-len(ext)]
                break

        # 尝试找 GT video_id 作为 prediction_video_path 的后缀
        for gt_vid in sorted(gt_by_video_id, key=len, reverse=True):
            if vp_no_ext.endswith(gt_vid) or vp_no_ext.endswith(gt_vid.replace("/", os.sep)):
                return gt_vid

        # 尝试反过来：prediction_video_path 中包含 GT video_id
        for gt_vid in sorted(gt_by_video_id, key=len, reverse=True):
            normalized_gt = gt_vid.replace("/", os.sep).lower()
            normalized_vp = vp_no_ext.lower()
            if normalized_gt in normalized_vp:
                return gt_vid

    return None


def _build_basename_index(gt_by_video_id: dict[str, RealVideoGT]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for video_id in gt_by_video_id:
        basename = basename_from_video_id(video_id)
        index.setdefault(basename, []).append(video_id)
    return index


def parse_prediction_response(prediction_file: dict[str, Any]) -> dict[str, Any] | None:
    response = prediction_file.get("response", "")
    if isinstance(response, dict):
        resp_obj = response
    elif isinstance(response, str) and response.strip():
        try:
            resp_obj = json.loads(response)
        except json.JSONDecodeError:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else ""
                cleaned = cleaned.rsplit("\n```", 1)[0] if cleaned.endswith("```") else cleaned.rstrip("`")
                try:
                    resp_obj = json.loads(cleaned)
                except json.JSONDecodeError:
                    return None
            else:
                return None
    else:
        return None

    if isinstance(resp_obj, dict):
        if "parsed_response" in resp_obj and isinstance(resp_obj["parsed_response"], dict):
            return _normalize_prediction_keys(resp_obj["parsed_response"])
        return _normalize_prediction_keys(resp_obj)

    return None


def _normalize_prediction_keys(pred: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "ab normal_actions": "abnormal_actions",
        "ab normal _actions": "abnormal_actions",
        "abnormal_actions ": "abnormal_actions",
        "affected_objects ": "affected_objects",
        "risk description": "risk_description",
        "risk_type ": "risk_type",
    }

    result: dict[str, Any] = {}
    for k, v in pred.items():
        clean_key = k.strip()
        clean_key = key_map.get(clean_key, clean_key)
        result[clean_key] = v

    if "risk_status" in result and isinstance(result["risk_status"], str):
        result["risk_status"] = normalize_risk_status_label(result["risk_status"])
    if "risk_type" in result and isinstance(result["risk_type"], str):
        result["risk_type"] = normalize_risk_type_cn(result["risk_type"])

    for list_key in ("risk_sources", "abnormal_actions", "affected_objects", "consequence_understanding"):
        if list_key in result:
            val = result[list_key]
            if isinstance(val, str):
                result[list_key] = [val] if val.strip() else []
            elif isinstance(val, list):
                result[list_key] = [str(item) for item in val if item]
            else:
                result[list_key] = []

    if "causal_chain" in result:
        cc = result["causal_chain"]
        if isinstance(cc, str):
            result["causal_chain_text"] = cc
            result["causal_chain"] = _parse_causal_chain_text(cc)
        elif isinstance(cc, list):
            result["causal_chain"] = [
                item if isinstance(item, dict) else {"source": str(item), "action": "", "consequence": ""}
                for item in cc
            ]
        else:
            result["causal_chain"] = []

    if "time_spans" in result:
        raw_spans = result["time_spans"]
        normalized_spans: list[list[float]] = []
        if isinstance(raw_spans, list):
            for item in raw_spans:
                if isinstance(item, dict):
                    start = item.get("start", item.get("start_second"))
                    end = item.get("end", item.get("end_second"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    start, end = item[0], item[1]
                else:
                    continue
                try:
                    normalized_spans.append([float(start), float(end)])
                except (TypeError, ValueError):
                    continue
        result["time_spans"] = normalized_spans
    return result


def _parse_causal_chain_text(text: str) -> list[dict[str, str]]:
    if not text or not text.strip():
        return []
    segments = re.split(r"\s*[→>]\s*", text)
    chain: list[dict[str, str]] = []
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if not seg:
            continue
        if i == 0:
            chain.append({"source": seg, "action": "", "consequence": ""})
        elif i == len(segments) - 1:
            chain.append({"source": "", "action": "", "consequence": seg})
        else:
            chain.append({"source": "", "action": seg, "consequence": ""})
    return chain


def parse_prediction_solution(raw_pred: dict[str, Any]) -> dict[str, str]:
    if "solution" in raw_pred and isinstance(raw_pred["solution"], str) and raw_pred["solution"].strip():
        sections = parse_prediction_solution_sections(raw_pred["solution"])
        if any(sections.values()):
            return sections

    result = {
        "person_solution": "",
        "hazard_solution": "",
        "overall_solution": "",
    }
    direct_map = {
        "person_solution": "person_solution",
        "hazard_solution": "hazard_solution",
        "prevention_solution": "overall_solution",
        "overall_solution": "overall_solution",
    }
    for pred_key, result_key in direct_map.items():
        val = raw_pred.get(pred_key, "")
        if isinstance(val, str) and val.strip():
            result[result_key] = compact_text(val)

    return result


def discover_prediction_files(
    predictions_root: Path | None = None,
    model_id_filter: str | None = None,
) -> list[dict[str, Any]]:
    predictions_root = predictions_root or DEFAULT_PREDICTIONS_ROOT
    payloads: list[dict[str, Any]] = []

    for model_dir in sorted(predictions_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        has_direct_files = any(f.name.endswith(".json") for f in model_dir.iterdir() if f.is_file())

        if has_direct_files:
            for json_file in sorted(model_dir.rglob("*.json")):
                try:
                    payload = json.loads(json_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if "video_path" not in payload and "model_id" not in payload:
                    continue
                if model_id_filter and payload.get("model_id") != model_id_filter:
                    continue
                payload["__source_path__"] = str(json_file)
                payload["__source_model_dir__"] = model_dir.name
                payloads.append(payload)
        else:
            for ts_dir in sorted(model_dir.iterdir()):
                if not ts_dir.is_dir():
                    continue
                for json_file in sorted(ts_dir.rglob("*.json")):
                    try:
                        payload = json.loads(json_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if "video_path" not in payload and "model_id" not in payload:
                        continue
                    if model_id_filter and payload.get("model_id") != model_id_filter:
                        continue
                    payload["__source_path__"] = str(json_file)
                    payload["__source_model_dir__"] = model_dir.name
                    payloads.append(payload)

    return payloads


def extract_video_id_from_prediction(payload: dict[str, Any]) -> str | None:
    source_path = payload.get("__source_path__", "")
    if source_path:
        return Path(source_path).stem
    return None

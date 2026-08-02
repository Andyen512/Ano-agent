#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from benchmark_models import BENCHMARK_MODELS
from model_snapshot import invalid_snapshot_files, is_model_snapshot_complete, missing_snapshot_files


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
MODELS_ROOT = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class BackendSpec:
    name: str
    code_dir: Path
    default_model_id: str
    description: str
    requires_local_model: bool = True


BACKENDS: dict[str, BackendSpec] = {
    "tarsier2": BackendSpec(
        name="tarsier2",
        code_dir=PROJECT_ROOT / "code" / "Tarsier2-7B",
        default_model_id="omni-research/Tarsier2-7b-0115",
        description="Bytedance Tarsier2 single-video inference",
    ),
    "videollama2": BackendSpec(
        name="videollama2",
        code_dir=PROJECT_ROOT / "code" / "VideoLLaMA2-7B",
        default_model_id="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F",
        description="VideoLLaMA2.1 7B 16-frame video chat model",
    ),
    "videollama3": BackendSpec(
        name="videollama3",
        code_dir=PROJECT_ROOT / "code" / "VideoLLaMA3-7B",
        default_model_id="DAMO-NLP-SG/VideoLLaMA3-7B",
        description="VideoLLaMA3 7B video chat model",
    ),
    "videoccam": BackendSpec(
        name="videoccam",
        code_dir=PROJECT_ROOT / "code" / "Video-CCAM-7B",
        default_model_id="JaronTHU/Video-CCAM-7B-v1.2",
        description="Video-CCAM video chat model",
    ),
    "internvl35": BackendSpec(
        name="internvl35",
        code_dir=PROJECT_ROOT / "models" / "OpenGVLab__InternVL3_5-8B",
        default_model_id="OpenGVLab/InternVL3_5-8B",
        description="InternVL3.5 8B multimodal video chat model",
    ),
    "mplugowl3": BackendSpec(
        name="mplugowl3",
        code_dir=PROJECT_ROOT / "code" / "mPLUG-Owl3-7B" / "mPLUG-Owl3",
        default_model_id="mPLUG/mPLUG-Owl3-7B-241101",
        description="mPLUG-Owl3 video chat model",
    ),
    "videollava": BackendSpec(
        name="videollava",
        code_dir=PROJECT_ROOT / "code" / "Video-LLaVA-7B",
        default_model_id="LanguageBind/Video-LLaVA-7B",
        description="Video-LLaVA single-video inference",
    ),
    "videochatgpt": BackendSpec(
        name="videochatgpt",
        code_dir=PROJECT_ROOT / "code" / "Video-ChatGPT-7B",
        default_model_id="MBZUAI/Video-ChatGPT-7B",
        description="Video-ChatGPT single-video inference",
    ),
    "videochat2": BackendSpec(
        name="videochat2",
        code_dir=PROJECT_ROOT / "models" / "OpenGVLab__VideoChat2_HD_stage4_Mistral_7B_hf",
        default_model_id="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
        description="VideoChat2 HD Mistral single-video inference",
    ),
    "minigpt4video": BackendSpec(
        name="minigpt4video",
        code_dir=PROJECT_ROOT / "code" / "MiniGPT4-Video",
        default_model_id="Vision-CAIR/MiniGPT4-Video",
        description="MiniGPT4-Video single-video inference",
    ),
    "qwen25vl": BackendSpec(
        name="qwen25vl",
        code_dir=PROJECT_ROOT / "code" / "Qwen2.5VL-7B",
        default_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Qwen2.5-VL single-video inference",
    ),
    "qwen35vl": BackendSpec(
        name="qwen35vl",
        code_dir=PROJECT_ROOT / "models" / "Qwen" / "Qwen3.5-9B",
        default_model_id="Qwen/Qwen3.5-9B",
        description="Qwen3.5 multimodal single-video inference",
    ),
    "minicpmv45": BackendSpec(
        name="minicpmv45",
        code_dir=PROJECT_ROOT,
        default_model_id="openbmb/MiniCPM-V-4_5",
        description="MiniCPM-V 4.5 efficient image/video understanding model",
    ),
    "commercial": BackendSpec(
        name="commercial",
        code_dir=PROJECT_ROOT,
        default_model_id="gpt-5.4",
        description="Closed-source commercial multimodal inference via llm.py and an OpenAI-compatible API.",
        requires_local_model=False,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified single-video inference entrypoint for supported LifeBench backends."
    )
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--video-path",
        help="Path to an input video/gif/image file.",
    )
    parser.add_argument(
        "--prompt",
        help="Text prompt. Defaults to a generic video description prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Read prompt text from a file instead of --prompt.",
    )
    parser.add_argument(
        "--model-path",
        help="Local path to model weights. If omitted, the runner looks under ./models for a downloaded copy.",
    )
    parser.add_argument(
        "--model-id",
        help="Model id. For local backends this resolves the downloaded checkpoint directory; for commercial it is the remote API model name.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='device_map passed to the backend loader. Default: "auto".',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Set > 0 to enable sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top_p for sampling backends.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video sampling fps for backends that expose it.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=180,
        help="Maximum video frames to load for backends that expose it.",
    )
    parser.add_argument(
        "--merge-size",
        type=int,
        default=2,
        help="VideoLLaMA3 merge_size for video inputs.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
        help="Attention implementation for VideoLLaMA3. Default selects flash_attention_2 when available, otherwise sdpa.",
    )
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Enable flash_attention_2 for VideoLLaMA2 when the dependency is installed.",
    )
    parser.add_argument(
        "--tarsier-config",
        default=str(PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"),
        help="Path to the Tarsier2 data config YAML.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the structured inference result as JSON.",
    )
    parser.add_argument(
        "--projection-path",
        help="Optional projection/adapter checkpoint path for backends such as Video-ChatGPT.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the structured result as JSON instead of only the model response.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List supported backends and exit.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List benchmark model ids and mapped backends, then exit.",
    )
    return parser


def ensure_backend_on_path(backend: BackendSpec) -> None:
    backend_path = str(backend.code_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def sanitize_model_id(model_id: str) -> list[str]:
    variants = [
        model_id.replace("/", "__"),
        model_id.replace("/", "_"),
        model_id.split("/")[-1],
    ]
    # keep order but drop duplicates
    return list(dict.fromkeys(variants))


def candidate_model_dirs(model_id: str) -> list[Path]:
    candidates = [MODELS_ROOT / name for name in sanitize_model_id(model_id)]
    parts = [part for part in model_id.split("/") if part]
    if parts:
        candidates.append(MODELS_ROOT.joinpath(*parts))
    return list(dict.fromkeys(candidates))


def default_download_dir(model_id: str) -> Path:
    return MODELS_ROOT / sanitize_model_id(model_id)[0]


def find_existing_model_dir(candidates: list[str]) -> Path | None:
    ranked: list[tuple[int, Path]] = []
    for candidate in candidates:
        for path in candidate_model_dirs(candidate):
            if path.exists():
                ranked.append((candidate_model_score(path), path.resolve()))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def candidate_model_score(path: Path) -> int:
    score = 0
    if (path / "config.json").exists():
        score += 4
    if (path / "preprocessor_config.json").exists():
        score += 3
    if (path / "processor_config.json").exists():
        score += 2
    if (path / "model.safetensors.index.json").exists():
        score += 3
    if any(path.glob("model-*.safetensors")):
        score += 3
    if any(path.glob("pytorch_model-*.bin")):
        score += 2
    if (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists():
        score += 1
    return score


def resolve_model_path(model_path: str | None, model_id: str) -> Path:
    if model_path:
        candidate = Path(model_path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Model path does not exist: {candidate}")
        if candidate.is_dir() and not is_model_snapshot_complete(candidate):
            missing_files = missing_snapshot_files(candidate)
            broken_files = invalid_snapshot_files(candidate)
            detail = missing_files or broken_files or "unknown / incomplete shards present"
            raise FileNotFoundError(
                f"Model snapshot is incomplete: {candidate}\n"
                f"Missing or invalid files: {detail}"
            )
        return candidate.resolve()

    existing_candidates = [path for path in candidate_model_dirs(model_id) if path.exists()]
    if existing_candidates:
        existing_candidates.sort(key=candidate_model_score, reverse=True)
        candidate = existing_candidates[0]
        if not is_model_snapshot_complete(candidate):
            missing_files = missing_snapshot_files(candidate)
            broken_files = invalid_snapshot_files(candidate)
            detail = missing_files or broken_files or "unknown / incomplete shards present"
            raise FileNotFoundError(
                f"Local model snapshot exists but is incomplete: {candidate}\n"
                f"Missing or invalid files: {detail}\n"
                f"Resume download first, for example:\n"
                f"  bash {PROJECT_ROOT / 'models' / 'download_all_benchmark_models.sh'} --no-skip-complete '{model_id}'"
            )
        return candidate.resolve()

    expected_dirs = [str(path) for path in candidate_model_dirs(model_id)]
    raise FileNotFoundError(
        "No local model weights found.\n"
        f"Expected one of: {', '.join(expected_dirs)}\n"
        f"Download first with:\n"
        f"  bash {PROJECT_ROOT / 'models' / 'download_with_monitor.sh'} '{model_id}'"
    )


def resolve_videochatgpt_base_model(projection_dir: Path) -> Path:
    candidates = [
        MODELS_ROOT / "LLaVA-7B-Lightening-v1-1",
        MODELS_ROOT / "mmaaz60__LLaVA-7B-Lightening-v1-1",
        MODELS_ROOT / "liuhaotian__LLaVA-Lightning-7B-delta-v1-1",
    ]
    base_dir = None
    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            weight_files = list(candidate.glob("pytorch_model-*.bin")) + list(candidate.glob("model-*.safetensors"))
            if weight_files:
                base_dir = candidate.resolve()
                break
    if base_dir is None:
        raise FileNotFoundError(
            "Video-ChatGPT requires a full local LLaVA-Lightening-7B-v1-1 base checkpoint. "
            "The current model directory only contains projection weights."
        )
    return base_dir


def load_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt and prompt_file:
        raise ValueError("Use either --prompt or --prompt-file, not both.")
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    if prompt:
        return prompt.strip()
    return "Describe the video in detail."


RISK_STATUS_NO_ANOMALY = "normal"
RISK_STATUS_POTENTIAL = "risk_only"
RISK_STATUS_OCCURRED = "abnormal"

RISK_STATUS_ALIASES: dict[str, str] = {
    "未发生异常": RISK_STATUS_NO_ANOMALY,
    "正常": RISK_STATUS_NO_ANOMALY,
    "无异常": RISK_STATUS_NO_ANOMALY,
    "noanomaly": RISK_STATUS_NO_ANOMALY,
    "normal": RISK_STATUS_NO_ANOMALY,
    "safe": RISK_STATUS_NO_ANOMALY,
    "norisk": RISK_STATUS_NO_ANOMALY,
    "未发生但有可能发生异常": RISK_STATUS_POTENTIAL,
    "风险未发生但可能发生": RISK_STATUS_POTENTIAL,
    "潜在异常": RISK_STATUS_POTENTIAL,
    "potentialanomaly": RISK_STATUS_POTENTIAL,
    "potentialrisk": RISK_STATUS_POTENTIAL,
    "riskonly": RISK_STATUS_POTENTIAL,
    "alreadyriskbutnotoccurred": RISK_STATUS_POTENTIAL,
    "risk_only": RISK_STATUS_POTENTIAL,
    "已经发生异常": RISK_STATUS_OCCURRED,
    "已发生异常": RISK_STATUS_OCCURRED,
    "abnormal": RISK_STATUS_OCCURRED,
    "anomalyoccurred": RISK_STATUS_OCCURRED,
    "occurred": RISK_STATUS_OCCURRED,
}


def normalize_risk_status_label(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    compact = re.sub(r"[\s_\-]+", "", raw).lower()
    return RISK_STATUS_ALIASES.get(compact, raw if raw in RISK_STATUS_ALIASES.values() else "")


def normalize_time_spans_payload(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    spans: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            start = float(item[0])
            end = float(item[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        spans.append([start, end])
    return spans


def normalize_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = normalize_risk_status_label(payload.get("risk_status"))
    if not status and "has_risk" in payload:
        status = RISK_STATUS_OCCURRED if bool(payload.get("has_risk")) else RISK_STATUS_NO_ANOMALY
    if not status:
        has_video_description = bool(str(payload.get("video_description", "")).strip())
        has_risk_fields = any(bool(str(payload.get(field, "")).strip()) for field in ("risk_type", "risk_description", "solution"))
        has_time_spans = bool(normalize_time_spans_payload(payload.get("time_spans")))
        if has_time_spans:
            status = RISK_STATUS_OCCURRED
        elif has_risk_fields and not has_video_description:
            status = RISK_STATUS_POTENTIAL
        elif has_video_description:
            status = RISK_STATUS_NO_ANOMALY
    if not status:
        return payload

    normalized: dict[str, Any] = {"risk_status": status}
    if status == RISK_STATUS_NO_ANOMALY:
        normalized["video_description"] = str(payload.get("video_description", "")).strip()
        return normalized
    normalized["risk_type"] = str(payload.get("risk_type", "")).strip()
    normalized["risk_description"] = str(payload.get("risk_description", "")).strip()
    normalized["solution"] = str(payload.get("solution", "")).strip()
    if status == RISK_STATUS_OCCURRED:
        normalized["time_spans"] = normalize_time_spans_payload(payload.get("time_spans"))
    return normalized


def compress_prompt_for_legacy_backend(prompt: str) -> str:
    compact = compact_prompt = " ".join(prompt.split())
    if len(compact_prompt) <= 600:
        return prompt
    return (
        "You are a video safety annotator. Output exactly one JSON object and nothing else. "
        'Use a conditional schema based on "risk_status". '
        f'"risk_status" must be one of "{RISK_STATUS_NO_ANOMALY}", "{RISK_STATUS_POTENTIAL}", "{RISK_STATUS_OCCURRED}". '
        f'If "{RISK_STATUS_NO_ANOMALY}", output only {{"risk_status":"{RISK_STATUS_NO_ANOMALY}","video_description":"..."}}. '
        f'If "{RISK_STATUS_POTENTIAL}", output only "risk_status","risk_type","risk_description","solution". '
        f'If "{RISK_STATUS_OCCURRED}", output only "risk_status","risk_type","risk_description","time_spans","solution". '
        'risk_type: short Chinese label (e.g. 高温火源, 跌倒失稳, 碰撞砸伤, 锐器危险, 用电安全, 夹伤, 起火冒烟, 动物攻击). '
        'solution must be exactly three Chinese lines: '
        '"对人的解决方案：...\\n对危险源的解决方案：...\\n整体的解决方案：...". '
        'time_spans must be integer-second arrays like [[2,5]].'
    )


def compact_structured_prompt_for_videochatgpt(prompt: str) -> str:
    compact = (
        "Watch the full video and output exactly one JSON object only. "
        f'Classify into "{RISK_STATUS_NO_ANOMALY}", "{RISK_STATUS_POTENTIAL}", or "{RISK_STATUS_OCCURRED}" using key "risk_status". '
        f'If "{RISK_STATUS_NO_ANOMALY}", output only "risk_status" and "video_description". '
        f'If "{RISK_STATUS_POTENTIAL}" or "{RISK_STATUS_OCCURRED}", output: '
        '"risk_status", "risk_type", "risk_sources", "abnormal_actions", "affected_objects", '
        '"risk_description", "consequence_understanding", "causal_chain", "time_spans", "solution". '
        'risk_type: short Chinese label (e.g. 高温火源, 跌倒失稳, 碰撞砸伤, 锐器危险, 用电安全, 夹伤, 起火冒烟, 动物攻击). '
        'risk_sources: list of risk source strings (e.g. ["刀具","灶台明火"]). '
        'abnormal_actions: list of abnormal action strings (e.g. ["摔倒","滑倒"]). '
        'affected_objects: list of affected subject strings (e.g. ["儿童","老人"]). '
        'consequence_understanding: one sentence describing possible consequences. '
        'causal_chain: one sentence describing "source -> action -> consequence". '
        'solution: three Chinese lines separated by \\n: '
        '"对人的解决方案：...\\n对危险源的解决方案：...\\n整体的解决方案：...". '
        'time_spans: integer-second arrays like [[2,5]]. '
        'For normal, omit risk-related fields. Do not output explanations or markdown.'
    )
    return compact if prompt else compact


def output_looks_invalid(response: str) -> bool:
    stripped = response.strip()
    if not stripped:
        return True
    if len(stripped) <= 8:
        return True
    non_punct = re.sub(r"[\W_]+", "", stripped, flags=re.UNICODE)
    if len(non_punct) <= 4:
        return True
    if len(set(stripped)) <= 2 and len(stripped) >= 16:
        return True
    if re.fullmatch(r"[!！?？.。,，:：;；\-\s]+", stripped):
        return True
    lowered = stripped.lower()
    if lowered in {"and or.", "you.", "and or", "you"}:
        return True
    return False


def jsonish_field_pattern(field_name: str) -> str:
    parts = [re.escape(part) for part in field_name.split("_") if part]
    return r"\s*_?\s*".join(parts)


def decode_jsonish_string(value: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\":
            decoded.append(char)
            index += 1
            continue
        if index + 1 >= len(value):
            break
        escaped = value[index + 1]
        if escaped == "n":
            decoded.append("\n")
            index += 2
            continue
        if escaped == "r":
            decoded.append("\r")
            index += 2
            continue
        if escaped == "t":
            decoded.append("\t")
            index += 2
            continue
        if escaped == "b":
            decoded.append("\b")
            index += 2
            continue
        if escaped == "f":
            decoded.append("\f")
            index += 2
            continue
        if escaped in {'"', "'", "\\", "/", "_"}:
            decoded.append(escaped)
            index += 2
            continue
        if escaped == "u" and index + 5 < len(value):
            hex_digits = value[index + 2 : index + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", hex_digits):
                decoded.append(chr(int(hex_digits, 16)))
                index += 6
                continue
        decoded.append(escaped)
        index += 2
    return "".join(decoded)


def extract_jsonish_string_field(text: str, field_name: str) -> str:
    match = re.search(
        rf'["\']\s*{jsonish_field_pattern(field_name)}\s*["\']\s*:\s*',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    index = match.end()
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] not in {'"', "'"}:
        return ""
    quote = text[index]
    start = index + 1
    index = start
    escaped = False
    while index < len(text):
        char = text[index]
        if char == quote and not escaped:
            return decode_jsonish_string(text[start:index]).strip()
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
        index += 1
    return decode_jsonish_string(text[start:]).strip()


def parse_structured_response_text(response: str) -> dict[str, Any] | None:
    stripped = response.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    candidate = stripped.replace("\\_", "_")
    if not (candidate.startswith("{") and candidate.endswith("}")):
        match = re.search(r"\{.*", candidate, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            if candidate.count("{") > candidate.count("}"):
                candidate = candidate + ("}" * (candidate.count("{") - candidate.count("}")))
        else:
            candidate = stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        fields: dict[str, Any] = {}
        status_value = extract_jsonish_string_field(candidate, "risk_status")
        if status_value:
            fields["risk_status"] = status_value
        else:
            bool_match = re.search(
                rf'["\']\s*{jsonish_field_pattern("has_risk")}\s*["\']\s*:\s*(true|false)',
                candidate,
                flags=re.IGNORECASE,
            )
            if bool_match:
                fields["has_risk"] = bool_match.group(1).lower() == "true"
        for field_name in ("risk_type", "video_description", "risk_description", "solution"):
            value = extract_jsonish_string_field(candidate, field_name)
            if value:
                fields[field_name] = value
        span_match = re.search(
            rf'["\']\s*{jsonish_field_pattern("time_spans")}\s*["\']\s*:\s*\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            candidate,
        )
        if span_match:
            start = float(span_match.group(1))
            end = float(span_match.group(2))
            fields["time_spans"] = [[start, end]]
        if fields:
            return normalize_structured_payload(fields)
        return None
    return normalize_structured_payload(payload) if isinstance(payload, dict) else None


def choose_attn_implementation(requested: str) -> str:
    if requested == "flash_attention_2" and not module_available("flash_attn"):
        raise RuntimeError("flash_attention_2 was requested, but the flash_attn package is not installed.")
    if requested != "auto":
        return requested
    return "flash_attention_2" if module_available("flash_attn") else "sdpa"


def run_tarsier2(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    backend = BACKENDS["tarsier2"]
    ensure_backend_on_path(backend)

    tarsier_vendor = PROJECT_ROOT / ".vendor" / "tarsier_flashattn"
    if (
        os.environ.get("LIFEBENCH_TARSIER_ATTN_IMPLEMENTATION", "eager") == "flash_attention_2"
        and tarsier_vendor.exists()
    ):
        vendor_path = str(tarsier_vendor)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

    from tasks.inference_quick_start import process_one
    from tasks.utils import load_model_and_processor

    tarsier_frame_limit = int(os.environ.get("LIFEBENCH_TARSIER_MAX_FRAMES", "32"))
    frame_budget = max(1, min(args.max_frames, tarsier_frame_limit, 32))
    data_config = yaml.safe_load(Path(args.tarsier_config).read_text(encoding="utf-8"))
    data_config["n_frames"] = frame_budget
    data_config["max_n_frames"] = frame_budget
    data_config["max_pixels"] = int(os.environ.get("LIFEBENCH_TARSIER_MAX_PIXELS", "50176"))
    video_sampling_strategy = dict(data_config.get("video_sampling_strategy") or {})
    video_sampling_strategy["use_multi_images_for_video"] = False
    data_config["video_sampling_strategy"] = video_sampling_strategy
    model, processor = load_model_and_processor(str(model_path), data_config=data_config)
    generate_kwargs = {
        "do_sample": args.temperature > 0,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "use_cache": True,
    }
    return process_one(model, processor, prompt, str(args.video_path), generate_kwargs).strip()


def run_videollama2(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    backend = BACKENDS["videollama2"]
    ensure_backend_on_path(backend)

    from videollama2 import mm_infer, model_init
    from videollama2.utils import disable_torch_init

    if args.use_flash_attn and not module_available("flash_attn"):
        raise RuntimeError("VideoLLaMA2 inference requested flash_attn, but the package is not installed.")

    disable_torch_init()
    model, processor, tokenizer = model_init(
        str(model_path),
        device_map=args.device_map,
        use_flash_attn=args.use_flash_attn and module_available("flash_attn"),
    )
    video_tensor = processor["video"](str(args.video_path))
    return mm_infer(
        video_tensor,
        prompt,
        model=model,
        tokenizer=tokenizer,
        modal="video",
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    ).strip()


def run_videollama3(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    backend = BACKENDS["videollama3"]
    ensure_backend_on_path(backend)

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from videollama3 import disable_torch_init

    disable_torch_init()
    effective_max_frames = min(args.max_frames, 32)
    effective_fps = controlled_video_fps(
        str(args.video_path),
        frame_budget=effective_max_frames,
        fps_cap=1.0,
    )
    max_visual_tokens = 512

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_tokens"):
        processor.image_processor.max_tokens = max_visual_tokens
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation=choose_attn_implementation(args.attn_implementation),
    )
    conversation = [
        {"role": "system", "content": "You are a careful video safety assessor."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(Path(args.video_path).expanduser().resolve()),
                        "fps": effective_fps,
                        "max_frames": effective_max_frames,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    target_device = "cuda" if args.device_map == "auto" else args.device_map
    inputs = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    generated_ids = model.generate(
        **inputs,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[1]
    decode_ids = generated_ids[:, input_len:] if generated_ids.shape[1] > input_len else generated_ids
    response = processor.batch_decode(
        decode_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if output_looks_invalid(response):
        raise RuntimeError(f"VideoLLaMA3 produced an invalid response: {response[:120]!r}")
    return response


def normalize_model_output(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, (list, tuple)):
        if not output:
            return ""
        return normalize_model_output(output[0])
    return str(output).strip()


def strip_qwen_thinking_content(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("<think>") and "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].lstrip()
    return stripped


def choose_video_load_backend() -> str | None:
    backend_modules = (
        ("decord", "decord"),
        ("pyav", "av"),
        ("opencv", "cv2"),
        ("torchvision", "torchvision"),
    )
    for backend_name, module_name in backend_modules:
        if module_available(module_name):
            return backend_name
    return None


def get_video_duration_seconds(video_path: str) -> float | None:
    resolved_path = str(Path(video_path).expanduser().resolve())
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(resolved_path, ctx=cpu(0))
        fps = float(reader.get_avg_fps())
        if fps > 0:
            return len(reader) / fps
    except Exception:
        pass

    try:
        import cv2

        capture = cv2.VideoCapture(resolved_path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception:
        pass

    return None


def controlled_video_fps(video_path: str, frame_budget: int, fps_cap: float) -> float:
    duration_seconds = get_video_duration_seconds(video_path)
    if duration_seconds is None or duration_seconds <= 0:
        return fps_cap
    return min(fps_cap, frame_budget / duration_seconds)


def load_video_frames_simple(video_path: str, max_num_frames: int = 16) -> list[Any]:
    from PIL import Image

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        step = max(1, round(vr.get_avg_fps()))
        frame_idx = list(range(0, len(vr), step))
        if len(frame_idx) > max_num_frames:
            gap = len(frame_idx) / max_num_frames
            frame_idx = [frame_idx[int(i * gap + gap / 2)] for i in range(max_num_frames)]
        frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame.astype("uint8")) for frame in frames]
    except Exception:
        pass

    try:
        import cv2

        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frames: list[Any] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            capture.release()
            if not frames:
                return []
            if len(frames) > max_num_frames:
                gap = len(frames) / max_num_frames
                frames = [frames[int(i * gap + gap / 2)] for i in range(max_num_frames)]
            return [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

        target_indices = [0]
        if frame_count > 1:
            target_indices = [round(i * (frame_count - 1) / max(max_num_frames - 1, 1)) for i in range(min(max_num_frames, frame_count))]
        target_index_set = set(target_indices)
        selected: list[Any] = []
        current_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if current_index in target_index_set:
                selected.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if len(selected) >= len(target_indices):
                    break
            current_index += 1
        capture.release()
        return selected
    except Exception:
        return []


def nearest_scale_ids(values: Any, scale: Any) -> Any:
    import numpy as np

    if len(scale) == 0:
        return np.zeros_like(values, dtype=np.int32)
    indices = np.searchsorted(scale, values, side="left")
    indices = np.clip(indices, 0, len(scale) - 1)
    previous = np.clip(indices - 1, 0, len(scale) - 1)
    choose_previous = np.abs(values - scale[previous]) <= np.abs(values - scale[indices])
    return np.where(choose_previous, previous, indices).astype(np.int32)


def group_sequence(values: Any, size: int) -> list[list[int]]:
    return [list(map(int, values[index:index + size])) for index in range(0, len(values), size)]


def load_minicpmv45_video(
    video_path: str,
    choose_fps: float,
    max_num_frames: int,
    max_num_packing: int = 3,
    time_scale: float = 0.1,
    force_packing: int | None = None,
) -> tuple[list[Any], list[list[int]]]:
    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if len(reader) <= 0:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")
    source_fps = float(reader.get_avg_fps())
    if source_fps <= 0:
        source_fps = max(choose_fps, 1.0)
    duration_seconds = max(len(reader) / source_fps, 1.0 / source_fps)
    max_num_frames = max(1, max_num_frames)
    max_num_packing = max(1, min(max_num_packing, 6))
    choose_fps = max(float(choose_fps), 0.1)

    target_frames = max(1, round(duration_seconds * choose_fps))
    target_frames = min(target_frames, len(reader), max_num_frames * max_num_packing)
    if force_packing is not None:
        packing_nums = max(1, min(int(force_packing), max_num_packing))
    else:
        packing_nums = max(1, min(math.ceil(target_frames / max_num_frames), max_num_packing))

    sample_positions = np.linspace(0, len(reader) - 1, num=target_frames)
    frame_indices = np.rint(sample_positions).astype(np.int64)
    frames_array = reader.get_batch(frame_indices).asnumpy()

    frame_timestamps = frame_indices / source_fps
    scale = np.arange(0, duration_seconds + time_scale, time_scale)
    frame_ts_ids = nearest_scale_ids(frame_timestamps, scale)
    frames = [Image.fromarray(frame.astype("uint8")).convert("RGB") for frame in frames_array]
    temporal_ids = group_sequence(frame_ts_ids, packing_nums)
    return frames, temporal_ids


def run_commercial(model_path: Path | None, args: argparse.Namespace, prompt: str) -> str:
    del model_path
    from llm import chat_completion, image_to_data_url

    frame_budget = max(1, min(args.max_frames, 12))
    video_frames = load_video_frames_simple(str(args.video_path), max_num_frames=frame_budget)
    if not video_frames:
        raise RuntimeError(f"Unable to decode video frames for commercial backend: {args.video_path}")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "These sampled frames are ordered chronologically from the same video. "
                "Use them to infer the full scene dynamics and answer the task below."
            ),
        }
    ]
    for index, frame in enumerate(video_frames, start=1):
        content.append({"type": "text", "text": f"Frame {index}/{len(video_frames)}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(frame, format="JPEG", quality=85),
                    "detail": "low",
                },
            }
        )
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a careful video safety assessor."},
        {"role": "user", "content": content},
    ]
    return chat_completion(
        messages,
        model=args.model_id or BACKENDS["commercial"].default_model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    ).strip()


def run_videoccam(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    backend = BACKENDS["videoccam"]
    ensure_backend_on_path(backend)

    from eval.utils import load_decord
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    torch_dtype = "bfloat16"
    attn_implementation = choose_attn_implementation(args.attn_implementation)
    if args.device_map == "auto":
        device_map: str | dict[str, str] = {"": "cuda:0"}
    else:
        device_map = args.device_map
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    image_processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=getattr(__import__("torch"), torch_dtype),
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    messages = [[{"role": "user", "content": f"<video>\n{prompt}"}]]
    images = [load_decord(str(args.video_path), sample_type="uniform", num_frames=min(args.max_frames, 32))]
    response = model.chat(
        messages,
        images,
        tokenizer,
        image_processor,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
    )
    return normalize_model_output(response)


def build_internvl_transform(input_size: int):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: set[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def internvl_dynamic_preprocess(
    image: Any,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = set(sorted(target_ratios, key=lambda item: item[0] * item[1]))
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_image = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_image.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def internvl_frame_indices(total_frames: int, num_segments: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video contains no decodable frames.")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive.")
    segment_size = float(total_frames - 1) / num_segments
    return [
        int(segment_size / 2 + round(segment_size * index))
        for index in range(num_segments)
    ]


def load_internvl_video(
    video_path: str,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 32,
):
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    transform = build_internvl_transform(input_size=input_size)
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_indices = internvl_frame_indices(len(video_reader), num_segments)
    pixel_values = []
    num_patches_list = []
    for frame_index in frame_indices:
        frame = Image.fromarray(video_reader[frame_index].asnumpy()).convert("RGB")
        images = internvl_dynamic_preprocess(
            frame,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num,
        )
        pixel_values.extend(transform(image) for image in images)
        num_patches_list.append(len(images))
    return torch.stack(pixel_values), num_patches_list


def load_internvl_model(model_path: Path, args: argparse.Namespace):
    vendor_dir = PROJECT_ROOT / ".vendor" / "qwen35_shim"
    if vendor_dir.exists():
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    target_device = "cuda" if args.device_map == "auto" else args.device_map
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_flash_attn": args.use_flash_attn,
    }
    model_parallel = os.environ.get("LIFEBENCH_INTERNVL35_MODEL_PARALLEL") == "1"
    if model_parallel:
        load_kwargs["device_map"] = "auto"
        load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModel.from_pretrained(str(model_path), **load_kwargs).eval()
    # InternVL falls back to eager Qwen attention when flash-attn is absent.
    # PyTorch SDPA keeps the 32-frame path memory-efficient without a CUDA toolkit.
    if not args.use_flash_attn:
        language_model = getattr(model, "language_model", None)
        if language_model is not None and hasattr(language_model, "config"):
            language_model.config._attn_implementation = "sdpa"
        if hasattr(model.config, "llm_config"):
            model.config.llm_config._attn_implementation = "sdpa"
    if not model_parallel:
        model = model.to(target_device)
    else:
        target_device = next(model.parameters()).device
    return model, tokenizer, target_device


def run_internvl35(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    model, tokenizer, target_device = load_internvl_model(model_path, args)
    frame_count = max(1, min(args.max_frames, 8))
    pixel_values, num_patches_list = load_internvl_video(
        str(args.video_path),
        num_segments=frame_count,
        max_num=1,
    )
    pixel_values = pixel_values.to(torch.bfloat16).to(target_device)
    frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_config["temperature"] = args.temperature
        generation_config["top_p"] = args.top_p
    response = model.chat(
        tokenizer,
        pixel_values,
        frame_prefix + prompt,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    if isinstance(response, tuple):
        response = response[0]
    return normalize_model_output(response)


def run_mplugowl3(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    backend = BACKENDS["mplugowl3"]
    ensure_backend_on_path(backend)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    model.eval()
    processor = model.init_processor(tokenizer)
    video_frames = [load_video_frames_simple(str(args.video_path), max_num_frames=min(args.max_frames, 32))]
    messages = [
        {"role": "user", "content": f"<|video|>\n{prompt}"},
        {"role": "assistant", "content": ""},
    ]
    inputs = processor(messages, images=None, videos=video_frames)
    if hasattr(inputs, "to"):
        target_device = "cuda" if args.device_map == "auto" else args.device_map
        inputs = inputs.to(target_device)
    inputs.update(
        {
            "tokenizer": tokenizer,
            "max_new_tokens": args.max_new_tokens,
            "decode_text": True,
        }
    )
    output = model.generate(**inputs)
    return normalize_model_output(output)


def run_subprocess_adapter(script_path: Path, adapter_args: list[str]) -> str:
    completed = subprocess.run(
        [sys.executable, str(script_path), *adapter_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"Adapter exited with code {completed.returncode}"
        raise RuntimeError(detail)
    return completed.stdout.strip()


def run_videollava(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    adapter = PROJECT_ROOT / "compat" / "videollava_once.py"
    effective_max_new_tokens = max(args.max_new_tokens, 256)
    return run_subprocess_adapter(
        adapter,
        [
            "--model-path",
            str(model_path),
            "--video-path",
            str(args.video_path),
            "--prompt",
            prompt,
            "--temperature",
            str(args.temperature),
            "--max-new-tokens",
            str(effective_max_new_tokens),
        ],
    )


def run_videochat2(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    adapter = PROJECT_ROOT / "compat" / "videochat2_once.py"
    effective_max_new_tokens = max(args.max_new_tokens, 256)
    return run_subprocess_adapter(
        adapter,
        [
            "--model-path",
            str(model_path),
            "--video-path",
            str(args.video_path),
            "--prompt",
            prompt,
            "--temperature",
            str(args.temperature),
            "--max-new-tokens",
            str(effective_max_new_tokens),
        ],
    )


def run_videochatgpt(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    backend = BACKENDS["videochatgpt"]
    ensure_backend_on_path(backend)

    from video_chatgpt.eval.model_utils import initialize_model, load_video
    from video_chatgpt.inference import video_chatgpt_infer

    base_model_path = model_path
    projection_path = args.projection_path
    if not (model_path / "config.json").exists() and (model_path / "video_chatgpt-7B.bin").exists():
        base_model_path = resolve_videochatgpt_base_model(model_path)
        if projection_path is None:
            projection_path = str(model_path / "video_chatgpt-7B.bin")

    compact_prompt = compact_structured_prompt_for_videochatgpt(prompt)
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
        str(base_model_path),
        projection_path,
    )
    video_frames = load_video(str(args.video_path), num_frm=min(args.max_frames, 32))
    response = video_chatgpt_infer(
        video_frames,
        compact_prompt,
        "video-chatgpt_v1",
        model,
        vision_tower,
        tokenizer,
        image_processor,
        video_token_len,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 0.0),
        max_new_tokens=min(args.max_new_tokens, 320),
    ).strip()
    if output_looks_invalid(response):
        raise RuntimeError(f"Video-ChatGPT produced an invalid response: {response[:120]!r}")
    return response


def run_minigpt4video(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    adapter = PROJECT_ROOT / "compat" / "minigpt4_video_once.py"
    effective_max_new_tokens = max(args.max_new_tokens, 512)
    return run_subprocess_adapter(
        adapter,
        [
            "--video-path",
            str(args.video_path),
            "--prompt",
            prompt,
            "--max-frames",
            str(min(args.max_frames, 32)),
            "--max-new-tokens",
            str(effective_max_new_tokens),
            "--temperature",
            str(args.temperature),
        ],
    )


def run_qwen25vl(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    if not module_available("qwen_vl_utils"):
        raise RuntimeError("Qwen2.5-VL inference requires qwen_vl_utils, but it is not installed.")

    vendor_dir = PROJECT_ROOT / ".vendor" / "qwen251_shim"
    if vendor_dir.exists():
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

    from qwen_vl_utils import process_vision_info
    original_find_spec = importlib.util.find_spec
    original_metadata = importlib_metadata.metadata

    def patched_find_spec(name: str, *args: Any, **kwargs: Any):
        if name == "deepspeed":
            return None
        return original_find_spec(name, *args, **kwargs)

    def patched_metadata(name: str, *args: Any, **kwargs: Any):
        if name == "deepspeed":
            raise importlib_metadata.PackageNotFoundError(name)
        return original_metadata(name, *args, **kwargs)

    importlib.util.find_spec = patched_find_spec
    importlib_metadata.metadata = patched_metadata
    try:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL requires transformers>=4.51.3. "
                "Install it into lifebench/.vendor/qwen251_shim or upgrade the runtime environment."
            ) from exc
    finally:
        importlib.util.find_spec = original_find_spec
        importlib_metadata.metadata = original_metadata

    load_path = model_path
    processor = AutoProcessor.from_pretrained(str(load_path), trust_remote_code=True)
    model_cls = Qwen2_5_VLForConditionalGeneration

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
    }
    target_device = "cuda" if args.device_map == "auto" else args.device_map
    model = model_cls.from_pretrained(
        str(load_path),
        **load_kwargs,
    )
    model = model.to(target_device)
    frame_budget = min(args.max_frames, 32)
    effective_fps = controlled_video_fps(
        str(args.video_path),
        frame_budget=frame_budget,
        fps_cap=1.0,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{Path(args.video_path).expanduser().resolve()}",
                    "fps": effective_fps,
                    "max_pixels": 360 * 420,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(target_device)
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return normalize_model_output(response)


def run_qwen35vl(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    vendor_dir = PROJECT_ROOT / ".vendor" / "qwen35_shim"
    if vendor_dir.exists():
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

    original_find_spec = importlib.util.find_spec
    original_metadata = importlib_metadata.metadata

    def patched_find_spec(name: str, *args: Any, **kwargs: Any):
        if name == "deepspeed":
            return None
        return original_find_spec(name, *args, **kwargs)

    def patched_metadata(name: str, *args: Any, **kwargs: Any):
        if name == "deepspeed":
            raise importlib_metadata.PackageNotFoundError(name)
        return original_metadata(name, *args, **kwargs)

    importlib.util.find_spec = patched_find_spec
    importlib_metadata.metadata = patched_metadata
    try:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 local inference requires a recent Transformers build with Qwen3.5 support. "
                "Install transformers 5.4.0+ into lifebench/.vendor/qwen35_shim or upgrade the runtime environment."
            ) from exc
    finally:
        importlib.util.find_spec = original_find_spec
        importlib_metadata.metadata = original_metadata

    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
    }
    target_device = "cuda" if args.device_map == "auto" else args.device_map

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "fps"):
        processor.video_processor.fps = None
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        **load_kwargs,
    )
    model = model.to(target_device)

    video_path = str(Path(args.video_path).expanduser().resolve())
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "path": video_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    processor_kwargs: dict[str, Any] = {
        "num_frames": max(1, min(args.max_frames, 32)),
    }
    inputs = processor.apply_chat_template(messages, processor_kwargs=processor_kwargs, **template_kwargs)
    inputs = inputs.to(target_device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
    generated_ids = model.generate(**inputs, **generate_kwargs)
    input_len = inputs["input_ids"].shape[1]
    decode_ids = generated_ids[:, input_len:] if generated_ids.shape[1] > input_len else generated_ids
    response = processor.batch_decode(
        decode_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return strip_qwen_thinking_content(normalize_model_output(response))


def load_minicpmv45_model(model_path: Path, args: argparse.Namespace):
    from transformers import AutoModel, AutoTokenizer

    target_device = "cuda" if args.device_map == "auto" else args.device_map
    attn_implementation = (
        "flash_attention_2"
        if args.attn_implementation == "flash_attention_2" and module_available("flash_attn")
        else "sdpa"
    )
    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().to(target_device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    return model, tokenizer, target_device


def minicpmv45_chat(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    prompt: str,
) -> str:
    frames, temporal_ids = load_minicpmv45_video(
        str(args.video_path),
        choose_fps=args.fps,
        max_num_frames=args.max_frames,
    )
    messages = [{"role": "user", "content": frames + [prompt]}]
    chat_kwargs: dict[str, Any] = {
        "msgs": messages,
        "tokenizer": tokenizer,
        "use_image_id": False,
        "max_slice_nums": 1,
        "temporal_ids": temporal_ids,
        "enable_thinking": False,
        "stream": False,
        "max_new_tokens": args.max_new_tokens,
        "sampling": args.temperature > 0,
    }
    if args.temperature > 0:
        chat_kwargs["temperature"] = args.temperature
        chat_kwargs["top_p"] = args.top_p
    try:
        response = model.chat(**chat_kwargs)
    except TypeError:
        chat_kwargs.pop("max_new_tokens", None)
        response = model.chat(**chat_kwargs)
    return normalize_model_output(response)


def run_minicpmv45(model_path: Path, args: argparse.Namespace, prompt: str) -> str:
    model, tokenizer, _target_device = load_minicpmv45_model(model_path, args)
    return minicpmv45_chat(model, tokenizer, args, prompt)


RUNNERS = {
    "tarsier2": run_tarsier2,
    "videollama2": run_videollama2,
    "videollama3": run_videollama3,
    "videoccam": run_videoccam,
    "internvl35": run_internvl35,
    "mplugowl3": run_mplugowl3,
    "videollava": run_videollava,
    "videochat2": run_videochat2,
    "videochatgpt": run_videochatgpt,
    "minigpt4video": run_minigpt4video,
    "qwen25vl": run_qwen25vl,
    "qwen35vl": run_qwen35vl,
    "minicpmv45": run_minicpmv45,
    "commercial": run_commercial,
}


def list_backends() -> None:
    payload = {
        name: {
            "default_model_id": spec.default_model_id,
            "default_download_dir": (
                str(find_existing_model_dir([spec.default_model_id]) or default_download_dir(spec.default_model_id))
                if spec.requires_local_model
                else None
            ),
            "code_dir": str(spec.code_dir),
            "description": spec.description,
            "requires_local_model": spec.requires_local_model,
        }
        for name, spec in sorted(BACKENDS.items())
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def list_benchmarks() -> None:
    payload = [
        {
            "name": model.name,
            "model_id": model.model_id,
            "backend": model.backend,
            "download_dir": str(find_existing_model_dir([model.model_id]) or default_download_dir(model.model_id)),
            "notes": model.notes,
        }
        for model in BENCHMARK_MODELS
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def validate_args(args: argparse.Namespace) -> None:
    if args.list_backends:
        return
    if not args.backend:
        raise ValueError("--backend is required unless --list-backends is used.")
    if not args.video_path:
        raise ValueError("--video-path is required for inference.")

    video_path = Path(args.video_path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")


def build_result(
    args: argparse.Namespace,
    backend: BackendSpec,
    model_id: str,
    model_path: Path | None,
    prompt: str,
    response: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    try:
        parsed_response = parse_structured_response_text(response)
    except Exception:
        parsed_response = None
    if parsed_response is None:
        try:
            from evaluation.common import structured_prediction as derive_structured_prediction

            parsed_response = derive_structured_prediction(
                {
                    "model_id": model_id,
                    "backend": backend.name,
                    "video_path": str(Path(args.video_path).expanduser().resolve()),
                    "response": response,
                }
            )["summary"]
        except Exception:
            parsed_response = None

    return {
        "backend": backend.name,
        "description": backend.description,
        "model_id": model_id,
        "model_path": str(model_path) if model_path is not None else None,
        "video_path": str(Path(args.video_path).expanduser().resolve()),
        "prompt": prompt,
        "response": response,
        "parsed_response": parsed_response,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "merge_size": args.merge_size,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_backends:
        list_backends()
        return 0
    if args.list_benchmarks:
        list_benchmarks()
        return 0

    validate_args(args)
    prompt = load_prompt(args.prompt, args.prompt_file)
    backend = BACKENDS[args.backend]
    model_id = args.model_id or backend.default_model_id
    model_path = resolve_model_path(args.model_path, model_id) if backend.requires_local_model else None

    started_at = time.time()
    response = RUNNERS[backend.name](model_path, args, prompt)
    elapsed_seconds = time.time() - started_at

    result = build_result(args, backend, model_id, model_path, prompt, response, elapsed_seconds)
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(response)
    return 0


def run_cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())

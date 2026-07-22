#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from benchmark_models import BENCHMARK_MODELS  # noqa: E402
from evaluation.common import (  # noqa: E402
    GroundTruthEntry,
    RISK_STATUS_NO_ANOMALY,
    RISK_STATUS_OCCURRED,
    build_judge,
    format_json,
    judge_or_heuristic,
    load_ground_truth,
    normalize_risk_status_label,
    normalize_risk_type_label,
    risk_type_heuristic,
    risk_description_heuristic,
    safe_stem_from_video_path,
    match_video_id_from_path,
    solution_heuristic,
    structured_prediction,
    summary_completeness,
    time_iou,
    video_description_heuristic,
    parse_prediction_solution_sections,
    weighted_overall_score,
)


DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "gen_prompt.txt"
DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "outputs" / "benchmark_inference"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "latest"
EVALUATION_CACHE_VERSION = 1
PER_PREDICTION_CACHE_DIRNAME = "per_prediction_results"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate benchmark model outputs against structured ground truth.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Path to gen_prompt.txt.")
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Root directory that contains benchmark inference outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation artifacts.")
    parser.add_argument("--model-id", help="Optional filter for a single model id.")
    parser.add_argument(
        "--judge-mode",
        choices=("auto", "heuristic", "openai"),
        default="auto",
        help="Use heuristic scoring only, or an OpenAI-compatible judge when configured.",
    )
    parser.add_argument("--judge-model", help="LLM judge model name for an OpenAI-compatible endpoint.")
    parser.add_argument("--judge-api-key", help="LLM judge API key. Falls back to env vars when omitted.")
    parser.add_argument(
        "--judge-base-url",
        help="Single OpenAI-compatible base URL or /chat/completions endpoint.",
    )
    parser.add_argument(
        "--judge-base-urls",
        nargs="*",
        help=(
            "Multiple OpenAI-compatible base URLs. Requests are dispatched round-robin "
            "across endpoints, with failover on errors. Use this to scale across multiple "
            "Ollama instances (e.g. one per GPU) bound to different ports, or a single "
            "multi-GPU Ollama. Takes precedence over --judge-base-url when set."
        ),
    )
    parser.add_argument(
        "--require-complete-videos",
        action="store_true",
        help="Only keep videos where every expected model has a valid prediction.",
    )
    parser.add_argument(
        "--expected-models",
        nargs="*",
        help="Explicit model ids required for complete-video filtering. Defaults to all 10 benchmark models.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers used for per-prediction evaluation. Default: 8.",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="Include ad-hoc prediction JSONs outside the standard model output directories when possible.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible per-prediction evaluation caches from <output-dir>/per_prediction_results.",
    )
    return parser


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def model_cache_dirname(model_id: str) -> str:
    normalized = model_id.replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in normalized)


def semantic_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not key.startswith("__")
    }


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    write_text_atomic(path, format_json(payload))


def build_judge_signature(
    requested_mode: str,
    effective_judge_mode: str,
    base_judge: Any,
) -> dict[str, Any]:
    return {
        "cache_version": EVALUATION_CACHE_VERSION,
        "requested_judge_mode": requested_mode,
        "effective_judge_mode": effective_judge_mode,
        "judge_model": getattr(base_judge, "model", None),
        "judge_base_url": getattr(base_judge, "base_url", None),
    }


def cache_path_for_prediction(output_dir: Path, model_id: str, video_id: str) -> Path:
    return output_dir / PER_PREDICTION_CACHE_DIRNAME / model_cache_dirname(model_id) / f"{video_id}.json"


def resolve_ground_truth_video_id(video_path: str | None, gt_by_video: dict[str, GroundTruthEntry]) -> str | None:
    video_id = match_video_id_from_path(video_path)
    if video_id in gt_by_video:
        return video_id
    if not video_id:
        return None
    suffix = f"_{video_id}"
    matches = [candidate for candidate in gt_by_video if candidate.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def load_cached_evaluation(
    path: Path,
    *,
    model_id: str,
    video_id: str,
    prediction_hash: str,
    ground_truth_hash: str,
    judge_signature: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_version") != EVALUATION_CACHE_VERSION:
        return None
    if payload.get("model_id") != model_id or payload.get("video_id") != video_id:
        return None
    if payload.get("prediction_hash") != prediction_hash:
        return None
    if payload.get("ground_truth_hash") != ground_truth_hash:
        return None
    if payload.get("judge_signature") != judge_signature:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    if result.get("model_id") != model_id or result.get("video_id") != video_id:
        return None
    return result


def write_cached_evaluation(
    path: Path,
    *,
    model_id: str,
    video_id: str,
    prediction_hash: str,
    ground_truth_hash: str,
    judge_signature: dict[str, Any],
    result: dict[str, Any],
) -> None:
    write_json_atomic(
        path,
        {
            "cache_version": EVALUATION_CACHE_VERSION,
            "updated_at": now_iso(),
            "model_id": model_id,
            "video_id": video_id,
            "prediction_hash": prediction_hash,
            "ground_truth_hash": ground_truth_hash,
            "judge_signature": judge_signature,
            "result": result,
        },
    )


def discover_prediction_files(root: Path, include_manual: bool) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if any(part in {"status", "logs", ".auto_infer_state"} for part in path.parts):
            continue
        if path.name in {"summary.json", "evaluation_status.json", "state.json"}:
            continue
        if path.name.startswith("manual_"):
            if include_manual:
                candidates.append(path)
            continue
        if path.parent.name.endswith("-7b") or path.parent.name.endswith("_7b") or path.parent.parent.name == "benchmark_inference":
            candidates.append(path)
            continue
        candidates.append(path)
    return sorted(dict.fromkeys(candidates))


def load_prediction_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if "response" not in payload or "model_id" not in payload or "video_path" not in payload:
        return None
    return payload


def schema_absence_eval(field_name: str, structured_pred: dict[str, Any]) -> dict[str, Any]:
    raw_response = str(structured_pred.get("raw_response", ""))
    present = re.search(rf'["\']{re.escape(field_name)}["\']\s*:', raw_response) is not None
    return {
        "score": 0.0 if present else 5.0,
        "judge_mode": "heuristic",
        "reason": f'"{field_name}" should be omitted for this risk_status',
        "meta": {"prediction_present": present, "field_name": field_name},
    }


def evaluate_prediction(
    gt: GroundTruthEntry,
    raw_payload: dict[str, Any],
    judge,
) -> dict[str, Any]:
    structured_pred = structured_prediction(raw_payload)
    summary = structured_pred["summary"]
    completeness = summary_completeness(summary)
    gt_status = normalize_risk_status_label(gt.summary_gt.get("risk_status", ""))
    pred_status = normalize_risk_status_label(summary.get("risk_status", ""))
    gt_risk_type = normalize_risk_type_label(gt.summary_gt.get("risk_type", ""))
    pred_risk_type = normalize_risk_type_label(summary.get("risk_type", ""))
    predicted_time_field_present = re.search(r'["\']time_spans["\']\s*:', structured_pred["raw_response"]) is not None

    risk_status_acc = 1.0 if pred_status == gt_status else 0.0
    iou = round(
        time_iou(gt.summary_gt["time_spans"], summary["time_spans"])
        if gt_status != RISK_STATUS_NO_ANOMALY
        else (0.0 if predicted_time_field_present else 1.0),
        4,
    )

    video_score, video_meta = video_description_heuristic(gt, structured_pred)
    risk_score, risk_meta = risk_description_heuristic(gt, structured_pred)
    solution_score, solution_meta = solution_heuristic(gt, structured_pred)

    if gt_status == RISK_STATUS_NO_ANOMALY:
        predicted_risk_type_present = re.search(r'["\']risk_type["\']\s*:', structured_pred["raw_response"]) is not None
        risk_type_accuracy = 1.0 if not predicted_risk_type_present else 0.0
        risk_type_eval = {
            "score": 5.0 if risk_type_accuracy else 0.0,
            "judge_mode": "heuristic",
            "reason": "risk_type should be omitted when there is no anomaly",
            "meta": {
                "ground_truth": "",
                "prediction": pred_risk_type,
                "prediction_present": predicted_risk_type_present,
                "accuracy": risk_type_accuracy,
            },
        }
    else:
        risk_type_accuracy = 1.0 if pred_risk_type == gt_risk_type and gt_risk_type else 0.0
        risk_type_heuristic_score, risk_type_meta = risk_type_heuristic(gt, structured_pred)
        risk_type_eval = {
            "score": 5.0 if risk_type_accuracy else 0.0,
            "judge_mode": "heuristic",
            "reason": "exact match on normalized risk_type",
            "meta": {
                "ground_truth": gt_risk_type,
                "prediction": pred_risk_type,
                "accuracy": risk_type_accuracy,
                "heuristic_reference_score": risk_type_heuristic_score,
                "heuristic_meta": risk_type_meta,
            },
        }
    if gt_status == RISK_STATUS_NO_ANOMALY:
        video_eval = judge_or_heuristic(
            "video_description",
            {
                "video_description": gt.summary_gt["video_description"],
                "timeline_segments": gt.safe_segments + gt.risk_segments,
            },
            {
                "video_description": summary.get("video_description"),
                "raw_response": structured_pred["raw_response"],
            },
            video_score,
            video_meta,
            judge,
        )
        risk_eval = schema_absence_eval("risk_description", structured_pred)
    else:
        video_eval = schema_absence_eval("video_description", structured_pred)
        risk_eval = judge_or_heuristic(
            "risk_description",
            {
                "risk_segments": gt.risk_segments,
                "risk_summary": gt.summary_gt["risk_description"],
            },
            {
                "risk_description": summary.get("risk_description"),
                "raw_response": structured_pred["raw_response"],
            },
            risk_score,
            risk_meta,
            judge,
        )
    pred_solution_sections = parse_prediction_solution_sections(summary.get("solution") or "")
    gt_solution_sections = {
        "person_solution": gt.solution_sections.get("person_solution", ""),
        "hazard_solution": gt.solution_sections.get("hazard_solution", ""),
        "overall_solution": gt.solution_sections.get("prevention_solution", ""),
    }
    solution_section_scores = solution_meta.get("section_scores", {})
    solution_section_meta = solution_meta.get("section_meta", {})
    if gt_status == RISK_STATUS_NO_ANOMALY:
        absence_eval = schema_absence_eval("solution", structured_pred)
        solution_person_eval = absence_eval
        solution_hazard_eval = absence_eval
        solution_overall_eval = absence_eval
    else:
        solution_person_eval = judge_or_heuristic(
            "solution_person",
            {
                "risk_type": gt.summary_gt["risk_type"],
                "risk_description": gt.summary_gt["risk_description"],
                "solution_section": gt_solution_sections["person_solution"],
            },
            {
                "solution_section": pred_solution_sections.get("person_solution", ""),
                "full_solution": summary.get("solution"),
                "raw_response": structured_pred["raw_response"],
            },
            float(solution_section_scores.get("person_solution", 0.0)),
            solution_section_meta.get("person_solution", {}),
            judge,
        )
        solution_hazard_eval = judge_or_heuristic(
            "solution_hazard",
            {
                "risk_type": gt.summary_gt["risk_type"],
                "risk_description": gt.summary_gt["risk_description"],
                "solution_section": gt_solution_sections["hazard_solution"],
            },
            {
                "solution_section": pred_solution_sections.get("hazard_solution", ""),
                "full_solution": summary.get("solution"),
                "raw_response": structured_pred["raw_response"],
            },
            float(solution_section_scores.get("hazard_solution", 0.0)),
            solution_section_meta.get("hazard_solution", {}),
            judge,
        )
        solution_overall_eval = judge_or_heuristic(
            "solution_overall",
            {
                "risk_type": gt.summary_gt["risk_type"],
                "risk_description": gt.summary_gt["risk_description"],
                "solution_section": gt_solution_sections["overall_solution"],
            },
            {
                "solution_section": pred_solution_sections.get("overall_solution", ""),
                "full_solution": summary.get("solution"),
                "raw_response": structured_pred["raw_response"],
            },
            float(solution_section_scores.get("overall_solution", 0.0)),
            solution_section_meta.get("overall_solution", {}),
            judge,
        )
    solution_eval = {
        "score": round(
            (
                float(solution_person_eval["score"])
                + float(solution_hazard_eval["score"])
                + float(solution_overall_eval["score"])
            )
            / 3.0,
            3,
        ),
        "judge_mode": "mixed" if len({solution_person_eval["judge_mode"], solution_hazard_eval["judge_mode"], solution_overall_eval["judge_mode"]}) > 1 else solution_person_eval["judge_mode"],
        "reason": "average of person/hazard/overall solution scores",
        "sections": {
            "person_solution": solution_person_eval,
            "hazard_solution": solution_hazard_eval,
            "overall_solution": solution_overall_eval,
        },
    }

    overall = weighted_overall_score(
        risk_status=risk_status_acc,
        risk_type_score=float(risk_type_eval["score"]),
        iou_score=iou,
        video_desc_score=float(video_eval["score"]),
        risk_desc_score=float(risk_eval["score"]),
        solution_score=float(solution_eval["score"]),
    )

    return {
        "video_id": gt.video_id,
        "model_id": raw_payload["model_id"],
        "backend": raw_payload.get("backend"),
        "prediction_path": str(raw_payload.get("__source_path__", "")),
        "structured_prediction": structured_pred,
        "ground_truth": gt.as_dict(),
        "metrics": {
            "risk_status_accuracy": risk_status_acc,
            "risk_type_accuracy": risk_type_accuracy,
            "risk_type_score": float(risk_type_eval["score"]),
            "time_iou": iou,
            "video_description_score": float(video_eval["score"]),
            "risk_description_score": float(risk_eval["score"]),
            "solution_score": float(solution_eval["score"]),
            "person_solution_score": float(solution_person_eval["score"]),
            "hazard_solution_score": float(solution_hazard_eval["score"]),
            "overall_solution_score": float(solution_overall_eval["score"]),
            "summary_completeness": completeness,
            "overall_score": overall,
        },
        "judgement": {
            "risk_status": {
                "score": risk_status_acc,
                "judge_mode": "heuristic",
                "reason": "exact match on risk_status",
                "meta": {"ground_truth": gt_status, "prediction": pred_status},
            },
            "risk_type": risk_type_eval,
            "video_description": video_eval,
            "risk_description": risk_eval,
            "solution": solution_eval,
        },
    }


def summarize_by_model(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in per_video:
        grouped[record["model_id"]].append(record)

    summary: dict[str, Any] = {}
    for model_id, records in grouped.items():
        count = len(records)
        metric_names = (
            "risk_status_accuracy",
            "risk_type_accuracy",
            "risk_type_score",
            "time_iou",
            "video_description_score",
            "risk_description_score",
            "solution_score",
            "person_solution_score",
            "hazard_solution_score",
            "overall_solution_score",
            "overall_score",
        )
        averages = {
            name: round(sum(record["metrics"][name] for record in records) / count, 4)
            for name in metric_names
        }
        completeness = round(
            sum(record["metrics"]["summary_completeness"]["ratio"] for record in records) / count,
            4,
        )
        summary[model_id] = {
            "sample_count": count,
            "averages": averages,
            "summary_completeness_ratio": completeness,
            "videos": [record["video_id"] for record in records],
        }
    return summary


def write_summary_csv(path: Path, per_video: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "video_id",
                "backend",
                "risk_status_accuracy",
                "risk_type_accuracy",
                "risk_type_score",
                "time_iou",
                "video_description_score",
                "risk_description_score",
                "solution_score",
                "person_solution_score",
                "hazard_solution_score",
                "overall_solution_score",
                "overall_score",
            ],
        )
        writer.writeheader()
        for record in per_video:
            writer.writerow(
                {
                    "model_id": record["model_id"],
                    "video_id": record["video_id"],
                    "backend": record.get("backend"),
                    "risk_status_accuracy": record["metrics"]["risk_status_accuracy"],
                    "risk_type_accuracy": record["metrics"]["risk_type_accuracy"],
                    "risk_type_score": record["metrics"]["risk_type_score"],
                    "time_iou": record["metrics"]["time_iou"],
                    "video_description_score": record["metrics"]["video_description_score"],
                    "risk_description_score": record["metrics"]["risk_description_score"],
                    "solution_score": record["metrics"]["solution_score"],
                    "person_solution_score": record["metrics"]["person_solution_score"],
                    "hazard_solution_score": record["metrics"]["hazard_solution_score"],
                    "overall_solution_score": record["metrics"]["overall_solution_score"],
                    "overall_score": record["metrics"]["overall_score"],
                }
            )


def resolve_expected_model_ids(model_id_filter: str | None, expected_models: list[str] | None) -> list[str]:
    if model_id_filter:
        return [model_id_filter]
    if expected_models:
        return list(dict.fromkeys(expected_models))
    return [model.model_id for model in BENCHMARK_MODELS]


def filter_complete_video_payloads(
    payloads: list[dict[str, Any]],
    expected_model_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = set(expected_model_ids)
    if not expected:
        return payloads, {
            "required": False,
            "expected_model_ids": [],
            "raw_video_count": 0,
            "complete_video_count": 0,
            "excluded_incomplete_video_count": 0,
            "raw_prediction_file_count": len(payloads),
            "filtered_prediction_file_count": len(payloads),
            "complete_video_ids": [],
        }

    video_to_models: dict[str, set[str]] = defaultdict(set)
    for payload in payloads:
        video_id = payload.get("__video_id__") or match_video_id_from_path(payload.get("video_path"))
        model_id = payload.get("model_id")
        if isinstance(video_id, str) and isinstance(model_id, str):
            video_to_models[video_id].add(model_id)

    complete_video_ids = sorted(video_id for video_id, model_ids in video_to_models.items() if expected.issubset(model_ids))
    complete_video_id_set = set(complete_video_ids)
    filtered_payloads = [
        payload
        for payload in payloads
        if (payload.get("__video_id__") or match_video_id_from_path(payload.get("video_path"))) in complete_video_id_set
    ]
    return filtered_payloads, {
        "required": True,
        "expected_model_ids": expected_model_ids,
        "raw_video_count": len(video_to_models),
        "complete_video_count": len(complete_video_ids),
        "excluded_incomplete_video_count": len(video_to_models) - len(complete_video_ids),
        "raw_prediction_file_count": len(payloads),
        "filtered_prediction_file_count": len(filtered_payloads),
        "complete_video_ids": complete_video_ids,
    }


def main() -> int:
    args = build_parser().parse_args()
    gt_by_video = load_ground_truth(args.ground_truth)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_base_url: str | list[str] | None
    if args.judge_base_urls:
        judge_base_url = [u for u in args.judge_base_urls if u]
    else:
        judge_base_url = args.judge_base_url
    base_judge = build_judge(args.judge_mode, args.judge_model, args.judge_api_key, judge_base_url)
    effective_judge_mode = "mixed_heuristic" if base_judge is None else "mixed_openai"
    judge_signature = build_judge_signature(args.judge_mode, effective_judge_mode, base_judge)

    discovered_files = discover_prediction_files(args.predictions_root, include_manual=args.include_manual)
    deduped_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for path in discovered_files:
        payload = load_prediction_payload(path)
        if payload is None:
            continue
        payload["__source_path__"] = str(path)
        payload["__source_mtime__"] = path.stat().st_mtime
        if args.model_id and payload.get("model_id") != args.model_id:
            continue
        source_video_id = path.stem
        video_id = source_video_id if source_video_id in gt_by_video else resolve_ground_truth_video_id(payload.get("video_path"), gt_by_video)
        if video_id not in gt_by_video:
            continue
        payload["__video_id__"] = video_id
        key = (payload["model_id"], video_id)
        existing = deduped_payloads.get(key)
        if existing is None or payload["__source_mtime__"] >= existing["__source_mtime__"]:
            deduped_payloads[key] = payload

    raw_payloads = list(deduped_payloads.values())
    mixed_coverage_meta = {
        "required": False,
        "expected_model_ids": [],
        "raw_video_count": len({payload.get("__video_id__") for payload in raw_payloads if payload.get("__video_id__")}),
        "complete_video_count": 0,
        "excluded_incomplete_video_count": 0,
        "raw_prediction_file_count": len(raw_payloads),
        "filtered_prediction_file_count": len(raw_payloads),
        "complete_video_ids": [],
    }
    selected_payloads = raw_payloads
    full_coverage_meta = mixed_coverage_meta
    if args.require_complete_videos:
        expected_model_ids = resolve_expected_model_ids(args.model_id, args.expected_models)
        selected_payloads, full_coverage_meta = filter_complete_video_payloads(raw_payloads, expected_model_ids)

    per_video_inputs = [
        (payload.get("__video_id__"), payload)
        for payload in selected_payloads
    ]
    per_video_inputs = [(video_id, payload) for video_id, payload in per_video_inputs if video_id]

    cache_root = output_dir / PER_PREDICTION_CACHE_DIRNAME
    state_path = output_dir / "state.json"
    started_at = now_iso()
    progress_lock = threading.Lock()
    progress = {
        "total_predictions": len(per_video_inputs),
        "completed_predictions": 0,
        "cache_hits": 0,
    }

    def write_state(state: str) -> None:
        write_json_atomic(
            state_path,
            {
                "state": state,
                "started_at": started_at,
                "updated_at": now_iso(),
                "judge_mode": effective_judge_mode,
                "requested_judge_mode": args.judge_mode,
                "ground_truth_path": str(args.ground_truth),
                "predictions_root": str(args.predictions_root),
                "output_dir": str(output_dir),
                "cache_dir": str(cache_root),
                "resume": args.resume,
                **progress,
                "pending_predictions": max(0, progress["total_predictions"] - progress["completed_predictions"]),
            },
        )

    def finalize_result(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        finalized = deepcopy(result)
        finalized["prediction_path"] = str(payload.get("__source_path__", finalized.get("prediction_path", "")))
        finalized["backend"] = payload.get("backend")
        return finalized

    def mark_progress(*, cache_hit: bool) -> None:
        with progress_lock:
            progress["completed_predictions"] += 1
            if cache_hit:
                progress["cache_hits"] += 1
            write_state("running")

    thread_local = threading.local()

    def evaluate_one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        video_id, payload = item
        gt_entry = gt_by_video[video_id]
        prediction_hash = hash_payload(semantic_prediction_payload(payload))
        ground_truth_hash = hash_payload(gt_entry.as_dict())
        cache_path = cache_path_for_prediction(output_dir, payload["model_id"], video_id)

        if args.resume:
            cached_result = load_cached_evaluation(
                cache_path,
                model_id=payload["model_id"],
                video_id=video_id,
                prediction_hash=prediction_hash,
                ground_truth_hash=ground_truth_hash,
                judge_signature=judge_signature,
            )
            if cached_result is not None:
                mark_progress(cache_hit=True)
                return finalize_result(cached_result, payload)

        if not hasattr(thread_local, "judge"):
            thread_local.judge = build_judge(
                args.judge_mode,
                args.judge_model,
                args.judge_api_key,
                judge_base_url,
            )
        result = evaluate_prediction(gt_entry, payload, thread_local.judge)
        result = finalize_result(result, payload)
        write_cached_evaluation(
            cache_path,
            model_id=payload["model_id"],
            video_id=video_id,
            prediction_hash=prediction_hash,
            ground_truth_hash=ground_truth_hash,
            judge_signature=judge_signature,
            result=result,
        )
        mark_progress(cache_hit=False)
        return result

    write_state("running")
    if args.workers <= 1 or len(per_video_inputs) <= 1:
        all_per_video_results = [evaluate_one(item) for item in per_video_inputs]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            all_per_video_results = list(executor.map(evaluate_one, per_video_inputs))
    all_per_video_results.sort(key=lambda record: (record["model_id"], record["video_id"]))

    selected_prediction_keys = {
        (payload["model_id"], payload.get("__video_id__") or match_video_id_from_path(payload.get("video_path")))
        for payload in selected_payloads
    }
    per_video_results = [
        record
        for record in all_per_video_results
        if (record["model_id"], record["video_id"]) in selected_prediction_keys
    ]

    mixed_model_summary = summarize_by_model(all_per_video_results)
    model_summary = summarize_by_model(per_video_results)
    selected_basis = "complete_videos" if args.require_complete_videos and per_video_results else "mixed_all_valid"
    if selected_basis == "mixed_all_valid":
        per_video_results = all_per_video_results
        model_summary = mixed_model_summary
    per_video_results.sort(key=lambda record: (record["model_id"], record["video_id"]))

    structured_predictions = [record["structured_prediction"] for record in per_video_results]
    ground_truth_dump = {video_id: entry.as_dict() for video_id, entry in gt_by_video.items()}

    summary_payload = {
        "generated_at": now_iso(),
        "judge_mode": effective_judge_mode,
        "ground_truth_path": str(args.ground_truth),
        "predictions_root": str(args.predictions_root),
        "coverage_basis": selected_basis,
        "mixed_model_coverage": mixed_coverage_meta,
        "full_model_coverage": full_coverage_meta,
        "mixed_video_count": len({record["video_id"] for record in all_per_video_results}),
        "mixed_prediction_file_count": len(all_per_video_results),
        "mixed_model_summary": mixed_model_summary,
        "video_count": len({record["video_id"] for record in per_video_results}),
        "prediction_file_count_raw": mixed_coverage_meta["raw_prediction_file_count"],
        "prediction_file_count": len(per_video_results),
        "model_summary": model_summary,
    }

    write_json_atomic(output_dir / "ground_truth_structured.json", ground_truth_dump)
    write_json_atomic(output_dir / "structured_predictions.json", structured_predictions)
    write_json_atomic(output_dir / "per_video_scores.json", per_video_results)
    write_json_atomic(output_dir / "summary.json", summary_payload)
    write_summary_csv(output_dir / "summary.csv", per_video_results)
    write_state("completed")

    print(
        format_json(
            {
                "generated_at": summary_payload["generated_at"],
                "judge_mode": effective_judge_mode,
                "prediction_file_count": len(per_video_results),
                "cache_hits": progress["cache_hits"],
                "workers": max(1, args.workers),
                "models": model_summary,
                "output_dir": str(output_dir),
            }
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

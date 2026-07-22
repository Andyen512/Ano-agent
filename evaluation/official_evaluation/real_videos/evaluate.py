#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = Path(__file__).resolve().parents[2]
OFFICIAL_EVAL_DIR = Path(__file__).resolve().parent
for p in (str(PROJECT_ROOT), str(EVALUATION_DIR), str(OFFICIAL_EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluation.common import (
    RISK_STATUS_NO_ANOMALY,
    build_judge,
    format_json,
    judge_or_heuristic,
    normalize_risk_status_label,
    normalize_risk_type_label,
    normalize_text,
    text_keyword_groups,
    heuristic_group_score,
    time_iou,
    start_time_error,
    end_time_error,
    duration_error,
    merge_spans,
)

from evaluation.official_evaluation.real_videos.common_real import (
    RealVideoGT,
    basename_from_video_id,
    discover_prediction_files,
    extract_video_id_from_prediction,
    load_real_video_gt,
    match_prediction_to_gt,
    parse_prediction_response,
    parse_prediction_solution,
    normalize_risk_type_cn,
)

DEFAULT_GT_DIR = PROJECT_ROOT / "data" / "public_data_release" / "annotations" / "real_videos"
DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "prediction" / "real_videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "real_videos"
EVALUATION_CACHE_VERSION = 2

_GT_BASENAME_CACHE: dict[str, list[str]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(format_json(payload), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate real_videos model predictions across 5 dimensions.")
    p.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR,
                   help="Directory containing perception/cognition/grounding/planning GT JSON files.")
    p.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT,
                   help="Root directory of model prediction outputs.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for evaluation artifacts.")
    p.add_argument("--model-id", help="Filter to a single model_id.")
    p.add_argument("--judge-mode", choices=("auto", "heuristic", "openai"), default="auto",
                   help="Heuristic-only or OpenAI-compatible LLM judge.")
    p.add_argument("--judge-model", help="Judge model name for OpenAI-compatible endpoint.")
    p.add_argument("--judge-api-key", help="Judge API key (falls back to env).")
    p.add_argument("--judge-base-url", help="Single OpenAI-compatible base URL.")
    p.add_argument("--judge-base-urls", nargs="*",
                   help="Multiple base URLs dispatched round-robin.")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for evaluation.")
    p.add_argument("--resume", action="store_true", help="Reuse compatible per-prediction caches.")
    return p


def _list_match_heuristic(gt_list: tuple[str, ...], pred_list: list[str] | None) -> tuple[float, dict[str, Any]]:
    if not gt_list or len(gt_list) == 0:
        return 0.0, {"not_annotated": True, "message": "GT list is empty"}
    pred_list = pred_list or []
    pred_set = {normalize_text(p) for p in pred_list if p}
    matched: list[str] = []
    for item in gt_list:
        if normalize_text(item) in pred_set:
            matched.append(item)
        else:
            for pred_item in pred_set:
                if normalize_text(item) in pred_item or pred_item in normalize_text(item):
                    matched.append(item)
                    break
    ratio = len(matched) / len(gt_list)
    score = round(5.0 * ratio, 3)
    return score, {"matched": matched, "total": len(gt_list), "match_ratio": ratio}


def _heuristic_text_match(gt_text: str, pred_text: str, max_score: float = 5.0) -> tuple[float, dict[str, Any]]:
    if not gt_text or not pred_text:
        return 0.0, {"match_ratio": 0.0, "matched_groups": []}
    groups = text_keyword_groups(gt_text, max_keywords=10)
    return heuristic_group_score(pred_text, groups, max_score=max_score)


def _causal_chain_component_match(
    causal_chain: tuple[dict[str, str], ...], pred_text: str,
) -> tuple[float, dict[str, Any]]:
    if not causal_chain:
        return 0.0, {"not_annotated": True, "message": "causal_chain not in GT"}

    norm_pred = normalize_text(pred_text)
    total_components = 0
    matched_components = 0
    link_details: list[dict[str, Any]] = []

    for link in causal_chain:
        source = link.get("source", "").strip()
        action = link.get("action", "").strip()
        consequence = link.get("consequence", "").strip()

        s_ok = not source or normalize_text(source) in norm_pred
        a_ok = not action or normalize_text(action) in norm_pred
        c_ok = not consequence or normalize_text(consequence) in norm_pred

        comps = 0
        matched = 0
        if source:
            comps += 1
            if s_ok:
                matched += 1
        if action:
            comps += 1
            if a_ok:
                matched += 1
        if consequence:
            comps += 1
            if c_ok:
                matched += 1

        total_components += comps
        matched_components += matched
        link_details.append({
            "source": source, "action": action, "consequence": consequence,
            "source_match": s_ok, "action_match": a_ok, "consequence_match": c_ok,
            "components": comps, "matched": matched,
        })

    if total_components == 0:
        return 0.0, {"not_annotated": True}

    ratio = matched_components / total_components
    score = round(5.0 * ratio, 3)
    return score, {
        "total_links": len(causal_chain),
        "total_components": total_components,
        "matched_components": matched_components,
        "match_ratio": round(ratio, 4),
        "link_details": link_details,
    }


def evaluate_one_prediction(
    gt: RealVideoGT,
    parsed_pred: dict[str, Any],
    raw_response: str,
    judge: Any,
) -> dict[str, Any]:
    is_normal = gt.risk_status == RISK_STATUS_NO_ANOMALY
    gt_status = gt.risk_status
    pred_status = normalize_risk_status_label(parsed_pred.get("risk_status", ""))
    gt_risk_type = normalize_risk_type_label(gt.risk_type)
    pred_risk_type = normalize_risk_type_cn(parsed_pred.get("risk_type", ""))

    # ── Perception ──────────────────────────────────────────────
    risk_status_acc = 1.0 if pred_status == gt_status else 0.0

    if is_normal:
        risk_type_acc = 1.0
        risk_source_score: float = 5.0
        abnormal_action_score: float = 5.0
        affected_object_score: float = 5.0
    else:
        if not gt_risk_type:
            risk_type_acc = 1.0
        else:
            risk_type_acc = 1.0 if pred_risk_type == gt_risk_type else 0.0

        risk_source_heuristic, _ = _list_match_heuristic(
            gt.risk_sources, parsed_pred.get("risk_sources")
        )
        risk_source_eval = judge_or_heuristic(
            "risk_source_recognition",
            {"gt_risk_sources": list(gt.risk_sources)},
            {
                "pred_risk_sources": parsed_pred.get("risk_sources", []),
                "raw_response": raw_response,
            },
            risk_source_heuristic,
            {},
            judge,
        )
        risk_source_score = float(risk_source_eval["score"])

        abnormal_action_heuristic, _ = _list_match_heuristic(
            gt.abnormal_actions, parsed_pred.get("abnormal_actions")
        )
        abnormal_action_eval = judge_or_heuristic(
            "abnormal_action_recognition",
            {"gt_abnormal_actions": list(gt.abnormal_actions)},
            {
                "pred_abnormal_actions": parsed_pred.get("abnormal_actions", []),
                "raw_response": raw_response,
            },
            abnormal_action_heuristic,
            {},
            judge,
        )
        abnormal_action_score = float(abnormal_action_eval["score"])

        affected_object_heuristic, _ = _list_match_heuristic(
            gt.affected_objects, parsed_pred.get("affected_objects")
        )
        affected_object_eval = judge_or_heuristic(
            "affected_object_recognition",
            {"gt_affected_objects": list(gt.affected_objects)},
            {
                "pred_affected_objects": parsed_pred.get("affected_objects", []),
                "raw_response": raw_response,
            },
            affected_object_heuristic,
            {},
            judge,
        )
        affected_object_score = float(affected_object_eval["score"])

    # ── Cognition ───────────────────────────────────────────────
    if is_normal:
        risk_desc_eval = judge_or_heuristic(
            "risk_description_normal",
            {"gt_risk_description": gt.risk_description},
            {
                "video_description": parsed_pred.get("video_description", ""),
                "raw_response": raw_response,
            },
            5.0,
            {},
            judge,
        )
        risk_desc_score = float(risk_desc_eval["score"])
        conseq_score: float = 5.0
        factual_boundary_score: float = 5.0
        causal_chain_score: float = 5.0
    else:
        risk_desc_heuristic, risk_desc_meta = _heuristic_text_match(
            gt.risk_description or gt.event_summary or "",
            parsed_pred.get("risk_description", ""),
        )
        risk_desc_eval = judge_or_heuristic(
            "risk_description",
            {
                "risk_status": gt_status,
                "gt_risk_description": gt.risk_description,
                "gt_risk_type": gt.risk_type,
            },
            {
                "risk_description": parsed_pred.get("risk_description", ""),
                "raw_response": raw_response,
            },
            risk_desc_heuristic,
            risk_desc_meta,
            judge,
        )
        risk_desc_score = float(risk_desc_eval["score"])

        conseq_heuristic, _ = _list_match_heuristic(
            gt.consequence_understanding,
            parsed_pred.get("consequence_understanding"),
        )
        conseq_eval = judge_or_heuristic(
            "consequence_understanding",
            {
                "gt_consequences": list(gt.consequence_understanding),
                "gt_risk_description": gt.risk_description,
            },
            {
                "pred_consequences": parsed_pred.get("consequence_understanding", []),
                "raw_response": raw_response,
            },
            conseq_heuristic,
            {},
            judge,
        )
        conseq_score = float(conseq_eval["score"])

        factual_eval = judge_or_heuristic(
            "factual_boundary_error",
            {
                "gt_risk_status": gt_status,
                "gt_summary": gt.risk_description,
            },
            {
                "pred_text": parsed_pred.get("risk_description", ""),
                "raw_response": raw_response,
            },
            3.0,
            {},
            judge,
        )
        factual_boundary_score = float(factual_eval["score"])

        causal_heuristic, causal_meta = _causal_chain_component_match(
            gt.causal_chain,
            parsed_pred.get("risk_description", "") or raw_response,
        )
        causal_eval = judge_or_heuristic(
            "causal_chain_understanding",
            {
                "gt_causal_chain": list(gt.causal_chain),
                "gt_risk_description": gt.risk_description,
            },
            {
                "pred_causal_chain": parsed_pred.get("causal_chain", []),
                "raw_response": raw_response,
            },
            causal_heuristic,
            causal_meta,
            judge,
        )
        causal_chain_score = float(causal_eval["score"])

    # ── Temporal Grounding ──────────────────────────────────────
    gt_time_spans = gt.time_spans or []
    pred_time_spans = parsed_pred.get("time_spans") or []
    if isinstance(pred_time_spans, str):
        pred_time_spans = []
    pred_time_spans = merge_spans([
        [float(s), float(e)] for s, e in (pred_time_spans or [])
        if isinstance((pred_time_spans or []), list)
    ])

    if is_normal:
        has_pred_time_spans = bool(pred_time_spans)
        iou_score = 0.0 if has_pred_time_spans else 1.0
        start_err = 0.0
        end_err = 0.0
        dur_err = 0.0
    else:
        iou_score = time_iou(gt_time_spans, pred_time_spans) if gt_time_spans else (
            1.0 if not pred_time_spans else 0.0
        )
        start_err = start_time_error(gt_time_spans, pred_time_spans)
        end_err = end_time_error(gt_time_spans, pred_time_spans)
        dur_err = duration_error(gt_time_spans, pred_time_spans)

    # ── Planning ────────────────────────────────────────────────
    pred_solutions = parse_prediction_solution(parsed_pred)

    if is_normal:
        person_sol_score: float = 5.0
        hazard_sol_score: float = 5.0
        overall_sol_score: float = 5.0
    else:
        gt_sol_map = {
            "person_solution": gt.person_solution,
            "hazard_solution": gt.hazard_solution,
            "overall_solution": gt.prevention_solution,
        }

        def _score_solution(section_key: str, gt_sol_text: str) -> float:
            pred_text = pred_solutions.get(section_key, "") or parsed_pred.get("solution", "")
            if not gt_sol_text:
                return 0.0
            eval_result = judge_or_heuristic(
                f"solution_{section_key}",
                {
                    "solution_section": gt_sol_text,
                    "risk_type": gt.risk_type,
                    "risk_description": gt.risk_description,
                },
                {
                    "solution_section": pred_text,
                    "full_solution": parsed_pred.get("solution"),
                    "raw_response": raw_response,
                },
                0.0,
                {},
                judge,
            )
            return float(eval_result["score"])

        person_sol_score = _score_solution("person_solution", gt.person_solution)
        hazard_sol_score = _score_solution("hazard_solution", gt.hazard_solution)
        overall_sol_score = _score_solution("overall_solution", gt.prevention_solution)

    solution_score = round(
        (person_sol_score + hazard_sol_score + overall_sol_score) / 3.0, 3
    )

    # ── Overall ─────────────────────────────────────────────────
    def _norm(v: float, scale: float = 5.0) -> float:
        return min(max(v / scale, 0.0), 1.0)

    if is_normal:
        perception = _norm(risk_status_acc * 5.0)
        cognition = _norm(risk_desc_score)
        overall = round(
            (0.5 * perception + 0.5 * cognition) * 100.0, 3
        )
        temporal = iou_score
        intervention = _norm(solution_score)
    else:
        perception = (
            0.20 * _norm(risk_status_acc * 5.0)
            + 0.20 * _norm(risk_type_acc * 5.0)
            + 0.20 * _norm(risk_source_score)
            + 0.20 * _norm(abnormal_action_score)
            + 0.20 * _norm(affected_object_score)
        )
        cognition = (
            0.25 * _norm(risk_desc_score)
            + 0.25 * _norm(conseq_score)
            + 0.25 * _norm(factual_boundary_score)
            + 0.25 * _norm(causal_chain_score)
        )
        temporal = iou_score
        intervention = _norm(solution_score)
        overall = round(
            (0.25 * perception + 0.25 * cognition
             + 0.25 * temporal + 0.25 * intervention) * 100.0,
            3,
        )
    category_scores = {
        "perception": round(perception * 100.0, 3),
        "cognition": round(cognition * 100.0, 3),
        "temporal_grounding": round(temporal * 100.0, 3),
        "intervention_planning": round(intervention * 100.0, 3),
    }

    return {
        "video_id": gt.video_id,
        "risk_status": gt_status,
        "metrics": {
            # Perception
            "risk_status_accuracy": round(5.0 * risk_status_acc, 3),
            "risk_type_accuracy": round(5.0 * risk_type_acc, 3),
            "risk_source_recognition_score": round(risk_source_score, 3),
            "abnormal_action_recognition_score": round(abnormal_action_score, 3),
            "affected_object_recognition_score": round(affected_object_score, 3),
            # Cognition
            "risk_description_score": round(risk_desc_score, 3),
            "consequence_understanding_score": round(conseq_score, 3),
            "factual_boundary_error_score": round(factual_boundary_score, 3),
            "causal_chain_understanding_score": round(causal_chain_score, 3),
            # Temporal
            "time_iou": round(iou_score, 4),
            "start_time_error": round(start_err, 3),
            "end_time_error": round(end_err, 3),
            "duration_error": round(dur_err, 3),
            # Planning
            "solution_score": solution_score,
            "person_solution_score": round(person_sol_score, 3),
            "hazard_solution_score": round(hazard_sol_score, 3),
            "overall_solution_score": round(overall_sol_score, 3),
            # Overall
            "overall_score": overall,
        },
        "category_scores": category_scores,
        "ground_truth": gt.perception,
    }


def summarize_by_model(per_video: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in per_video:
        grouped[record["model_id"]].append(record)

    summary: dict[str, Any] = {}
    for model_id, records in grouped.items():
        count = len(records)
        metric_names = list(records[0]["metrics"].keys()) if records else []
        averages = {
            name: round(sum(r["metrics"].get(name, 0.0) for r in records) / count, 4)
            for name in metric_names
        }
        category_keys = list(records[0].get("category_scores", {}).keys()) if records else []
        cat_avgs = {}
        for ck in category_keys:
            vals = [r.get("category_scores", {}).get(ck, 0.0) for r in records]
            cat_avgs[ck] = round(sum(vals) / count, 3) if vals else 0.0

        summary[model_id] = {
            "sample_count": count,
            "averages": averages,
            "category_averages": cat_avgs,
        }
    return summary


def main() -> int:
    args = build_parser().parse_args()

    print(f"Loading GT from {args.gt_dir} ...")
    gt_by_video = load_real_video_gt(args.gt_dir)
    print(f"  Loaded {len(gt_by_video)} GT entries.")

    from evaluation.official_evaluation.real_videos.common_real import _build_basename_index
    gt_basename_index = _build_basename_index(gt_by_video)

    print(f"Discovering predictions from {args.predictions_root} ...")
    raw_payloads = discover_prediction_files(args.predictions_root, args.model_id)
    print(f"  Found {len(raw_payloads)} prediction files.")

    # Build judge
    judge_base_url = args.judge_base_urls if args.judge_base_urls else args.judge_base_url
    base_judge = build_judge(args.judge_mode, args.judge_model, args.judge_api_key, judge_base_url)
    effective_judge_mode = "mixed_heuristic" if base_judge is None else "mixed_openai"

    # Match predictions to GT
    matched_pairs: list[tuple[RealVideoGT, dict[str, Any], dict[str, Any]]] = []
    unmatched = 0
    parse_fail = 0
    seen_videos: set[tuple[str, str]] = set()

    for payload in raw_payloads:
        model_id = payload.get("model_id", "")
        source_path = payload.get("__source_path__", "")
        filename_stem = Path(source_path).stem if source_path else ""

        gt_video_id = match_prediction_to_gt(
            filename_stem, payload.get("video_path"),
            gt_by_video, gt_basename_index,
        )
        if gt_video_id is None:
            unmatched += 1
            continue

        # 去重：同一 (model_id, video_id) 只保留第一个
        dedup_key = (model_id, gt_video_id)
        if dedup_key in seen_videos:
            unmatched += 1
            continue
        seen_videos.add(dedup_key)

        parsed = parse_prediction_response(payload)
        if parsed is None:
            parse_fail += 1
            continue

        matched_pairs.append((gt_by_video[gt_video_id], payload, parsed))

    print(f"  Matched: {len(matched_pairs)}, unmatched: {unmatched}, parse_fail: {parse_fail}, dup_skipped: {len(seen_videos) - len(matched_pairs) - parse_fail}")

    output_dir = args.output_dir
    ensure_dir(output_dir)

    # Build judge signature for caching
    judge_signature = {
        "cache_version": EVALUATION_CACHE_VERSION,
        "judge_mode": effective_judge_mode,
        "judge_model": getattr(base_judge, "model", None) if base_judge else None,
    }

    per_video_inputs = [
        (gt, payload, parsed)
        for gt, payload, parsed in matched_pairs
    ]

    progress_lock = threading.Lock()
    progress: dict[str, int] = {"total": len(per_video_inputs), "completed": 0, "cache_hits": 0}
    thread_local = threading.local()

    def evaluate_one(item: tuple[RealVideoGT, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        gt, payload, parsed_pred = item
        model_id = payload["model_id"]
        video_id = gt.video_id
        raw_response = (
            payload.get("response", "")
            if isinstance(payload.get("response", ""), str)
            else json.dumps(payload.get("response", ""))
        )

        if not hasattr(thread_local, "judge"):
            thread_local.judge = build_judge(
                args.judge_mode, args.judge_model,
                args.judge_api_key, judge_base_url,
            )

        result = evaluate_one_prediction(gt, parsed_pred, raw_response, thread_local.judge)
        result["model_id"] = model_id
        result["backend"] = payload.get("backend", "")

        with progress_lock:
            progress["completed"] += 1
        return result

    print(f"Evaluating {len(per_video_inputs)} predictions with {args.workers} workers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(evaluate_one, per_video_inputs))

    results.sort(key=lambda r: (r.get("model_id", ""), r.get("video_id", "")))

    # Summarize
    model_summary = summarize_by_model(results)

    # Output
    summary_payload = {
        "generated_at": now_iso(),
        "judge_mode": effective_judge_mode,
        "gt_dir": str(args.gt_dir),
        "predictions_root": str(args.predictions_root),
        "total_gt_entries": len(gt_by_video),
        "total_predictions_evaluated": len(results),
        "model_summary": model_summary,
    }

    write_json_atomic(output_dir / "per_video_scores.json", results)
    write_json_atomic(output_dir / "summary.json", summary_payload)

    print(format_json(summary_payload).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

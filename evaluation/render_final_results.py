#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    "overall_score",
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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render LifeBench final_model_results.csv and FINAL_RESULTS.md from evaluation outputs.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Path to evaluation summary.json.")
    parser.add_argument("--status-dir", type=Path, help="Optional benchmark status directory.")
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Output CSV path. Defaults to <summary-json dir>/final_model_results.csv.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        help="Output markdown path. Defaults to <summary-json dir>/FINAL_RESULTS.md.",
    )
    return parser


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_statuses(status_dir: Path | None) -> dict[str, str]:
    if status_dir is None or not status_dir.exists():
        return {}

    statuses: dict[str, str] = {}
    for path in sorted(status_dir.glob("*.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        model_id = payload.get("model_id")
        state = payload.get("state")
        if isinstance(model_id, str) and isinstance(state, str):
            statuses[model_id] = state
    return statuses


def metric_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def resolve_status(summary_payload: dict[str, Any], entry: dict[str, Any], raw_status: str | None) -> str:
    if not entry:
        return raw_status or "unknown"

    sample_count = int(entry.get("sample_count") or 0)
    coverage_basis = summary_payload.get("coverage_basis")
    video_count = int(summary_payload.get("video_count") or 0)

    # Final result tables should reflect the completeness of the evaluated result set.
    # If this model already has scores for every evaluated video, a stale runtime status
    # file from a later abandoned rerun should not keep the row marked as "running".
    if coverage_basis == "complete_videos" and video_count > 0 and sample_count >= video_count:
        return "completed"

    return raw_status or "completed"


def build_rows(summary_payload: dict[str, Any], status_map: dict[str, str]) -> list[dict[str, Any]]:
    model_summary = summary_payload.get("model_summary") or {}
    if not model_summary:
        model_summary = summary_payload.get("mixed_model_summary") or {}
    model_ids = sorted(set(model_summary) | set(status_map))
    rows: list[dict[str, Any]] = []

    for model_id in model_ids:
        entry = model_summary.get(model_id) or {}
        averages = entry.get("averages") or {}
        row = {
            "model_id": model_id,
            "status": resolve_status(summary_payload, entry, status_map.get(model_id)),
            "sample_count": int(entry.get("sample_count") or 0),
            "overall_score": metric_value(averages.get("overall_score")),
            "risk_status_accuracy": metric_value(averages.get("risk_status_accuracy")),
            "risk_type_accuracy": metric_value(averages.get("risk_type_accuracy")),
            "risk_type_score": metric_value(averages.get("risk_type_score")),
            "time_iou": metric_value(averages.get("time_iou")),
            "video_description_score": metric_value(averages.get("video_description_score")),
            "risk_description_score": metric_value(averages.get("risk_description_score")),
            "solution_score": metric_value(averages.get("solution_score")),
            "person_solution_score": metric_value(averages.get("person_solution_score")),
            "hazard_solution_score": metric_value(averages.get("hazard_solution_score")),
            "overall_solution_score": metric_value(averages.get("overall_solution_score")),
        }
        rows.append(row)

    rows.sort(key=lambda row: (-row["overall_score"], -row["sample_count"], row["model_id"]))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "model_id",
                "status",
                "sample_count",
                "overall_score",
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
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "overall_score": fmt_metric(row["overall_score"]),
                    "risk_status_accuracy": fmt_metric(row["risk_status_accuracy"]),
                    "risk_type_accuracy": fmt_metric(row["risk_type_accuracy"]),
                    "risk_type_score": fmt_metric(row["risk_type_score"]),
                    "time_iou": fmt_metric(row["time_iou"]),
                    "video_description_score": fmt_metric(row["video_description_score"]),
                    "risk_description_score": fmt_metric(row["risk_description_score"]),
                    "solution_score": fmt_metric(row["solution_score"]),
                    "person_solution_score": fmt_metric(row["person_solution_score"]),
                    "hazard_solution_score": fmt_metric(row["hazard_solution_score"]),
                    "overall_solution_score": fmt_metric(row["overall_solution_score"]),
                }
            )


def fmt_metric(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_markdown(summary_payload: dict[str, Any], rows: list[dict[str, Any]], csv_out: Path) -> str:
    coverage_basis = summary_payload.get("coverage_basis") or "complete_videos"
    is_mixed = coverage_basis == "mixed_all_valid"
    coverage = (
        summary_payload.get("mixed_model_coverage")
        if is_mixed
        else summary_payload.get("full_model_coverage")
    ) or {}
    require_complete = bool(coverage.get("required"))
    complete_video_count = summary_payload.get("video_count", 0)
    excluded_video_count = coverage.get("excluded_incomplete_video_count", 0)
    prediction_file_count = summary_payload.get("prediction_file_count", 0)
    if is_mixed:
        coverage_label = "混合评测（全部有效预测，不要求所有模型齐套）"
    else:
        coverage_label = f"仅统计{len(coverage.get('expected_model_ids', []))}个模型都有结果的视频" if require_complete else "全部有效预测"
    lines = [
        "# 最终结果汇总",
        "",
        f"生成时间: `{summary_payload.get('generated_at', 'unknown')}`",
        "",
        f"评测模式: `{summary_payload.get('judge_mode', 'unknown')}`",
        "",
        f"统计口径: `{coverage_label}`",
        "",
        f"统计视频数: `{complete_video_count}`",
        "",
        f"预测条数: `{prediction_file_count}`",
        "",
        f"排除未齐套视频数: `{excluded_video_count}`",
        "",
        "",
        "| 排名 | 模型 | 状态 | 样本数 | 总分 | 风险状态 | 风险类型Acc | 风险类型Score | 时间IoU | 视频描述 | 风险描述 | 解决方案 | 对人方案 | 对危险源方案 | 整体方案 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {rank} | `{model_id}` | `{status}` | {sample_count} | {overall_score} | {risk_status_accuracy} | {risk_type_accuracy} | {risk_type_score} | {time_iou} | {video_description_score} | {risk_description_score} | {solution_score} | {person_solution_score} | {hazard_solution_score} | {overall_solution_score} |".format(
                rank=row["rank"],
                model_id=row["model_id"],
                status=row["status"],
                sample_count=row["sample_count"],
                overall_score=fmt_metric(row["overall_score"]),
                risk_status_accuracy=fmt_metric(row["risk_status_accuracy"]),
                risk_type_accuracy=fmt_metric(row["risk_type_accuracy"]),
                risk_type_score=fmt_metric(row["risk_type_score"]),
                time_iou=fmt_metric(row["time_iou"]),
                video_description_score=fmt_metric(row["video_description_score"]),
                risk_description_score=fmt_metric(row["risk_description_score"]),
                solution_score=fmt_metric(row["solution_score"]),
                person_solution_score=fmt_metric(row["person_solution_score"]),
                hazard_solution_score=fmt_metric(row["hazard_solution_score"]),
                overall_solution_score=fmt_metric(row["overall_solution_score"]),
            )
        )

    lines.extend(
        [
            "",
            f"CSV 文件: [{csv_out.name}]({csv_out})",
            "",
        ]
    )
    return "\n".join(lines)


def default_paths(summary_json: Path, csv_out: Path | None, markdown_out: Path | None) -> tuple[Path, Path]:
    summary_dir = summary_json.parent
    return (
        csv_out or (summary_dir / "final_model_results.csv"),
        markdown_out or (summary_dir / "FINAL_RESULTS.md"),
    )


def main() -> int:
    args = build_parser().parse_args()
    summary_payload = load_json(args.summary_json)
    csv_out, markdown_out = default_paths(args.summary_json, args.csv_out, args.markdown_out)
    status_map = load_statuses(args.status_dir)
    rows = build_rows(summary_payload, status_map)

    write_csv(csv_out, rows)
    ensure_parent(markdown_out)
    markdown_out.write_text(build_markdown(summary_payload, rows, csv_out), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_json": str(args.summary_json),
                "status_dir": str(args.status_dir) if args.status_dir else None,
                "csv_out": str(csv_out),
                "markdown_out": str(markdown_out),
                "row_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

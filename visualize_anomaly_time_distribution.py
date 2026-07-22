#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "data_0315" / "gen_prompt_custom.txt"
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "data_0315" / "videos_flat"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_0315_analysis"
INTERVAL_COLOR = "#2F6690"

TIMELINE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s\s*([^；]+)")
DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)s")
SCENE_PATTERN = re.compile(r"场景：([^。]+)。")

ANOMALY_CUES = (
    "不慎",
    "突然",
    "开始",
    "出现",
    "导致",
    "冒出",
    "窜起",
    "窜高",
    "火花",
    "火苗",
    "火焰",
    "起火",
    "着火",
    "冒烟",
    "烟雾",
    "过热",
    "打滑",
    "滑倒",
    "失衡",
    "绊倒",
    "摔倒",
    "跌倒",
    "倒地",
    "滑落",
    "掉落",
    "坠落",
    "翻落",
    "脱手",
    "倾斜",
    "晃动",
    "碎裂",
    "破裂",
    "飞溅",
    "喷溅",
    "洒落",
    "回弹",
    "夹住",
    "夹到",
    "夹伤",
    "碰到",
    "撞到",
    "踉跄",
    "受惊",
    "后退",
    "烧焦",
    "热点",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize where anomalies occur in videos over time.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Path to gen_prompt_custom.txt.")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR, help="Directory containing flattened .mp4 videos.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots and summary files.")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins for normalized timeline summaries.")
    return parser


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": round_float(ordered[0]),
        "p10": round_float(np.percentile(ordered, 10)),
        "median": round_float(median(ordered)),
        "mean": round_float(mean(ordered)),
        "p90": round_float(np.percentile(ordered, 90)),
        "max": round_float(ordered[-1]),
    }


def parse_segments(payload: str) -> list[dict[str, Any]]:
    if "时间轴：" not in payload:
        return []
    timeline_text = payload.split("时间轴：", 1)[1].strip()
    segments = []
    for match in TIMELINE_PATTERN.finditer(timeline_text):
        segments.append(
            {
                "start": float(match.group(1)),
                "end": float(match.group(2)),
                "description": re.sub(r"\s+", " ", match.group(3).strip()),
            }
        )
    segments.sort(key=lambda item: (item["start"], item["end"]))
    return segments


def anomaly_score(description: str) -> int:
    return sum(1 for cue in ANOMALY_CUES if cue in description)


def infer_anomaly_start_index(segments: list[dict[str, Any]]) -> tuple[int | None, str]:
    if not segments:
        return None, "no_segments"

    scores = [anomaly_score(segment["description"]) for segment in segments]
    if scores[0] > 0:
        return 0, "first_segment_cue"

    for index in range(1, len(segments)):
        if scores[index] > 0:
            return index, "cue_after_setup"

    if len(segments) >= 2:
        return 1, "fallback_second_segment"
    return 0, "fallback_single_segment"


def collect_records(ground_truth_path: Path, videos_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_paths = {path.stem: path for path in sorted(videos_dir.glob("*.mp4"))}

    records: list[dict[str, Any]] = []
    missing_video_ids: list[str] = []
    total_entries = 0
    for raw_line in ground_truth_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        total_entries += 1
        video_id, payload = line.split("|", 1)
        video_id = video_id.strip()
        if video_id not in video_paths:
            missing_video_ids.append(video_id)
            continue
        segments = parse_segments(payload)
        anomaly_start_index, strategy = infer_anomaly_start_index(segments)
        if anomaly_start_index is None:
            continue
        anomaly_segments = segments[anomaly_start_index:]
        if not anomaly_segments:
            continue
        start_sec = min(segment["start"] for segment in anomaly_segments)
        end_sec = max(segment["end"] for segment in anomaly_segments)
        duration_match = DURATION_PATTERN.search(payload)
        duration_sec = float(duration_match.group(1)) if duration_match else 0.0
        if duration_sec <= 0:
            duration_sec = max(end_sec, start_sec, 1.0)
        scene_match = SCENE_PATTERN.search(payload)
        scene_text = scene_match.group(1).strip() if scene_match else ""
        start_ratio = max(0.0, min(1.0, start_sec / duration_sec))
        end_ratio = max(0.0, min(1.0, end_sec / duration_sec))
        if end_ratio < start_ratio:
            start_ratio, end_ratio = end_ratio, start_ratio
        midpoint_ratio = (start_ratio + end_ratio) / 2.0
        records.append(
            {
                "video_id": video_id,
                "video_path": str(video_paths[video_id]),
                "duration_sec": round_float(duration_sec),
                "risk_start_sec": round_float(start_sec),
                "risk_end_sec": round_float(end_sec),
                "risk_midpoint_sec": round_float((start_sec + end_sec) / 2.0),
                "risk_span_sec": round_float(max(0.0, end_sec - start_sec)),
                "risk_start_ratio": round_float(start_ratio),
                "risk_end_ratio": round_float(end_ratio),
                "risk_midpoint_ratio": round_float(midpoint_ratio),
                "risk_span_ratio": round_float(max(0.0, end_ratio - start_ratio)),
                "scene_text": scene_text,
                "timeline_segment_count": len(segments),
                "anomaly_start_strategy": strategy,
            }
        )

    meta = {
        "videos_in_dir": len(video_paths),
        "ground_truth_entries": total_entries,
        "matched_videos_with_anomaly_span": len(records),
        "ground_truth_missing_videos": len(missing_video_ids),
        "missing_video_ids_preview": missing_video_ids[:20],
    }
    return records, meta


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "video_path",
        "duration_sec",
        "risk_start_sec",
        "risk_end_sec",
        "risk_midpoint_sec",
        "risk_span_sec",
        "risk_start_ratio",
        "risk_end_ratio",
        "risk_midpoint_ratio",
        "risk_span_ratio",
        "scene_text",
        "timeline_segment_count",
        "anomaly_start_strategy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def coverage_curve(records: list[dict[str, Any]], resolution: int = 400) -> tuple[np.ndarray, np.ndarray]:
    positions = np.linspace(0.0, 1.0, resolution + 1)
    coverage = []
    for pos in positions:
        covered = sum(record["risk_start_ratio"] <= pos <= record["risk_end_ratio"] for record in records)
        coverage.append(covered / len(records) if records else 0.0)
    return positions, np.asarray(coverage)


def build_summary(records: list[dict[str, Any]], meta: dict[str, Any], bins: int) -> dict[str, Any]:
    starts = [record["risk_start_ratio"] for record in records]
    ends = [record["risk_end_ratio"] for record in records]
    mids = [record["risk_midpoint_ratio"] for record in records]
    spans = [record["risk_span_ratio"] for record in records]
    coverage_x, coverage_y = coverage_curve(records)
    peak_index = int(np.argmax(coverage_y)) if len(coverage_y) else 0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    hist_counts, _ = np.histogram(mids, bins=bin_edges)

    return {
        **meta,
        "normalized_position_stats": {
            "start_ratio": describe(starts),
            "midpoint_ratio": describe(mids),
            "end_ratio": describe(ends),
            "span_ratio": describe(spans),
        },
        "coverage_peak_ratio": round_float(float(coverage_x[peak_index])) if len(coverage_x) else 0.0,
        "coverage_peak_share": round_float(float(coverage_y[peak_index])) if len(coverage_y) else 0.0,
        "midpoint_histogram": [
            {
                "bin_start": round_float(float(bin_edges[index])),
                "bin_end": round_float(float(bin_edges[index + 1])),
                "count": int(hist_counts[index]),
            }
            for index in range(len(hist_counts))
        ],
        "anomaly_start_strategy_counts": dict(sorted(Counter(record["anomaly_start_strategy"] for record in records).items())),
    }


def plot(records: list[dict[str, Any]], summary: dict[str, Any], output_path: Path, bins: int) -> None:
    records_sorted = sorted(records, key=lambda item: (item["risk_start_ratio"], item["risk_end_ratio"], item["video_id"]))
    starts = np.asarray([record["risk_start_ratio"] for record in records_sorted])
    ends = np.asarray([record["risk_end_ratio"] for record in records_sorted])
    mids = np.asarray([record["risk_midpoint_ratio"] for record in records_sorted])
    coverage_x, coverage_y = coverage_curve(records_sorted)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 14),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.4, 1.5, 1.5]},
    )

    ax_intervals, ax_coverage, ax_hist = axes

    for index, record in enumerate(records_sorted):
        ax_intervals.broken_barh(
            [(record["risk_start_ratio"], record["risk_span_ratio"])],
            (index - 0.4, 0.8),
            facecolors=INTERVAL_COLOR,
            edgecolors="none",
            alpha=0.85,
        )

    ax_intervals.set_xlim(0.0, 1.0)
    ax_intervals.set_ylim(-1, len(records_sorted))
    ax_intervals.set_yticks([])
    ax_intervals.set_xlabel("Normalized video time")
    ax_intervals.set_title(
        f"Anomaly interval in each labeled video (n={len(records_sorted)})\n"
        "Each row is one video, sorted by inferred anomaly start time."
    )
    ax_intervals.grid(axis="x", alpha=0.2)

    ax_coverage.fill_between(coverage_x, coverage_y, color="#4C956C", alpha=0.3)
    ax_coverage.plot(coverage_x, coverage_y, color="#2C6E49", linewidth=2.5)
    peak_ratio = summary["coverage_peak_ratio"]
    peak_share = summary["coverage_peak_share"]
    ax_coverage.axvline(peak_ratio, color="#BC4749", linestyle="--", linewidth=1.5)
    ax_coverage.text(
        peak_ratio,
        peak_share,
        f" peak {peak_ratio:.2f}\n share {peak_share:.2%}",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax_coverage.set_xlim(0.0, 1.0)
    ax_coverage.set_ylim(0.0, min(1.0, max(coverage_y) * 1.15 if len(coverage_y) else 1.0))
    ax_coverage.set_xlabel("Normalized video time")
    ax_coverage.set_ylabel("Share of videos")
    ax_coverage.set_title("How many videos are inside an anomaly interval at each time position")
    ax_coverage.grid(alpha=0.2)

    hist_bins = np.linspace(0.0, 1.0, bins + 1)
    ax_hist.hist(starts, bins=hist_bins, histtype="step", linewidth=2.0, color="#1D3557", label="Start")
    ax_hist.hist(mids, bins=hist_bins, histtype="step", linewidth=2.0, color="#E76F51", label="Midpoint")
    ax_hist.hist(ends, bins=hist_bins, histtype="step", linewidth=2.0, color="#2A9D8F", label="End")
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Normalized video time")
    ax_hist.set_ylabel("Video count")
    ax_hist.set_title("Distribution of anomaly start / midpoint / end positions")
    ax_hist.grid(alpha=0.2)
    ax_hist.legend(frameon=False, ncol=3, loc="upper center")

    fig.suptitle("data_0315 anomaly timing distribution", fontsize=16, y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    records, meta = collect_records(args.ground_truth, args.videos_dir)
    if not records:
        raise FileNotFoundError("No matched videos with anomaly spans were found.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "anomaly_time_distribution.csv"
    json_path = output_dir / "anomaly_time_distribution_summary.json"
    png_path = output_dir / "anomaly_time_distribution.png"

    write_csv(csv_path, records)
    summary = build_summary(records, meta, bins=args.bins)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(records, summary, png_path, bins=args.bins)

    print(
        json.dumps(
            {
                "ground_truth": str(args.ground_truth),
                "videos_dir": str(args.videos_dir),
                "output_dir": str(output_dir),
                "plot_path": str(png_path),
                "csv_path": str(csv_path),
                "summary_path": str(json_path),
                "matched_videos_with_anomaly_span": len(records),
                "coverage_peak_ratio": summary["coverage_peak_ratio"],
                "coverage_peak_share": summary["coverage_peak_share"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

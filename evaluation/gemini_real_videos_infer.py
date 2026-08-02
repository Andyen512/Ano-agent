#!/usr/bin/env python3
"""Run one-call Gemini predictions for the LifeBench real-video release."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .gemini_infer import DEFAULT_API_KEY, build_messages, call_gemini
except ImportError:
    from gemini_infer import DEFAULT_API_KEY, build_messages, call_gemini


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "real_videos"
PREDICTION_ROOT = PROJECT_ROOT / "data" / "public_data_release" / "prediction" / "real_videos"
SKIP_FILE = PROJECT_ROOT / "data" / "public_data_release" / "real_videos_infer_skip.txt"
MODEL_DIR_NAME = "gemini_3_1_pro"

SINGLE_PASS_PROMPT = """你是视频安全分析专家。请完整观看视频，只输出一个 JSON 对象，不要输出解释、前缀、后缀或 Markdown。

请一次性完成视频安全分析，并严格输出以下全部字段：
- risk_status: 只能是 normal、risk_only、abnormal
- risk_type: 只能从 跌倒失稳、动物攻击、碰撞砸伤、用电安全、高温火源、人际冲突、中毒误食、锐器危险、陌生盗窃/入侵、溺水、自然灾害 中选择；normal 时为空字符串
- risk_sources: 风险源字符串数组；normal 时为空数组
- abnormal_actions: 异常行为字符串数组；normal 时为空数组
- affected_objects: 受影响对象字符串数组，只能使用 儿童、年轻人、中年人、老年人、所有人；normal 时为空数组
- risk_description: 风险或异常描述；normal 时为空字符串
- consequence_understanding: 可能后果；normal 时为空字符串
- causal_chain: 风险源 → 异常行为 → 后果；normal 时为空字符串
- causal_chain_text: 自然语言描述因果链；normal 时为空字符串
- event_summary: 事件完整摘要；normal 时为空字符串
- time_spans: 异常或风险发生的整秒区间数组，如 [[5, 12]]；normal 时为空数组
- solution: 必须包含“对人的解决方案：”“对危险源的解决方案：”“整体的解决方案：”三段，每段单独一行；normal 时为空字符串

分类规则：normal 表示无异常且无明显风险征兆；risk_only 表示尚未发生异常但存在明显风险征兆；abnormal 表示异常事件已经发生。risk_only 使用“可能”“存在风险”等推测语气，abnormal 对已观察到的事件使用陈述语气。所有字段都必须输出，即使值为空。

输出格式：
{"risk_status":"normal","risk_type":"","risk_sources":[],"abnormal_actions":[],"affected_objects":[],"risk_description":"","consequence_understanding":"","causal_chain":"","causal_chain_text":"","event_summary":"","time_spans":[],"solution":""}"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-call Gemini inference over LifeBench real videos.")
    parser.add_argument("--videos-root", default=str(VIDEO_ROOT))
    parser.add_argument("--output-root", default=str(PREDICTION_ROOT))
    parser.add_argument("--run-dir", help="Existing run directory to resume, relative to --output-root or absolute.")
    parser.add_argument("--model", default="gemini-3.1-pro")
    parser.add_argument("--api-url", default="https://www.lingganyaapi.com/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API requests. Default: 1.")
    parser.add_argument("--limit", type=int, help="Process at most this many videos after sorting.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--include-skipped", action="store_true")
    return parser


def load_skip_keys() -> set[str]:
    if not SKIP_FILE.exists():
        return set()
    return {
        line.strip().removesuffix(".mp4")
        for line in SKIP_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_videos(root: Path, include_skipped: bool) -> list[Path]:
    videos = sorted(path for path in root.rglob("*.mp4") if path.is_file())
    if include_skipped:
        return videos
    skip_keys = load_skip_keys()
    return [path for path in videos if str(path.relative_to(root)).removesuffix(".mp4") not in skip_keys]


def parse_json_object(response: str) -> dict[str, Any] | None:
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.index("{") : cleaned.rindex("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_status(value: Any) -> str:
    aliases = {
        "正常": "normal",
        "无异常": "normal",
        "潜在异常": "risk_only",
        "风险未发生但可能发生": "risk_only",
        "已发生异常": "abnormal",
        "已经发生异常": "abnormal",
    }
    raw = str(value or "").strip().lower()
    return aliases.get(raw, raw)


def output_path(run_dir: Path, video_root: Path, video_path: Path, status: str) -> Path:
    relative = video_path.relative_to(video_root)
    if relative.parts and relative.parts[0] in {"normal", "risk_only", "abnormal"}:
        relative = Path(*relative.parts[1:])
    return run_dir / status / relative.with_suffix(".json")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def infer_one(
    video_path: Path,
    video_root: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    video_key = str(video_path.relative_to(video_root))
    messages = build_messages(video_path, SINGLE_PASS_PROMPT, args.max_frames)
    response = call_gemini(
        argparse.Namespace(
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
        messages,
    )
    parsed = parse_json_object(response)
    status = normalize_status(parsed.get("risk_status") if parsed else "")
    if status not in {"normal", "risk_only", "abnormal"}:
        raise ValueError(f"Response has invalid risk_status for {video_key}: {response[:300]!r}")

    payload = {
        "backend": "gemini",
        "model_id": args.model,
        "video_path": str(video_path.resolve()),
        "video_key": video_key,
        "response": response,
    }
    write_json(output_path(run_dir, video_root, video_path, status), payload)
    return video_key, status


def main() -> int:
    args = build_parser().parse_args()
    video_root = Path(args.videos_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Videos root does not exist: {video_root}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.max_frames < 1 or args.max_tokens < 1:
        raise ValueError("--max-frames and --max-tokens must be positive.")

    videos = load_videos(video_root, args.include_skipped)
    videos = videos[args.start_index :]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        videos = videos[: args.limit]
    if not videos:
        raise ValueError("No videos selected.")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.is_absolute():
            run_dir = output_root / run_dir
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / MODEL_DIR_NAME / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    pending: list[Path] = []
    for video_path in videos:
        relative = video_path.relative_to(video_root)
        if any((run_dir / status / relative.with_suffix(".json")).exists() for status in ("normal", "risk_only", "abnormal")):
            continue
        pending.append(video_path)

    print(f"Videos selected: {len(videos)}; pending: {len(pending)}", flush=True)
    print(f"Output directory: {run_dir}", flush=True)
    if not pending:
        return 0

    success = 0
    failed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(infer_one, path, video_root, run_dir, args): path for path in pending}
        for future in as_completed(futures):
            path = futures[future]
            try:
                video_key, status = future.result()
            except Exception as exc:
                failed += 1
                print(f"[error] {path.relative_to(video_root)}: {exc}", file=sys.stderr, flush=True)
            else:
                success += 1
                print(f"[{success}/{len(pending)}] {status}: {video_key}", flush=True)

    elapsed = time.time() - started
    print(f"Finished: success={success}, failed={failed}, elapsed_seconds={elapsed:.1f}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

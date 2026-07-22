#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from benchmark_models import BENCHMARK_MODELS, BY_MODEL_ID, BY_NAME, BenchmarkModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFER_SCRIPT = Path(__file__).resolve().parent / "lifebench_infer.py"
INFER_BATCH_SCRIPT = Path(__file__).resolve().parent / "lifebench_batch_infer.py"
EVALUATE_SCRIPT = Path(__file__).resolve().parent / "evaluate_outputs.py"
RENDER_SCRIPT = Path(__file__).resolve().parent / "render_final_results.py"
MODELS_ROOT = PROJECT_ROOT / "models"
PROJECT_TMP = PROJECT_ROOT / ".tmp"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def model_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


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


def model_dir(model_id: str) -> Path:
    candidates = [
        MODELS_ROOT / model_id.replace("/", "__"),
        MODELS_ROOT / model_id.replace("/", "_"),
        MODELS_ROOT.joinpath(*[part for part in model_id.split("/") if part]),
        MODELS_ROOT / model_id.split("/")[-1],
    ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        existing.sort(key=candidate_model_score, reverse=True)
        return existing[0]
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the integrated LifeBench models on a custom flat video dataset.")
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory containing .mp4 videos, either flat or nested.")
    parser.add_argument("--prompt-file", type=Path, required=True, help="Inference prompt file, usually infer_prompt_structured.txt.")
    parser.add_argument("--predictions-root", type=Path, required=True, help="Root directory for model predictions.")
    parser.add_argument("--ground-truth", type=Path, help="Ground-truth txt for evaluation.")
    parser.add_argument("--evaluation-output-dir", type=Path, help="Directory for evaluation artifacts.")
    parser.add_argument("--final-results-md", type=Path, help="Optional markdown output path for rendered final results.")
    parser.add_argument("--final-results-csv", type=Path, help="Optional CSV output path for rendered final results.")
    parser.add_argument("--status-root", type=Path, help="Status directory. Defaults to <predictions-root>/status.")
    parser.add_argument("--logs-root", type=Path, help="Log directory. Defaults to <predictions-root>/logs.")
    parser.add_argument("--run-id", default=default_run_id(), help="Run id shared by all models for this dataset.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional model names or model ids. Also supports ad-hoc commercial specs like commercial:gpt-5.4. Defaults to all integrated benchmark models.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated worker slots. Use GPU ids for local models, or cpu for commercial-only runs. Default: all 8 local A100s.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation length for inference.")
    parser.add_argument("--resume", action="store_true", help="Skip videos whose output JSON already exists and is valid.")
    parser.add_argument(
        "--replicas-per-model",
        type=int,
        default=1,
        help="Default number of parallel replicas per model for pending videos. Default: 1.",
    )
    parser.add_argument(
        "--model-replica-overrides",
        nargs="*",
        help='Optional per-model replica overrides like "Qwen/Qwen3.5-9B=2" or "VideoLLaMA3-7B=3".',
    )
    parser.add_argument(
        "--schedule-order",
        choices=("model-first", "video-first"),
        default="model-first",
        help="Task dispatch policy. 'video-first' prioritizes finishing all selected models for each video earlier.",
    )
    parser.add_argument(
        "--reuse-predictions-roots",
        nargs="*",
        type=Path,
        help="Optional existing predictions roots to reuse via symlink/copy into the current run before dispatch.",
    )
    parser.add_argument(
        "--reuse-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize reused predictions into the current run.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("auto", "heuristic", "openai"),
        default="auto",
        help="Judge mode forwarded to evaluate_outputs.py.",
    )
    parser.add_argument("--judge-model", help="Optional judge model name.")
    parser.add_argument("--judge-api-key", help="Optional judge API key.")
    parser.add_argument("--judge-base-url", help="Optional judge base URL.")
    return parser


def resolve_models(model_specs: list[str] | None) -> list[BenchmarkModel]:
    if not model_specs:
        return list(BENCHMARK_MODELS)

    resolved: list[BenchmarkModel] = []
    seen: set[str] = set()
    for item in model_specs:
        model = BY_NAME.get(item) or BY_MODEL_ID.get(item)
        if model is None and ":" in item:
            prefix, model_id = item.split(":", 1)
            if prefix in {"commercial", "closed"} and model_id.strip():
                cleaned_model_id = model_id.strip()
                model = BenchmarkModel(
                    name=f"Commercial::{cleaned_model_id}",
                    model_id=cleaned_model_id,
                    backend="commercial",
                    notes="Closed-source commercial model routed through llm.py.",
                )
        if model is None:
            raise KeyError(f"Unknown model: {item}")
        if model.model_id in seen:
            continue
        seen.add(model.model_id)
        resolved.append(model)
    return resolved


def default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        args.status_root or (args.predictions_root / "status"),
        args.logs_root or (args.predictions_root / "logs"),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_project_tmp_env(env: dict[str, str] | None = None) -> dict[str, str]:
    PROJECT_TMP.mkdir(parents=True, exist_ok=True)
    target = os.environ if env is None else env
    target["TMPDIR"] = str(PROJECT_TMP)
    target["TMP"] = str(PROJECT_TMP)
    target["TEMP"] = str(PROJECT_TMP)
    return target


def video_path_matches_video_id(video_path: str, video_id: str) -> bool:
    video_path_stem = Path(video_path).stem
    return video_path_stem == video_id or video_id.endswith(f"_{video_path_stem}")


def valid_output_json(path: Path, model_id: str, video_id: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("model_id") != model_id:
        return False
    video_path = payload.get("video_path")
    if not isinstance(video_path, str):
        return False
    return video_path_matches_video_id(video_path, video_id) and "response" in payload


def scan_videos(videos_dir: Path) -> list[Path]:
    return sorted(path for path in videos_dir.rglob("*.mp4") if path.is_file() or path.is_symlink())


def parse_replica_overrides(raw_overrides: list[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not raw_overrides:
        return overrides
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Replica override must be NAME=COUNT or MODEL_ID=COUNT, got: {item}")
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Replica override key is empty: {item}")
        try:
            count = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Replica override count must be an integer: {item}") from exc
        if count < 1:
            raise ValueError(f"Replica override count must be >= 1: {item}")
        overrides[key] = count
    return overrides


def replica_count_for_model(
    model: BenchmarkModel,
    default_replicas: int,
    override_map: dict[str, int],
) -> int:
    return max(
        1,
        int(
            override_map.get(
                model.model_id,
                override_map.get(model.name, default_replicas),
            )
        ),
    )


def inference_command(model: BenchmarkModel, video: Path, prompt_file: Path, output_json: Path, max_new_tokens: int) -> list[str]:
    return [
        sys.executable,
        str(INFER_SCRIPT),
        "--backend",
        model.backend or "",
        "--model-id",
        model.model_id,
        "--video-path",
        str(video),
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-json",
        str(output_json),
    ]


def batch_inference_command(
    model: BenchmarkModel,
    prompt_file: Path,
    manifest_path: Path,
    summary_path: Path,
    max_new_tokens: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(INFER_BATCH_SCRIPT),
        "--backend",
        model.backend or "",
        "--model-id",
        model.model_id,
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        str(max_new_tokens),
        "--task-manifest",
        str(manifest_path),
        "--summary-json",
        str(summary_path),
    ]
    try:
        import flash_attn  # noqa: F401
        cmd.append("--use-flash-attn")
    except ImportError:
        pass
    return cmd


def runtime_paths(
    model: BenchmarkModel,
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
) -> tuple[str, Path, Path, Path]:
    slug = model_slug(model.name)
    results_dir = predictions_root / slug / run_id
    model_logs_dir = logs_root / slug / run_id
    status_path = status_root / f"{slug}.json"
    return slug, results_dir, model_logs_dir, status_path


def build_base_payload(
    model: BenchmarkModel,
    run_id: str,
    results_dir: Path,
    logs_dir: Path,
    gpu_id: str,
    schedule_order: str,
) -> dict[str, Any]:
    return {
        "name": model.name,
        "model_id": model.model_id,
        "backend": model.backend,
        "model_dir": None if model.backend == "commercial" else str(model_dir(model.model_id)),
        "run_id": run_id,
        "results_dir": str(results_dir),
        "logs_dir": str(logs_dir),
        "gpu_id": gpu_id,
        "schedule_order": schedule_order,
        "updated_at": now_iso(),
    }


def build_inference_env(gpu_id: str) -> dict[str, str]:
    env = ensure_project_tmp_env(os.environ.copy())
    if gpu_id.lower() in {"cpu", "none", "-1"}:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONNOUSERSITE"] = "1"
    env_root = Path(sys.executable).resolve().parents[1]
    bin_dir = str(env_root / "bin")
    lib_dir = str(env_root / "lib")
    existing_path = env.get("PATH", "").strip()
    env["PATH"] = f"{bin_dir}:{existing_path}" if existing_path else bin_dir
    existing_ld = env.get("LD_LIBRARY_PATH", "").strip()
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing_ld}" if existing_ld else lib_dir
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HOME"] = str(PROJECT_ROOT / "models" / "hub")
    env["TRANSFORMERS_CACHE"] = str(PROJECT_ROOT / "models" / "hub")
    shim_lib = env_root / "lib" / "libittnotify.so"
    if shim_lib.exists():
        existing_preload = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = f"{shim_lib}:{existing_preload}" if existing_preload else str(shim_lib)
    return env


def skipped_existing_record(
    video: Path,
    output_json: Path,
    stdout_log: Path,
    stderr_log: Path,
    reuse_source: Path | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "video": str(video),
        "video_id": video.stem,
        "started_at": None,
        "finished_at": now_iso(),
        "returncode": 0,
        "skipped_existing": True,
        "output_json": str(output_json),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    if reuse_source is not None:
        record["reuse_source"] = str(reuse_source)
    return record


def should_skip_json_candidate(path: Path) -> bool:
    if any(part in {"status", "logs", ".auto_infer_state"} for part in path.parts):
        return True
    return path.name in {"summary.json", "evaluation_status.json", "state.json"}


def discover_reusable_predictions(
    roots: list[Path] | None,
    models: list[BenchmarkModel],
    video_ids: set[str],
) -> dict[tuple[str, str], Path]:
    if not roots:
        return {}

    selected_model_ids = {model.model_id for model in models}
    reusable: dict[tuple[str, str], tuple[float, Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if should_skip_json_candidate(path):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            model_id = payload.get("model_id")
            video_path = payload.get("video_path")
            if not isinstance(model_id, str) or model_id not in selected_model_ids:
                continue
            if not isinstance(video_path, str):
                continue
            candidate_video_id = path.stem if path.stem in video_ids else Path(video_path).stem
            if candidate_video_id not in video_ids:
                continue
            if not valid_output_json(path, model_id, candidate_video_id):
                continue
            key = (model_id, candidate_video_id)
            mtime = path.stat().st_mtime
            existing = reusable.get(key)
            if existing is None or mtime >= existing[0]:
                reusable[key] = (mtime, path)
    return {key: value[1] for key, value in reusable.items()}


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def materialize_reused_prediction(source: Path, target: Path, reuse_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        remove_path(target)
    if reuse_mode == "symlink":
        target.symlink_to(source.resolve())
        return
    shutil.copy2(source, target)


def seed_reused_predictions(
    models: list[BenchmarkModel],
    videos: list[Path],
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    reusable_predictions: dict[tuple[str, str], Path],
    reuse_mode: str,
) -> dict[tuple[str, str], Path]:
    seeded: dict[tuple[str, str], Path] = {}
    if not reusable_predictions:
        return seeded

    for model in models:
        _, results_dir, model_logs_dir, _ = runtime_paths(model, predictions_root, status_root, logs_root, run_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        model_logs_dir.mkdir(parents=True, exist_ok=True)
        for video in videos:
            key = (model.model_id, video.stem)
            source = reusable_predictions.get(key)
            if source is None:
                continue
            output_json = results_dir / f"{video.stem}.json"
            if valid_output_json(output_json, model.model_id, video.stem):
                seeded[key] = output_json
                continue
            materialize_reused_prediction(source, output_json, reuse_mode)
            seeded[key] = source
    return seeded


def build_running_payload(
    base_payload: dict[str, Any],
    total_videos: int,
    records: list[dict[str, Any]],
    failure_count: int = 0,
    final: bool = False,
    replica_count: int = 1,
    pending_videos: int | None = None,
) -> dict[str, Any]:
    if final:
        state = "failed" if failure_count else "completed"
    else:
        state = "running"
    return {
        **base_payload,
        "state": state,
        "total_videos": total_videos,
        "completed_videos": len(records),
        "pending_videos": max(0, total_videos - len(records)) if pending_videos is None else pending_videos,
        "replica_count": replica_count,
        "failed_tasks": failure_count,
        "per_video": records,
        "updated_at": now_iso(),
    }


def run_single_inference_task(
    model: BenchmarkModel,
    gpu_id: str,
    video: Path,
    prompt_file: Path,
    output_json: Path,
    stdout_log: Path,
    stderr_log: Path,
    max_new_tokens: int,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    if output_json.exists() or output_json.is_symlink():
        if not valid_output_json(output_json, model.model_id, video.stem):
            remove_path(output_json)

    command = inference_command(model, video, prompt_file, output_json, max_new_tokens)
    started_at = now_iso()
    with print_lock:
        print(
            f"[{started_at}] [gpu {gpu_id}] start {model.name} -> {video.name}",
            flush=True,
        )

    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=build_inference_env(gpu_id),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except Exception as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        returncode = 1

    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")

    finished_at = now_iso()
    with print_lock:
        print(
            f"[{finished_at}] [gpu {gpu_id}] done {model.name} -> {video.name} rc={returncode}",
            flush=True,
        )

    return {
        "video": str(video),
        "video_id": video.stem,
        "gpu_id": gpu_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": returncode,
        "output_json": str(output_json),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stdout_tail": stdout.strip()[-1000:],
        "stderr_tail": stderr.strip()[-1000:],
    }


def build_batch_manifest_entries(
    videos: list[Path],
    results_dir: Path,
    logs_dir: Path,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for video in videos:
        entries.append(
            {
                "video_path": str(video),
                "output_json": str(results_dir / f"{video.stem}.json"),
                "stdout_log": str(logs_dir / f"{video.stem}.stdout.log"),
                "stderr_log": str(logs_dir / f"{video.stem}.stderr.log"),
            }
        )
    return entries


def run_batch_inference_task(
    model: BenchmarkModel,
    gpu_id: str,
    videos: list[Path],
    prompt_file: Path,
    results_dir: Path,
    logs_dir: Path,
    max_new_tokens: int,
    replica_index: int,
    print_lock: threading.Lock,
) -> list[dict[str, Any]]:
    manifest_entries = build_batch_manifest_entries(videos, results_dir, logs_dir)
    manifest_path = logs_dir / f"replica_{replica_index}.manifest.json"
    summary_path = logs_dir / f"replica_{replica_index}.summary.json"
    batch_stdout_log = logs_dir / f"replica_{replica_index}.batch.stdout.log"
    batch_stderr_log = logs_dir / f"replica_{replica_index}.batch.stderr.log"
    for path in (summary_path, batch_stdout_log, batch_stderr_log):
        if path.exists() or path.is_symlink():
            remove_path(path)
    manifest_path.write_text(json.dumps(manifest_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    command = batch_inference_command(model, prompt_file, manifest_path, summary_path, max_new_tokens)
    started_at = now_iso()
    with print_lock:
        print(
            f"[{started_at}] [gpu {gpu_id}] replica {replica_index} start batch {model.name} videos={len(videos)}",
            flush=True,
        )

    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=build_inference_env(gpu_id),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except Exception as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        returncode = 1

    batch_stdout_log.write_text(stdout, encoding="utf-8")
    batch_stderr_log.write_text(stderr, encoding="utf-8")

    records: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                records = [record for record in payload["records"] if isinstance(record, dict)]
        except (OSError, json.JSONDecodeError):
            records = []

    recorded_ids = {str(record.get("video_id")) for record in records}
    for video in videos:
        if video.stem in recorded_ids:
            continue
        output_json = results_dir / f"{video.stem}.json"
        stdout_log = logs_dir / f"{video.stem}.stdout.log"
        stderr_log = logs_dir / f"{video.stem}.stderr.log"
        records.append(
            {
                "video": str(video),
                "video_id": video.stem,
                "gpu_id": gpu_id,
                "started_at": started_at,
                "finished_at": now_iso(),
                "returncode": 0 if returncode == 0 and valid_output_json(output_json, model.model_id, video.stem) else 1,
                "output_json": str(output_json),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stdout_tail": stdout.strip()[-1000:],
                "stderr_tail": stderr.strip()[-1000:],
            }
        )

    finished_at = now_iso()
    with print_lock:
        print(
            f"[{finished_at}] [gpu {gpu_id}] replica {replica_index} done batch {model.name} rc={returncode}",
            flush=True,
        )

    for record in records:
        record.setdefault("gpu_id", gpu_id)
    records.sort(key=lambda item: item.get("video_id", ""))
    return records


def initialize_video_first_model_states(
    models: list[BenchmarkModel],
    videos: list[Path],
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    resume: bool,
    seeded_predictions: dict[tuple[str, str], Path],
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for model in models:
        slug, results_dir, model_logs_dir, status_path = runtime_paths(
            model,
            predictions_root,
            status_root,
            logs_root,
            run_id,
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        model_logs_dir.mkdir(parents=True, exist_ok=True)
        base_payload = build_base_payload(
            model,
            run_id,
            results_dir,
            model_logs_dir,
            "dynamic",
            "video-first",
        )
        records: list[dict[str, Any]] = []
        completed_video_ids: set[str] = set()
        if resume:
            for video in videos:
                output_json = results_dir / f"{video.stem}.json"
                if not valid_output_json(output_json, model.model_id, video.stem):
                    continue
                stdout_log = model_logs_dir / f"{video.stem}.stdout.log"
                stderr_log = model_logs_dir / f"{video.stem}.stderr.log"
                records.append(
                    skipped_existing_record(
                        video=video,
                        output_json=output_json,
                        stdout_log=stdout_log,
                        stderr_log=stderr_log,
                        reuse_source=seeded_predictions.get((model.model_id, video.stem)),
                    )
                )
                completed_video_ids.add(video.stem)

        pending_videos = [video for video in videos if video.stem not in completed_video_ids]
        state = {
            "model": model,
            "slug": slug,
            "results_dir": results_dir,
            "logs_dir": model_logs_dir,
            "status_path": status_path,
            "base_payload": base_payload,
            "records": records,
            "completed_video_ids": completed_video_ids,
            "pending_videos": pending_videos,
            "failure_count": 0,
            "lock": threading.Lock(),
        }
        write_json(
            status_path,
            build_running_payload(
                base_payload,
                len(videos),
                records,
                replica_count=1,
                pending_videos=len(pending_videos),
            ),
        )
        states[model.model_id] = state
    return states


def build_video_first_tasks(
    models: list[BenchmarkModel],
    videos: list[Path],
    model_states: dict[str, dict[str, Any]],
) -> Queue[tuple[Path, BenchmarkModel]]:
    task_queue: Queue[tuple[Path, BenchmarkModel]] = Queue()
    for video in videos:
        for model in models:
            state = model_states[model.model_id]
            if video.stem in state["completed_video_ids"]:
                continue
            task_queue.put((video, model))
    return task_queue


def run_video_first_schedule(
    models: list[BenchmarkModel],
    videos: list[Path],
    gpu_ids: list[str],
    prompt_file: Path,
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    max_new_tokens: int,
    resume: bool,
    print_lock: threading.Lock,
    seeded_predictions: dict[tuple[str, str], Path],
) -> dict[str, Any]:
    model_states = initialize_video_first_model_states(
        models=models,
        videos=videos,
        predictions_root=predictions_root,
        status_root=status_root,
        logs_root=logs_root,
        run_id=run_id,
        resume=resume,
        seeded_predictions=seeded_predictions,
    )
    task_queue = build_video_first_tasks(models, videos, model_states)
    queued_task_count = task_queue.qsize()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                video, model = task_queue.get_nowait()
            except Empty:
                return
            try:
                state = model_states[model.model_id]
                output_json = state["results_dir"] / f"{video.stem}.json"
                stdout_log = state["logs_dir"] / f"{video.stem}.stdout.log"
                stderr_log = state["logs_dir"] / f"{video.stem}.stderr.log"
                record = run_single_inference_task(
                    model=model,
                    gpu_id=gpu_id,
                    video=video,
                    prompt_file=prompt_file,
                    output_json=output_json,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    max_new_tokens=max_new_tokens,
                    print_lock=print_lock,
                )
                with state["lock"]:
                    state["records"].append(record)
                    state["completed_video_ids"].add(video.stem)
                    if record["returncode"] != 0:
                        state["failure_count"] += 1
                    payload = build_running_payload(
                        state["base_payload"],
                        len(videos),
                        state["records"],
                        failure_count=state["failure_count"],
                        replica_count=1,
                        pending_videos=max(0, len(videos) - len(state["records"])),
                    )
                    write_json(state["status_path"], payload)
            finally:
                task_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, gpu_id) for gpu_id in gpu_ids]
        for future in as_completed(futures):
            future.result()

    model_results: list[dict[str, Any]] = []
    for state in model_states.values():
        payload = build_running_payload(
            state["base_payload"],
            len(videos),
            state["records"],
            failure_count=state["failure_count"],
            final=True,
            replica_count=1,
            pending_videos=max(0, len(videos) - len(state["records"])),
        )
        write_json(state["status_path"], payload)
        model_results.append(payload)

    return {
        "models": model_results,
        "queued_task_count": queued_task_count,
        "reused_prediction_count": len(seeded_predictions),
    }


def initialize_model_states(
    models: list[BenchmarkModel],
    videos: list[Path],
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    resume: bool,
    seeded_predictions: dict[tuple[str, str], Path],
    schedule_order: str,
    replica_resolver,
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for model in models:
        slug, results_dir, model_logs_dir, status_path = runtime_paths(
            model,
            predictions_root,
            status_root,
            logs_root,
            run_id,
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        model_logs_dir.mkdir(parents=True, exist_ok=True)
        base_payload = build_base_payload(
            model,
            run_id,
            results_dir,
            model_logs_dir,
            "dynamic",
            schedule_order,
        )
        records: list[dict[str, Any]] = []
        completed_video_ids: set[str] = set()
        if resume:
            for video in videos:
                output_json = results_dir / f"{video.stem}.json"
                if not valid_output_json(output_json, model.model_id, video.stem):
                    continue
                stdout_log = model_logs_dir / f"{video.stem}.stdout.log"
                stderr_log = model_logs_dir / f"{video.stem}.stderr.log"
                records.append(
                    skipped_existing_record(
                        video=video,
                        output_json=output_json,
                        stdout_log=stdout_log,
                        stderr_log=stderr_log,
                        reuse_source=seeded_predictions.get((model.model_id, video.stem)),
                    )
                )
                completed_video_ids.add(video.stem)
        pending_videos = [video for video in videos if video.stem not in completed_video_ids]
        replica_count = replica_resolver(model, pending_videos)
        state = {
            "model": model,
            "slug": slug,
            "results_dir": results_dir,
            "logs_dir": model_logs_dir,
            "status_path": status_path,
            "base_payload": base_payload,
            "records": records,
            "completed_video_ids": completed_video_ids,
            "pending_videos": pending_videos,
            "requested_replica_count": replica_count,
            "failure_count": 0,
            "lock": threading.Lock(),
        }
        write_json(
            status_path,
            build_running_payload(
                base_payload,
                len(videos),
                records,
                replica_count=replica_count,
                pending_videos=len(pending_videos),
            ),
        )
        states[model.model_id] = state
    return states


def split_round_robin(items: list[Path], parts: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(parts)]
    for index, item in enumerate(items):
        buckets[index % parts].append(item)
    return [bucket for bucket in buckets if bucket]


def build_model_replica_tasks(
    model_states: dict[str, dict[str, Any]],
    gpu_count: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for state in model_states.values():
        pending_videos: list[Path] = state["pending_videos"]
        if not pending_videos:
            continue
        replica_count = min(max(1, state["requested_replica_count"]), max(1, gpu_count), len(pending_videos))
        state["active_replica_count"] = replica_count
        for replica_index, chunk in enumerate(split_round_robin(pending_videos, replica_count), start=1):
            tasks.append(
                {
                    "model": state["model"],
                    "replica_index": replica_index,
                    "videos": chunk,
                    "video_count": len(chunk),
                }
            )
    tasks.sort(key=lambda item: (item["video_count"], item["model"].name), reverse=True)
    return tasks


def assign_replica_tasks_to_gpus(
    tasks: list[dict[str, Any]],
    gpu_ids: list[str],
) -> list[tuple[str, list[dict[str, Any]]]]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in gpu_ids]
    gpu_loads = [0 for _ in gpu_ids]
    for task in tasks:
        gpu_index = min(range(len(gpu_ids)), key=lambda index: gpu_loads[index])
        assignments[gpu_index].append(task)
        gpu_loads[gpu_index] += int(task["video_count"])
    return [(gpu_ids[index], bucket) for index, bucket in enumerate(assignments) if bucket]


def run_model_replica_schedule(
    models: list[BenchmarkModel],
    videos: list[Path],
    gpu_ids: list[str],
    prompt_file: Path,
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    max_new_tokens: int,
    resume: bool,
    print_lock: threading.Lock,
    seeded_predictions: dict[tuple[str, str], Path],
    default_replicas: int,
    override_map: dict[str, int],
) -> dict[str, Any]:
    def replica_resolver(model: BenchmarkModel, pending_videos: list[Path]) -> int:
        if not pending_videos:
            return 1
        return min(replica_count_for_model(model, default_replicas, override_map), len(pending_videos), len(gpu_ids))

    model_states = initialize_model_states(
        models=models,
        videos=videos,
        predictions_root=predictions_root,
        status_root=status_root,
        logs_root=logs_root,
        run_id=run_id,
        resume=resume,
        seeded_predictions=seeded_predictions,
        schedule_order="model-first",
        replica_resolver=replica_resolver,
    )
    tasks = build_model_replica_tasks(model_states, len(gpu_ids))
    queued_task_count = sum(task["video_count"] for task in tasks)
    assignments = assign_replica_tasks_to_gpus(tasks, gpu_ids)

    def worker(gpu_id: str, assigned_tasks: list[dict[str, Any]]) -> None:
        for task in assigned_tasks:
            model = task["model"]
            state = model_states[model.model_id]
            replica_index = task["replica_index"]
            records = run_batch_inference_task(
                model=model,
                gpu_id=gpu_id,
                videos=task["videos"],
                prompt_file=prompt_file,
                results_dir=state["results_dir"],
                logs_dir=state["logs_dir"],
                max_new_tokens=max_new_tokens,
                replica_index=replica_index,
                print_lock=print_lock,
            )
            with state["lock"]:
                for record in records:
                    state["records"].append(record)
                    state["completed_video_ids"].add(record["video_id"])
                    if record["returncode"] != 0:
                        state["failure_count"] += 1
                pending_count = max(0, len(videos) - len(state["records"]))
                write_json(
                    state["status_path"],
                    build_running_payload(
                        state["base_payload"],
                        len(videos),
                        state["records"],
                        failure_count=state["failure_count"],
                        replica_count=state.get("active_replica_count", 1),
                        pending_videos=pending_count,
                    ),
                )

    with ThreadPoolExecutor(max_workers=len(assignments) or 1) as executor:
        futures = [executor.submit(worker, gpu_id, assigned_tasks) for gpu_id, assigned_tasks in assignments]
        for future in as_completed(futures):
            future.result()

    model_results: list[dict[str, Any]] = []
    for state in model_states.values():
        payload = build_running_payload(
            state["base_payload"],
            len(videos),
            state["records"],
            failure_count=state["failure_count"],
            final=True,
            replica_count=state.get("active_replica_count", state["requested_replica_count"]),
            pending_videos=max(0, len(videos) - len(state["records"])),
        )
        write_json(state["status_path"], payload)
        model_results.append(payload)

    return {
        "models": model_results,
        "queued_task_count": queued_task_count,
        "reused_prediction_count": len(seeded_predictions),
        "task_count": len(tasks),
        "assignments": {
            gpu_id: [
                {
                    "model_id": task["model"].model_id,
                    "replica_index": task["replica_index"],
                    "video_count": task["video_count"],
                }
                for task in assigned_tasks
            ]
            for gpu_id, assigned_tasks in assignments
        },
    }


def process_model(
    model: BenchmarkModel,
    gpu_id: str,
    videos: list[Path],
    prompt_file: Path,
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    max_new_tokens: int,
    resume: bool,
    print_lock: threading.Lock,
) -> dict[str, Any]:
    slug, results_dir, model_logs_dir, status_path = runtime_paths(
        model,
        predictions_root,
        status_root,
        logs_root,
        run_id,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    model_logs_dir.mkdir(parents=True, exist_ok=True)

    base_payload = build_base_payload(model, run_id, results_dir, model_logs_dir, gpu_id, "model-first")
    running_payload = {
        **base_payload,
        "state": "running",
        "total_videos": len(videos),
        "completed_videos": 0,
        "per_video": [],
    }
    write_json(status_path, running_payload)

    per_video: list[dict[str, Any]] = []
    env = build_inference_env(gpu_id)

    for index, video in enumerate(videos, start=1):
        video_id = video.stem
        output_json = results_dir / f"{video_id}.json"
        stdout_log = model_logs_dir / f"{video_id}.stdout.log"
        stderr_log = model_logs_dir / f"{video_id}.stderr.log"

        if resume and valid_output_json(output_json, model.model_id, video_id):
            record = {
                "video": str(video),
                "video_id": video_id,
                "started_at": None,
                "finished_at": now_iso(),
                "returncode": 0,
                "skipped_existing": True,
                "output_json": str(output_json),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
            per_video.append(record)
            running_payload.update(
                {
                    "updated_at": now_iso(),
                    "completed_videos": len(per_video),
                    "per_video": per_video,
                }
            )
            write_json(status_path, running_payload)
            continue

        command = inference_command(model, video, prompt_file, output_json, max_new_tokens)
        started_at = now_iso()
        with print_lock:
            print(
                f"[{started_at}] [gpu {gpu_id}] start {model.name} {index}/{len(videos)} -> {video.name}",
                flush=True,
            )

        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")

        record = {
            "video": str(video),
            "video_id": video_id,
            "started_at": started_at,
            "finished_at": now_iso(),
            "returncode": completed.returncode,
            "output_json": str(output_json),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_tail": completed.stdout.strip()[-1000:],
            "stderr_tail": completed.stderr.strip()[-1000:],
        }
        per_video.append(record)

        running_payload.update(
            {
                "updated_at": now_iso(),
                "completed_videos": len(per_video),
                "per_video": per_video,
            }
        )
        write_json(status_path, running_payload)

        finished_at = record["finished_at"]
        with print_lock:
            print(
                f"[{finished_at}] [gpu {gpu_id}] done {model.name} {index}/{len(videos)} -> {video.name} rc={completed.returncode}",
                flush=True,
            )

        if completed.returncode != 0:
            payload = {
                **base_payload,
                "state": "failed",
                "updated_at": now_iso(),
                "completed_videos": len(per_video),
                "total_videos": len(videos),
                "per_video": per_video,
            }
            write_json(status_path, payload)
            return payload

    payload = {
        **base_payload,
        "state": "completed",
        "updated_at": now_iso(),
        "completed_videos": len(per_video),
        "total_videos": len(videos),
        "per_video": per_video,
    }
    write_json(status_path, payload)
    return payload


def run_evaluation(
    ground_truth: Path,
    predictions_root: Path,
    evaluation_output_dir: Path,
    judge_mode: str,
    judge_model: str | None,
    judge_api_key: str | None,
    judge_base_url: str | None,
    expected_model_ids: list[str] | None,
    resume: bool,
) -> dict[str, Any]:
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--ground-truth",
        str(ground_truth),
        "--predictions-root",
        str(predictions_root),
        "--output-dir",
        str(evaluation_output_dir),
        "--judge-mode",
        judge_mode,
        "--require-complete-videos",
    ]
    if expected_model_ids:
        command.append("--expected-models")
        command.extend(expected_model_ids)
    if judge_model:
        command.extend(["--judge-model", judge_model])
    if judge_api_key:
        command.extend(["--judge-api-key", judge_api_key])
    if judge_base_url:
        command.extend(["--judge-base-url", judge_base_url])
    if resume:
        command.append("--resume")

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    summary_json = evaluation_output_dir / "summary.json"
    payload: dict[str, Any] = {
        "returncode": completed.returncode,
        "command": command,
        "output_dir": str(evaluation_output_dir),
        "stdout_tail": completed.stdout.strip()[-2000:],
        "stderr_tail": completed.stderr.strip()[-2000:],
        "updated_at": now_iso(),
    }
    if summary_json.exists():
        try:
            payload["summary"] = json.loads(summary_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return payload


def render_final_results(
    summary_json: Path,
    status_root: Path,
    csv_out: Path | None,
    markdown_out: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--summary-json",
        str(summary_json),
        "--status-dir",
        str(status_root),
    ]
    if csv_out:
        command.extend(["--csv-out", str(csv_out)])
    if markdown_out:
        command.extend(["--markdown-out", str(markdown_out)])

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    payload = {
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout.strip()[-2000:],
        "stderr_tail": completed.stderr.strip()[-2000:],
        "updated_at": now_iso(),
    }
    try:
        parsed = json.loads(completed.stdout)
        if isinstance(parsed, dict):
            payload.update(parsed)
    except json.JSONDecodeError:
        pass
    return payload


def parse_gpu_ids(gpus: str) -> list[str]:
    return [item.strip() for item in gpus.split(",") if item.strip()]


def worker_assignments(models: list[BenchmarkModel], gpu_ids: list[str]) -> list[tuple[str, list[BenchmarkModel]]]:
    assignments: list[list[BenchmarkModel]] = [[] for _ in gpu_ids]
    for index, model in enumerate(models):
        assignments[index % len(gpu_ids)].append(model)
    return [(gpu_ids[index], bucket) for index, bucket in enumerate(assignments) if bucket]


def worker_run(
    gpu_id: str,
    models: list[BenchmarkModel],
    videos: list[Path],
    prompt_file: Path,
    predictions_root: Path,
    status_root: Path,
    logs_root: Path,
    run_id: str,
    max_new_tokens: int,
    resume: bool,
    print_lock: threading.Lock,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model in models:
        results.append(
            process_model(
                model=model,
                gpu_id=gpu_id,
                videos=videos,
                prompt_file=prompt_file,
                predictions_root=predictions_root,
                status_root=status_root,
                logs_root=logs_root,
                run_id=run_id,
                max_new_tokens=max_new_tokens,
                resume=resume,
                print_lock=print_lock,
            )
        )
    return results


def main() -> int:
    ensure_project_tmp_env()
    args = build_parser().parse_args()
    models = resolve_models(args.models)
    requested_model_ids = [model.model_id for model in models]
    if args.replicas_per_model < 1:
        raise ValueError("--replicas-per-model must be >= 1.")
    replica_overrides = parse_replica_overrides(args.model_replica_overrides)
    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")

    videos = scan_videos(args.videos_dir)
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found in {args.videos_dir}")
    if not args.prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    status_root, logs_root = default_paths(args)
    args.predictions_root.mkdir(parents=True, exist_ok=True)
    status_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    print_lock = threading.Lock()
    video_ids = {video.stem for video in videos}
    reusable_predictions = discover_reusable_predictions(args.reuse_predictions_roots, models, video_ids)
    seeded_predictions = seed_reused_predictions(
        models=models,
        videos=videos,
        predictions_root=args.predictions_root,
        status_root=status_root,
        logs_root=logs_root,
        run_id=args.run_id,
        reusable_predictions=reusable_predictions,
        reuse_mode=args.reuse_mode,
    )

    if args.resume:
        filtered_models: list[BenchmarkModel] = []
        skipped_model_ids: list[str] = []
        for model in models:
            _, results_dir, _, _ = runtime_paths(
                model,
                args.predictions_root,
                status_root,
                logs_root,
                args.run_id,
            )
            pending_found = False
            for video in videos:
                output_json = results_dir / f"{video.stem}.json"
                if not valid_output_json(output_json, model.model_id, video.stem):
                    pending_found = True
                    break
            if pending_found:
                filtered_models.append(model)
            else:
                skipped_model_ids.append(model.model_id)
        models = filtered_models
    else:
        skipped_model_ids = []

    assignments = worker_assignments(models, gpu_ids) if models else []
    started_at = now_iso()
    with print_lock:
        print(
            json.dumps(
                {
                    "started_at": started_at,
                    "run_id": args.run_id,
                    "video_count": len(videos),
                    "model_count": len(models),
                    "requested_model_count": len(requested_model_ids),
                    "schedule_order": args.schedule_order,
                    "gpu_ids": gpu_ids,
                    "reuse_predictions_roots": [str(path) for path in args.reuse_predictions_roots or []],
                    "reused_prediction_candidates": len(reusable_predictions),
                    "seeded_predictions": len(seeded_predictions),
                    "skipped_completed_models": skipped_model_ids,
                    "replicas_per_model": args.replicas_per_model,
                    "model_replica_overrides": replica_overrides,
                    "assignments": {gpu_id: [model.model_id for model in bucket] for gpu_id, bucket in assignments},
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    model_results: list[dict[str, Any]] = []
    queue_stats: dict[str, Any] | None = None
    if not models:
        queue_stats = {
            "queued_task_count": 0,
            "reused_prediction_count": len(seeded_predictions),
            "task_count": 0,
            "assignments": {},
        }
    elif args.schedule_order == "video-first":
        queue_stats = run_video_first_schedule(
            models=models,
            videos=videos,
            gpu_ids=gpu_ids,
            prompt_file=args.prompt_file,
            predictions_root=args.predictions_root,
            status_root=status_root,
            logs_root=logs_root,
            run_id=args.run_id,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            print_lock=print_lock,
            seeded_predictions=seeded_predictions,
        )
        model_results.extend(queue_stats["models"])
    else:
        queue_stats = run_model_replica_schedule(
            models=models,
            videos=videos,
            gpu_ids=gpu_ids,
            prompt_file=args.prompt_file,
            predictions_root=args.predictions_root,
            status_root=status_root,
            logs_root=logs_root,
            run_id=args.run_id,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            print_lock=print_lock,
            seeded_predictions=seeded_predictions,
            default_replicas=args.replicas_per_model,
            override_map=replica_overrides,
        )
        model_results.extend(queue_stats["models"])

    summary_payload: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "run_id": args.run_id,
        "video_count": len(videos),
        "videos_dir": str(args.videos_dir),
        "prompt_file": str(args.prompt_file),
        "requested_model_ids": requested_model_ids,
        "predictions_root": str(args.predictions_root),
        "status_root": str(status_root),
        "logs_root": str(logs_root),
        "schedule_order": args.schedule_order,
        "reuse_predictions_roots": [str(path) for path in args.reuse_predictions_roots or []],
        "reused_prediction_candidates": len(reusable_predictions),
        "seeded_predictions": len(seeded_predictions),
        "skipped_completed_models": skipped_model_ids,
        "replicas_per_model": args.replicas_per_model,
        "model_replica_overrides": replica_overrides,
        "models": model_results,
    }
    if queue_stats is not None:
        summary_payload["queue_stats"] = {
            "queued_task_count": queue_stats["queued_task_count"],
            "reused_prediction_count": queue_stats["reused_prediction_count"],
            "task_count": queue_stats.get("task_count"),
            "assignments": queue_stats.get("assignments"),
        }

    if args.ground_truth and args.evaluation_output_dir:
        evaluation_status = run_evaluation(
            ground_truth=args.ground_truth,
            predictions_root=args.predictions_root,
            evaluation_output_dir=args.evaluation_output_dir,
            judge_mode=args.judge_mode,
            judge_model=args.judge_model,
            judge_api_key=args.judge_api_key,
            judge_base_url=args.judge_base_url,
            expected_model_ids=requested_model_ids,
            resume=args.resume,
        )
        summary_payload["evaluation"] = evaluation_status

        if evaluation_status.get("returncode") == 0:
            render_status = render_final_results(
                summary_json=args.evaluation_output_dir / "summary.json",
                status_root=status_root,
                csv_out=args.final_results_csv,
                markdown_out=args.final_results_md,
            )
            summary_payload["final_results"] = render_status

    write_json(args.predictions_root / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

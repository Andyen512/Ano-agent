#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_view_risk_review import parse_keep_decision, summarize_results
from persistent_batch_multi_view_risk_review import parse_gpu_ids, worker_main


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_outputs" / "persistent_full"
TARGET_PERSPECTIVE = "Conservative Safety Perspective"
REPAIR_BACKEND = "minicpmv45"
REPAIR_MODEL_ID = "openbmb/MiniCPM-V-4_5"
REPAIR_SUFFIX = "minicpmv45"


@dataclass(frozen=True)
class RepairTask:
    key: str
    index: int
    total: int
    summary_path: Path
    repaired_summary_path: Path
    video_path: Path
    video_relpath: str
    prompt_path: Path
    output_json: Path
    stdout_log: Path
    stderr_log: Path
    assignment_index: int
    original_payload: dict[str, Any]


@dataclass
class WorkerSlot:
    worker_id: str
    gpu_id: str
    input_queue: mp.Queue
    process: mp.Process
    inflight: int = 0


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only the Tarsier2-backed Conservative Safety Perspective with "
            "MiniCPM-V-4.5 and write repaired summaries without overwriting originals."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse valid MiniCPM repair outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun even when a valid MiniCPM repair exists.")
    parser.add_argument("--max-inflight-videos", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generation_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "merge_size": args.merge_size,
    }


def is_target_summary(payload: dict[str, Any]) -> bool:
    majority = payload.get("majority_vote", {})
    if majority.get("complete") is True:
        return False
    undecidable_agents = majority.get("undecidable_agents") or []
    return undecidable_agents == [TARGET_PERSPECTIVE]


def find_target_agent(payload: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    for index, item in enumerate(payload.get("agent_results", [])):
        perspective = item.get("perspective", {})
        if perspective.get("name") == TARGET_PERSPECTIVE:
            return index, item
    return None


def repaired_output_paths(summary_path: Path, original_agent: dict[str, Any]) -> tuple[Path, Path, Path]:
    original_output = Path(original_agent["output_json"])
    stem = original_output.stem
    output_json = original_output.with_name(f"{stem}.{REPAIR_SUFFIX}.json")
    logs_dir = summary_path.parent / "logs"
    stdout_log = logs_dir / f"{stem}.{REPAIR_SUFFIX}.stdout.log"
    stderr_log = logs_dir / f"{stem}.{REPAIR_SUFFIX}.stderr.log"
    return output_json, stdout_log, stderr_log


def load_valid_repair(args: argparse.Namespace, task: RepairTask) -> dict[str, Any] | None:
    if not task.output_json.exists():
        return None
    try:
        payload = read_json(task.output_json)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("backend") != REPAIR_BACKEND:
        return None
    if payload.get("model_id") != REPAIR_MODEL_ID:
        return None
    if payload.get("video_path") != str(task.video_path):
        return None
    if str(payload.get("prompt", "")).strip() != task.prompt_path.read_text(encoding="utf-8").strip():
        return None
    generation = payload.get("generation")
    if not isinstance(generation, dict):
        return None
    for key, value in generation_payload(args).items():
        if generation.get(key) != value:
            return None
    response = str(payload.get("response", "")).strip()
    parsed_decision = parse_keep_decision(response)
    if parsed_decision.get("keep") is None:
        return None
    return build_agent_result_from_payload(task, payload, parsed_decision, reused=True)


def build_agent_result_from_payload(
    task: RepairTask,
    payload: dict[str, Any],
    parsed_decision: dict[str, Any],
    reused: bool,
) -> dict[str, Any]:
    original_agent = task.original_payload["agent_results"][task.assignment_index]
    result = {
        "perspective": original_agent["perspective"],
        "model": {
            "model_id": REPAIR_MODEL_ID,
            "backend": REPAIR_BACKEND,
        },
        "prompt_file": str(task.prompt_path),
        "output_json": str(task.output_json),
        "stdout_log": str(task.stdout_log),
        "stderr_log": str(task.stderr_log),
        "worker_id": None,
        "gpu_id": None,
        "returncode": 0,
        "elapsed_seconds": 0.0,
        "response": str(payload.get("response", "")).strip(),
        "parsed_decision": parsed_decision,
        "runner_payload": payload,
    }
    if reused:
        result["reused_agent_output"] = True
    return result


def build_agent_result_from_message(task: RepairTask, message: dict[str, Any]) -> dict[str, Any]:
    original_agent = task.original_payload["agent_results"][task.assignment_index]
    result = {
        "perspective": original_agent["perspective"],
        "model": {
            "model_id": REPAIR_MODEL_ID,
            "backend": REPAIR_BACKEND,
        },
        "prompt_file": str(task.prompt_path),
        "output_json": str(task.output_json),
        "stdout_log": str(task.stdout_log),
        "stderr_log": str(task.stderr_log),
        "worker_id": message.get("worker_id"),
        "gpu_id": message.get("gpu_id"),
        "returncode": message["returncode"],
        "elapsed_seconds": message["elapsed_seconds"],
    }
    if message["returncode"] != 0:
        result["error"] = {
            "message": message.get("error", "MiniCPM repair inference failed."),
            "traceback_tail": message.get("traceback", ""),
        }
        return result
    payload = message["runner_payload"]
    response = str(payload.get("response", "")).strip()
    result["response"] = response
    result["parsed_decision"] = parse_keep_decision(response)
    result["runner_payload"] = payload
    return result


def build_repaired_summary(task: RepairTask, replacement_agent: dict[str, Any]) -> dict[str, Any]:
    repaired = copy.deepcopy(task.original_payload)
    original_agent = repaired["agent_results"][task.assignment_index]
    repaired["agent_results"][task.assignment_index] = replacement_agent
    repaired["majority_vote"] = summarize_results(
        repaired["agent_results"],
        expected_votes=len(repaired["agent_results"]),
    )
    repaired["repair"] = {
        "repaired_at": utc_tag(),
        "method": "replace_conservative_safety_perspective",
        "source_summary_json": str(task.summary_path),
        "replacement_backend": REPAIR_BACKEND,
        "replacement_model_id": REPAIR_MODEL_ID,
        "replacement_output_json": str(task.output_json),
        "original_backend": original_agent.get("model", {}).get("backend"),
        "original_model_id": original_agent.get("model", {}).get("model_id"),
        "original_output_json": original_agent.get("output_json"),
    }
    return repaired


def row_from_payload(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("manifest", {})
    majority = payload.get("majority_vote", {})
    return {
        "video_path": manifest.get("video_path", ""),
        "video_relpath": video_relpath_from_summary(summary_path, manifest.get("video_path", "")),
        "majority_vote": majority,
        "elapsed_seconds": max(
            [float(item.get("elapsed_seconds") or 0.0) for item in payload.get("agent_results", [])] or [0.0]
        ),
        "summary_json": str(summary_path),
    }


def video_relpath_from_summary(summary_path: Path, video_path: str) -> str:
    if video_path:
        try:
            return str(Path(video_path).relative_to(PROJECT_ROOT / "data" / "public_data"))
        except ValueError:
            return video_path
    parts = summary_path.parts
    if "reviews" in parts:
        index = parts.index("reviews")
        return str(Path(*parts[index + 1:]).with_suffix(".mp4"))
    return str(summary_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_relpath",
                "video_path",
                "complete",
                "keep",
                "risk_presence",
                "yes_count",
                "no_count",
                "votes_cast",
                "level1_scene_top",
                "level2_subject_top",
                "level3_risk_type_top",
                "normal_level1_scene_top",
                "normal_level2_subject_top",
                "elapsed_seconds",
                "summary_json",
            ],
        )
        writer.writeheader()
        for row in rows:
            majority = row.get("majority_vote", {})
            writer.writerow(
                {
                    "video_relpath": row["video_relpath"],
                    "video_path": row["video_path"],
                    "complete": majority.get("complete"),
                    "keep": majority.get("keep"),
                    "risk_presence": majority.get("risk_presence"),
                    "yes_count": majority.get("yes_count"),
                    "no_count": majority.get("no_count"),
                    "votes_cast": majority.get("votes_cast"),
                    "level1_scene_top": majority.get("level1_scene_top"),
                    "level2_subject_top": majority.get("level2_subject_top"),
                    "level3_risk_type_top": majority.get("level3_risk_type_top"),
                    "normal_level1_scene_top": majority.get("normal_level1_scene_top"),
                    "normal_level2_subject_top": majority.get("normal_level2_subject_top"),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "summary_json": row["summary_json"],
                }
            )


def summarize_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        majority = row.get("majority_vote", {})
        if majority.get("complete") is not True:
            counts["incomplete"] += 1
        elif majority.get("keep") is True:
            counts["keep"] += 1
        elif majority.get("keep") is False:
            counts["drop"] += 1
        else:
            counts["unknown"] += 1
    return {"processed_videos": len(rows), "counts": dict(counts)}


def discover_tasks(args: argparse.Namespace) -> list[RepairTask]:
    output_root = args.output_root.expanduser().resolve()
    summary_paths = sorted((output_root / "reviews").rglob("review_summary.json"))
    tasks: list[RepairTask] = []
    for summary_path in summary_paths:
        payload = read_json(summary_path)
        if not is_target_summary(payload):
            continue
        agent_pair = find_target_agent(payload)
        if agent_pair is None:
            continue
        assignment_index, original_agent = agent_pair
        prompt_path = Path(original_agent["prompt_file"])
        if not prompt_path.exists():
            prompt_path = summary_path.parent / "prompts" / f"{Path(original_agent['output_json']).stem}.txt"
        output_json, stdout_log, stderr_log = repaired_output_paths(summary_path, original_agent)
        video_path = Path(payload.get("manifest", {}).get("video_path", ""))
        if not video_path.exists():
            continue
        tasks.append(
            RepairTask(
                key=str(summary_path),
                index=len(tasks) + 1,
                total=0,
                summary_path=summary_path,
                repaired_summary_path=summary_path.with_name("review_summary.repaired.json"),
                video_path=video_path,
                video_relpath=video_relpath_from_summary(summary_path, str(video_path)),
                prompt_path=prompt_path,
                output_json=output_json,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                assignment_index=assignment_index,
                original_payload=payload,
            )
        )
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]
    total = len(tasks)
    return [
        RepairTask(
            key=task.key,
            index=index,
            total=total,
            summary_path=task.summary_path,
            repaired_summary_path=task.repaired_summary_path,
            video_path=task.video_path,
            video_relpath=task.video_relpath,
            prompt_path=task.prompt_path,
            output_json=task.output_json,
            stdout_log=task.stdout_log,
            stderr_log=task.stderr_log,
            assignment_index=task.assignment_index,
            original_payload=task.original_payload,
        )
        for index, task in enumerate(tasks, start=1)
    ]


def job_from_task(args: argparse.Namespace, task: RepairTask) -> dict[str, Any]:
    return {
        "task_key": task.key,
        "assignment_index": task.assignment_index,
        "video_path": str(task.video_path),
        "prompt": task.prompt_path.read_text(encoding="utf-8").strip(),
        "prompt_file": str(task.prompt_path),
        "output_json": str(task.output_json),
        "stdout_log": str(task.stdout_log),
        "stderr_log": str(task.stderr_log),
        "model_id": REPAIR_MODEL_ID,
        "backend": REPAIR_BACKEND,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "merge_size": args.merge_size,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "use_flash_attn": args.use_flash_attn,
        "tarsier_config": "",
    }


def write_repaired_result(task: RepairTask, replacement_agent: dict[str, Any]) -> dict[str, Any]:
    repaired = build_repaired_summary(task, replacement_agent)
    write_json(task.repaired_summary_path, repaired)
    return row_from_payload(task.repaired_summary_path, repaired)


def final_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((output_root / "reviews").rglob("review_summary.json")):
        repaired_path = summary_path.with_name("review_summary.repaired.json")
        effective_path = repaired_path if repaired_path.exists() else summary_path
        rows.append(row_from_payload(effective_path, read_json(effective_path)))
    rows.sort(key=lambda item: item["video_relpath"])
    return rows


def write_batch_outputs(output_root: Path, tasks: list[RepairTask], rows: list[dict[str, Any]]) -> None:
    payload = {
        "repair_manifest": {
            "created_at": utc_tag(),
            "source_output_root": str(output_root),
            "target_perspective": TARGET_PERSPECTIVE,
            "replacement_backend": REPAIR_BACKEND,
            "replacement_model_id": REPAIR_MODEL_ID,
            "target_count": len(tasks),
        },
        "summary": summarize_batch(rows),
        "results": rows,
    }
    write_json(output_root / "batch_review_summary.minicpmv45_repaired.json", payload)
    write_csv(output_root / "batch_review_summary.minicpmv45_repaired.csv", rows)


def configure_worker_environment() -> None:
    if not os.environ.get("CUDA_HOME"):
        try:
            from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
        except Exception:
            TORCH_CUDA_HOME = None
        os.environ["CUDA_HOME"] = str(TORCH_CUDA_HOME or "/usr")
    os.environ.setdefault("CUDA_PATH", os.environ["CUDA_HOME"])

    # MiniCPM itself does not need DeepSpeed custom ops here. The swift
    # environment imports DeepSpeed during startup, and disabling optional op
    # builds avoids per-worker CUDA/AIO compile probes.
    for name in (
        "DS_BUILD_OPS",
        "DS_BUILD_AIO",
        "DS_BUILD_GDS",
        "DS_BUILD_CUTLASS_OPS",
        "DS_BUILD_TRANSFORMER",
        "DS_BUILD_TRANSFORMER_INFERENCE",
        "DS_BUILD_STOCHASTIC_TRANSFORMER",
        "DS_BUILD_SPARSE_ATTN",
        "DS_BUILD_QUANTIZER",
        "DS_BUILD_RAGGED_DEVICE_OPS",
        "DS_BUILD_RAGGED_OPS",
        "DS_BUILD_INFERENCE_CORE_OPS",
        "DS_BUILD_CPU_ADAM",
        "DS_BUILD_FUSED_ADAM",
        "DS_BUILD_FUSED_LAMB",
        "DS_BUILD_FUSED_LION",
        "DS_BUILD_CPU_LION",
        "DS_BUILD_CPU_ADAGRAD",
    ):
        os.environ.setdefault(name, "0")


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    tasks = discover_tasks(args)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "target_count": len(tasks),
                "backend": REPAIR_BACKEND,
                "model_id": REPAIR_MODEL_ID,
                "gpus": parse_gpu_ids(args.gpus),
                "resume": bool(args.resume),
                "overwrite": bool(args.overwrite),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    configure_worker_environment()
    gpu_ids = parse_gpu_ids(args.gpus)
    ctx = mp.get_context("spawn")
    output_queue: mp.Queue = ctx.Queue()
    args_payload = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "merge_size": args.merge_size,
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map,
        "use_flash_attn": args.use_flash_attn,
        "tarsier_config": "",
    }

    worker_logs = output_root / "worker_logs_minicpmv45_repair"
    worker_logs.mkdir(parents=True, exist_ok=True)
    workers: list[WorkerSlot] = []
    for index, gpu_id in enumerate(gpu_ids, start=1):
        input_queue: mp.Queue = ctx.Queue()
        worker_id = f"{index:02d}_{REPAIR_BACKEND}_gpu{gpu_id}"
        process = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_id,
                REPAIR_BACKEND,
                REPAIR_MODEL_ID,
                args_payload,
                input_queue,
                output_queue,
                str(worker_logs / f"{worker_id}.log"),
            ),
        )
        process.start()
        workers.append(WorkerSlot(worker_id=worker_id, gpu_id=gpu_id, input_queue=input_queue, process=process))

    ready_workers = 0
    while ready_workers < len(workers):
        message = output_queue.get()
        if message.get("type") == "ready":
            ready_workers += 1
            print(
                f"[worker-ready] {message['worker_id']} backend={message['backend']} gpu={message['gpu_id']}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if message.get("type") == "worker_error":
            raise RuntimeError(
                f"Worker {message['worker_id']} failed during startup: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )

    pending = deque(tasks)
    active: dict[str, RepairTask] = {}
    completed_count = 0
    reused_count = 0
    failed_count = 0

    def choose_worker() -> WorkerSlot:
        return min(workers, key=lambda worker: worker.inflight)

    def schedule_more() -> None:
        nonlocal completed_count, reused_count
        while pending and len(active) < max(1, args.max_inflight_videos):
            task = pending.popleft()
            if args.resume and not args.overwrite:
                reused = load_valid_repair(args, task)
                if reused is not None:
                    write_repaired_result(task, reused)
                    completed_count += 1
                    reused_count += 1
                    if args.log_every > 0 and (completed_count % args.log_every == 0 or completed_count == len(tasks)):
                        print(
                            f"[{completed_count}/{len(tasks)}] reused {task.video_relpath}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
            worker = choose_worker()
            worker.inflight += 1
            active[task.key] = task
            worker.input_queue.put(job_from_task(args, task))

    try:
        schedule_more()
        while active:
            try:
                message = output_queue.get(timeout=1.0)
            except queue.Empty:
                dead = [worker.worker_id for worker in workers if not worker.process.is_alive()]
                if dead:
                    raise RuntimeError(f"Worker process exited unexpectedly: {dead}")
                schedule_more()
                continue
            if message.get("type") == "worker_error":
                raise RuntimeError(
                    f"Worker {message['worker_id']} failed: {message.get('error')}\n{message.get('traceback', '')}"
                )
            if message.get("type") != "result":
                continue
            for worker in workers:
                if worker.worker_id == message.get("worker_id"):
                    worker.inflight = max(0, worker.inflight - 1)
                    break
            task = active.pop(message["task_key"], None)
            if task is None:
                schedule_more()
                continue
            replacement = build_agent_result_from_message(task, message)
            if replacement.get("returncode") != 0:
                failed_count += 1
            row = write_repaired_result(task, replacement)
            completed_count += 1
            if args.log_every > 0 and (completed_count % args.log_every == 0 or completed_count == len(tasks)):
                majority = row.get("majority_vote", {})
                status = "keep" if majority.get("keep") is True else "drop" if majority.get("keep") is False else "unknown"
                print(
                    f"[{completed_count}/{len(tasks)}] final_status={status} votes={majority.get('votes_cast')} {task.video_relpath}",
                    file=sys.stderr,
                    flush=True,
                )
            schedule_more()
    finally:
        for worker in workers:
            worker.input_queue.put(None)
        for worker in workers:
            worker.process.join(timeout=30)
            if worker.process.is_alive():
                worker.process.terminate()

    rows = final_rows(output_root)
    write_batch_outputs(output_root, tasks, rows)
    summary = {
        "processed_repairs": completed_count,
        "reused_repairs": reused_count,
        "failed_repairs": failed_count,
        "batch_summary": summarize_batch(rows),
        "batch_json": str(output_root / "batch_review_summary.minicpmv45_repaired.json"),
        "batch_csv": str(output_root / "batch_review_summary.minicpmv45_repaired.csv"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

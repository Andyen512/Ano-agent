#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_models import BENCHMARK_MODELS, BenchmarkModel
from model_snapshot import is_model_snapshot_complete


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "data"
PROMPTS_DIR = Path(__file__).resolve().parent / "scripts" / "prompts"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "benchmark_inference"
STATUS_ROOT = RESULTS_ROOT / "status"
LOG_ROOT = RESULTS_ROOT / "logs"
EVALUATION_SCRIPT = Path(__file__).resolve().parent / "evaluate_outputs.py"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "latest"
EVALUATION_STATUS_PATH = RESULTS_ROOT / "evaluation_status.json"
PROMPT_FILE = PROMPTS_DIR / "infer_prompt_structured.txt"

TERMINAL_STATES = {"completed"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def find_existing_model_dir(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path_candidates = [
            MODELS_ROOT / candidate.replace("/", "__"),
            MODELS_ROOT / candidate.replace("/", "_"),
            MODELS_ROOT.joinpath(*[part for part in candidate.split("/") if part]),
            MODELS_ROOT / candidate.split("/")[-1],
        ]
        for path in dict.fromkeys(path_candidates):
            if path.exists():
                return path
    return None


def model_slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def sample_videos() -> list[Path]:
    return sorted(DATA_ROOT.glob("A*.mp4"))


def status_path(model: BenchmarkModel) -> Path:
    return STATUS_ROOT / f"{model_slug(model.name)}.json"


def model_results_dir(model: BenchmarkModel) -> Path:
    return RESULTS_ROOT / model_slug(model.name)


def model_logs_dir(model: BenchmarkModel) -> Path:
    return LOG_ROOT / model_slug(model.name)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_status(model: BenchmarkModel) -> dict | None:
    path = status_path(model)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(model: BenchmarkModel, payload: dict) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    status_path(model).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def has_incomplete_files(path: Path) -> bool:
    return any(path.rglob("*.incomplete"))


def is_download_complete(path: Path) -> bool:
    return is_model_snapshot_complete(path)


def hf_cache_file_exists(repo_id: str, relative_path: str) -> bool:
    local_dir = model_dir(repo_id)
    if (local_dir / relative_path).exists():
        return True
    candidates = [
        MODELS_ROOT / ".hf_cache" / f"models--{repo_id.replace('/', '--')}" / "snapshots",
        PROJECT_ROOT / "cache_dir" / f"models--{repo_id.replace('/', '--')}" / "snapshots",
    ]
    for snapshots_root in candidates:
        if not snapshots_root.exists():
            continue
        for snapshot in snapshots_root.iterdir():
            if (snapshot / relative_path).exists():
                return True
    return False


def repo_snapshot_available(repo_id: str) -> bool:
    local_dir = model_dir(repo_id)
    if any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin")) or any(local_dir.glob("*.index.json")):
        return True
    candidates = [
        MODELS_ROOT / ".hf_cache" / f"models--{repo_id.replace('/', '--')}" / "snapshots",
        PROJECT_ROOT / "cache_dir" / f"models--{repo_id.replace('/', '--')}" / "snapshots",
    ]
    for snapshots_root in candidates:
        if not snapshots_root.exists():
            continue
        for snapshot in snapshots_root.iterdir():
            if any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin")) or any(snapshot.glob("*.index.json")):
                return True
    return False


def qwen35_runtime_available() -> bool:
    vendor_dir = PROJECT_ROOT / ".vendor" / "qwen35_shim"
    vendor_path = str(vendor_dir)
    inserted = False
    if vendor_dir.exists() and vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
        inserted = True
    try:
        return importlib.util.find_spec("transformers.models.qwen3_5") is not None
    finally:
        if inserted and sys.path and sys.path[0] == vendor_path:
            sys.path.pop(0)


def preflight_blocker(model: BenchmarkModel, model_path: Path) -> str | None:
    if model.backend is None:
        return model.notes or "Backend is not wired in this environment yet."

    if model.backend == "qwen35vl" and not qwen35_runtime_available():
        return (
            "Qwen3.5-9B requires a recent Transformers build with Qwen3.5 support. "
            "Install transformers 5.4.0+ under lifebench/.vendor/qwen35_shim or upgrade the runtime environment."
        )

    if model.backend == "videochat2":
        mistral_dir = find_existing_model_dir(
            [
                "Mistral-7B-Instruct-v0.2",
                "mistralai/Mistral-7B-Instruct-v0.2",
            ]
        )
        bert_cache = MODELS_ROOT / ".hf_cache" / "models--bert-base-uncased"
        if mistral_dir is None:
            return (
                "VideoChat2 requires a local Mistral-7B-Instruct-v0.2 directory under models/."
            )
        if not bert_cache.exists():
            return (
                "VideoChat2 requires a local bert-base-uncased cache under "
                f"{bert_cache}."
            )

    if model.backend == "minigpt4video":
        mistral_dir = find_existing_model_dir(
            [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "Mistral-7B-Instruct-v0.2",
                "Mistral-7B-Instruct-v0.2",
            ]
        )
        bert_cache = MODELS_ROOT / ".hf_cache" / "models--bert-base-uncased"
        eva_path = MODELS_ROOT / "eva_vit_g.pth"
        ckpt_path = model_path / "checkpoints" / "video_mistral_checkpoint_last.pth"
        if mistral_dir is None:
            return "MiniGPT4-Video requires a local Mistral-7B-Instruct-v0.2 directory under models/."
        if not bert_cache.exists():
            return f"MiniGPT4-Video requires a local bert-base-uncased cache under {bert_cache}."
        if not eva_path.exists():
            return f"MiniGPT4-Video requires eva_vit_g.pth at {eva_path}."
        if not ckpt_path.exists():
            return f"MiniGPT4-Video requires checkpoint {ckpt_path}."

    if model.backend == "videollama2":
        config = load_json(model_path / "config.json") or {}
        vision_tower = config.get("mm_vision_tower")
        if isinstance(vision_tower, str) and "/" in vision_tower:
            if not repo_snapshot_available(vision_tower):
                return (
                    f"Auxiliary vision tower weights are missing locally: {vision_tower}. "
                    "VideoLLaMA2 will otherwise try to fetch them from Hugging Face at inference time."
                )

    if model.backend == "videollava":
        config = load_json(model_path / "config.json") or {}
        image_tower = config.get("mm_image_tower")
        if isinstance(image_tower, str) and "/" in image_tower:
            if not repo_snapshot_available(image_tower):
                return (
                    f"Auxiliary image tower weights are missing locally: {image_tower}. "
                    "Video-LLaVA still tries to fetch this dependency from Hugging Face during inference."
                )
        video_tower = config.get("mm_video_tower")
        if isinstance(video_tower, str) and "/" in video_tower:
            if not repo_snapshot_available(video_tower):
                return (
                    f"Auxiliary video tower weights are missing locally: {video_tower}. "
                    "Video-LLaVA still tries to fetch this dependency from Hugging Face during inference."
                )

    if model.backend == "videochatgpt":
        if not (model_path / "config.json").exists():
            base_dir = None
            for candidate in [
                MODELS_ROOT / "LLaVA-7B-Lightening-v1-1",
                MODELS_ROOT / "mmaaz60__LLaVA-7B-Lightening-v1-1",
                MODELS_ROOT / "liuhaotian__LLaVA-Lightning-7B-delta-v1-1",
            ]:
                if candidate.exists() and (candidate / "config.json").exists():
                    weight_files = list(candidate.glob("pytorch_model-*.bin")) + list(candidate.glob("model-*.safetensors"))
                    if weight_files:
                        base_dir = candidate
                        break
            if base_dir is not None and (base_dir / "config.json").exists() and (model_path / "video_chatgpt-7B.bin").exists():
                return None
            return (
                "Current local Video-ChatGPT snapshot only contains projection weights. "
                "It still needs a full local LLaVA-Lightening-7B-v1-1 base checkpoint plus the existing "
                "video_chatgpt-7B.bin projection file."
            )
        clip_dir = find_existing_model_dir(
            [
                "openai/clip-vit-large-patch14",
                "clip-vit-large-patch14",
            ]
        )
        if clip_dir is None:
            return (
                "Video-ChatGPT requires a local CLIP vision tower: "
                "openai/clip-vit-large-patch14."
            )

    return None


def classify_runtime_blocker(model: BenchmarkModel, stdout: str, stderr: str) -> str | None:
    detail = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part).strip()
    if not detail:
        return None

    if model.backend == "videollava" and "LanguageBind/LanguageBind_Image" in detail:
        return (
            "Video-LLaVA requires the auxiliary LanguageBind/LanguageBind_Image weights at inference time, "
            "but they are not cached locally and the runtime fell back to a live Hugging Face request."
        )

    if model.backend == "videollava" and "LanguageBind/LanguageBind_Video_merge" in detail:
        return (
            "Video-LLaVA requires the auxiliary LanguageBind/LanguageBind_Video_merge weights at inference time, "
            "but they are not cached locally and the runtime fell back to a live Hugging Face request."
        )

    if model.backend == "videollama2" and "google/siglip-so400m-patch14-384" in detail:
        return (
            "VideoLLaMA2 requires the auxiliary google/siglip-so400m-patch14-384 vision tower, "
            "but it is not cached locally and the runtime fell back to a live Hugging Face request."
        )

    if model.backend == "videochatgpt" and "Unrecognized model in" in detail and "VideoChatGPT" in detail:
        return (
            "Video-ChatGPT local snapshot is not directly loadable by Transformers; "
            "it is missing the expected Hugging Face config/model metadata."
        )

    if "Qwen3.5 local inference requires a recent Transformers build with Qwen3.5 support" in detail:
        return (
            "Qwen3.5-9B requires a recent Transformers build with Qwen3.5 support, "
            "preferably installed under lifebench/.vendor/qwen35_shim (tested with transformers 5.4.0)."
        )

    if "requires the flash_attn package" in detail:
        return "The required flash_attn dependency is not installed for this backend."

    if "requires qwen_vl_utils" in detail:
        return "The required qwen_vl_utils dependency is not installed for this backend."

    return None


def infer_one_model(model: BenchmarkModel, videos: list[Path], max_new_tokens: int) -> dict:
    run_id = run_timestamp()
    results_dir = model_results_dir(model) / run_id
    logs_dir = model_logs_dir(model) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    per_video: list[dict] = []

    for video in videos:
        print(f"[{now_iso()}] infer start: {model.name} -> {video.name}", flush=True)
        output_json = results_dir / f"{video.stem}.json"
        stdout_log = logs_dir / f"{video.stem}.stdout.log"
        stderr_log = logs_dir / f"{video.stem}.stderr.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "lifebench_infer.py"),
            "--backend",
            model.backend,
            "--model-id",
            model.model_id,
            "--video-path",
            str(video),
            "--prompt-file",
            str(PROMPT_FILE),
            "--max-new-tokens",
            str(max_new_tokens),
            "--output-json",
            str(output_json),
        ]
        started_at = now_iso()
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        stdout_log.write_text(completed.stdout, encoding="utf-8")
        stderr_log.write_text(completed.stderr, encoding="utf-8")
        per_video.append(
            {
                "video": str(video),
                "started_at": started_at,
                "finished_at": now_iso(),
                "returncode": completed.returncode,
                "run_id": run_id,
                "output_json": str(output_json),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stdout_tail": completed.stdout.strip()[-500:],
                "stderr_tail": completed.stderr.strip()[-500:],
            }
        )
        if completed.returncode != 0:
            blocked_reason = classify_runtime_blocker(model, completed.stdout, completed.stderr)
            print(f"[{now_iso()}] infer failed: {model.name} -> {video.name}", flush=True)
            payload = {"state": "failed", "per_video": per_video}
            if blocked_reason:
                payload["state"] = "blocked"
                payload["blocked_reason"] = blocked_reason
            payload["run_id"] = run_id
            payload["results_dir"] = str(results_dir)
            payload["logs_dir"] = str(logs_dir)
            return payload

    print(f"[{now_iso()}] infer completed: {model.name}", flush=True)
    return {
        "state": "completed",
        "run_id": run_id,
        "results_dir": str(results_dir),
        "logs_dir": str(logs_dir),
        "per_video": per_video,
    }


def run_evaluation() -> dict[str, Any]:
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        "--predictions-root",
        str(RESULTS_ROOT),
        "--output-dir",
        str(EVALUATION_OUTPUT_DIR),
    ]
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    summary_payload = load_json(EVALUATION_OUTPUT_DIR / "summary.json")
    payload: dict[str, Any] = {
        "updated_at": now_iso(),
        "returncode": completed.returncode,
        "output_dir": str(EVALUATION_OUTPUT_DIR),
        "stdout_tail": completed.stdout.strip()[-1000:],
        "stderr_tail": completed.stderr.strip()[-1000:],
    }
    if summary_payload is not None:
        payload["summary"] = {
            "generated_at": summary_payload.get("generated_at"),
            "prediction_file_count": summary_payload.get("prediction_file_count"),
            "model_ids": sorted((summary_payload.get("model_summary") or {}).keys()),
        }
    EVALUATION_STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def process_model(model: BenchmarkModel, videos: list[Path], max_new_tokens: int) -> dict:
    model_path = model_dir(model.model_id)
    base_payload = {
        "name": model.name,
        "model_id": model.model_id,
        "backend": model.backend,
        "model_dir": str(model_path),
        "notes": model.notes,
        "updated_at": now_iso(),
    }
    existing = read_status(model)
    if existing and existing.get("state") in TERMINAL_STATES:
        merged = {**existing, **base_payload}
        if merged != existing:
            write_status(model, merged)
        return merged

    if not model_path.exists():
        payload = {**base_payload, "state": "waiting_download"}
        write_status(model, payload)
        return payload

    if not is_download_complete(model_path):
        payload = {**base_payload, "state": "downloading"}
        write_status(model, payload)
        return payload

    blocker = preflight_blocker(model, model_path)
    if blocker:
        payload = {**base_payload, "state": "blocked", "blocked_reason": blocker}
        write_status(model, payload)
        return payload

    running_payload = {**base_payload, "state": "running"}
    write_status(model, running_payload)
    result = infer_one_model(model, videos, max_new_tokens)
    final_payload = {
        **base_payload,
        **result,
        "updated_at": now_iso(),
    }
    write_status(model, final_payload)
    return final_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch benchmark model downloads and run sample inference when each model is ready.")
    parser.add_argument("--watch-interval", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length used in sample inference.")
    parser.add_argument("--once", action="store_true", help="Process one pass and exit.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    videos = sample_videos()
    if not videos:
        raise FileNotFoundError(f"No sample videos found under {DATA_ROOT}")
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    while True:
        print(f"[{now_iso()}] poll start", flush=True)
        all_terminal = True
        summary: list[dict] = []
        for model in BENCHMARK_MODELS:
            payload = process_model(model, videos, args.max_new_tokens)
            summary.append(
                {
                    "name": model.name,
                    "state": payload["state"],
                    "backend": model.backend,
                    "model_dir": payload["model_dir"],
                }
            )
            if payload["state"] not in TERMINAL_STATES:
                all_terminal = False

        evaluation_status = run_evaluation()

        (RESULTS_ROOT / "summary.json").write_text(
            json.dumps(
                {
                    "updated_at": now_iso(),
                    "models": summary,
                    "evaluation": evaluation_status,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "updated_at": now_iso(),
                    "models": summary,
                    "evaluation": {
                        "returncode": evaluation_status.get("returncode"),
                        "summary": evaluation_status.get("summary"),
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if args.once or all_terminal:
            return 0
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    raise SystemExit(main())

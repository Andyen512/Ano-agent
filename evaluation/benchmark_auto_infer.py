#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
from dataclasses import asdict, dataclass
import gc
from pathlib import Path
from typing import Any

# Qwen3.5 does not need pandas/pyarrow for inference.  The environment's
# pyarrow 24.0.0 crashes in a native libarrow thread while pandas probes it,
# so keep that optional stack out of the Qwen-only worker.
if os.environ.get("LIFEBENCH_DISABLE_PYARROW") == "1":
    sys.modules["pyarrow"] = None

# Monkeypatch: add VideoInput to transformers.image_utils for VideoLLaMA3 compatibility
try:
    import transformers.image_utils as _iu
    if not hasattr(_iu, "VideoInput"):
        from typing import Union, List
        import numpy as _np
        import torch as _torch
        _iu.VideoInput = Union[List[List[_iu.ImageInput]], _iu.ImageType, str, Path, _np.ndarray, _torch.Tensor]
except Exception:
    pass

from model_snapshot import is_model_snapshot_complete


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"
EVALUATION_DIR = Path(__file__).resolve().parent

# session-based persistent inference (model loaded once per process)
from lifebench_infer import load_prompt as _load_prompt
from lifebench_batch_infer import create_session_bundle as _create_session_bundle

_session_bundle: Any = None
_current_session_key: str | None = None


def _teardown_session() -> None:
    global _session_bundle, _current_session_key
    if _session_bundle is not None:
        del _session_bundle
        _session_bundle = None
        _current_session_key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _ensure_session(entry: PipelineModel) -> Any:
    global _session_bundle, _current_session_key
    if _current_session_key == entry.key and _session_bundle is not None:
        return _session_bundle

    _teardown_session()

    model_path = model_dir(entry.model_id)
    session_args = argparse.Namespace()
    session_args.backend = entry.backend
    session_args.model_id = entry.model_id
    session_args.model_path = str(model_path)
    session_args.device_map = "auto"
    session_args.temperature = 0.0
    session_args.top_p = 0.9
    session_args.max_new_tokens = 512
    session_args.fps = 1.0
    session_args.max_frames = 32
    session_args.merge_size = 2
    session_args.attn_implementation = "auto"
    session_args.use_flash_attn = False
    if entry.backend == "internvl35":
        session_args.max_frames = 8
    if entry.backend == "videochat2":
        from compat.videochat2_once import VideoChat2Session
        _session_bundle = type('SessionBundle', (object,), {
            'backend_name': entry.backend,
            'model_id': entry.model_id,
            'model_path': model_path,
            'session': VideoChat2Session(str(model_path), num_segments=8),
        })()
        _current_session_key = entry.key
        log(f"[session] {entry.label} model loaded (backend={entry.backend})")
        return _session_bundle
    if entry.backend == "videochatgpt":
        proj = projection_path_for(model_path)
        session_args.projection_path = str(proj) if proj else None
    else:
        session_args.projection_path = None
    session_args.tarsier_config = str(
        PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"
    )

    _session_bundle = _create_session_bundle(session_args)
    _current_session_key = entry.key
    log(f"[session] {entry.label} model loaded (backend={entry.backend})")
    return _session_bundle


def _run_session_inference(entry: PipelineModel, video_path: Path, prompt_file: Path) -> tuple[int, str, str]:
    bundle = _ensure_session(entry)
    prompt = _load_prompt(None, str(prompt_file))
    video_str = str(video_path)
    backend = entry.backend

    if backend == "videollava":
        response = bundle.session.generate(video_str, prompt, 0.0, 256)
    elif backend == "videochat2":
        response = bundle.session.generate(video_str, prompt, 0.0, 256)
    elif backend == "minigpt4video":
        response = bundle.session.generate(video_str, prompt, 0.0, 512, 32)
    else:
        response = bundle.session.generate(video_str, prompt)

    return 0, response.strip(), ""

POLL_SECONDS = 60
RETRY_SECONDS = 30

DIMENSIONS = ["perception", "cognition", "grounding", "planning"]
OFFICIAL_EVAL_DIR = EVALUATION_DIR / "official_evaluation"

OUTPUT_ROOT: Path = PROJECT_ROOT / "outputs" / "benchmark_inference"
STATE_ROOT: Path = OUTPUT_ROOT / ".auto_infer_state"
VIDEO_FILES: list[Path] = sorted((PROJECT_ROOT / "data").glob("A*.mp4"))
PROMPT_FILES: dict[str, Path] = {dim: OFFICIAL_EVAL_DIR / dim / "prompt.txt" for dim in DIMENSIONS}
VIDEO_FILES_ROOT: Path = PROJECT_ROOT / "data"
_chunk_index: int | None = None
_chunk_total: int | None = None


def _get_chunk_suffix() -> str:
    if _chunk_index is not None and _chunk_total is not None:
        return f"_chunk{_chunk_index}of{_chunk_total}"
    return ""


def apply_settings(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT, STATE_ROOT, VIDEO_FILES, PROMPT_FILES, VIDEO_FILES_ROOT, _chunk_index, _chunk_total
    videos_dir = Path(args.videos_dir).resolve()
    OUTPUT_ROOT = Path(args.output_dir).resolve()
    STATE_ROOT = OUTPUT_ROOT / ".auto_infer_state"
    prompts_dir = Path(args.prompts_dir).resolve()
    PROMPT_FILES = {dim: prompts_dir / dim / "prompt.txt" for dim in DIMENSIONS}
    VIDEO_FILES_ROOT = videos_dir
    _chunk_index = args.chunk_index
    _chunk_total = args.total_chunks

    found = sorted(videos_dir.rglob("*.mp4"))
    legacy = sorted(videos_dir.glob("A*.mp4"))
    VIDEO_FILES = found if found else legacy

    # filter to only annotated videos
    _annot_dir = videos_dir.parent / "annotations"
    if _annot_dir.exists():
        _annot_ids = set()
        for _af in _annot_dir.rglob("*.json"):
            try:
                for _item in json.loads(_af.read_text(encoding="utf-8")):
                    _vid = _item.get("video_id", "") if isinstance(_item, dict) else ""
                    if _vid:
                        _annot_ids.add(_vid)
            except (json.JSONDecodeError, TypeError):
                pass
        if _annot_ids:
            _filtered = []
            for _vp in VIDEO_FILES:
                _rel = str(_vp.relative_to(videos_dir)).replace(".mp4", "")
                _bare = _rel.split("/", 1)[1] if "/" in _rel else _rel
                if _bare in _annot_ids:
                    _filtered.append(_vp)
            if _filtered:
                VIDEO_FILES = _filtered

    # chunk-based video splitting
    if _chunk_index is not None and _chunk_total is not None:
        VIDEO_FILES = [vp for i, vp in enumerate(VIDEO_FILES) if i % _chunk_total == _chunk_index]

    if not VIDEO_FILES:
        raise FileNotFoundError(f"No .mp4 video files found under: {videos_dir}")


@dataclass(frozen=True)
class PipelineModel:
    key: str
    label: str
    model_id: str
    backend: str | None
    blocked_reason: str | None = None


MODELS: list[PipelineModel] = [
    PipelineModel("video-llava-7b", "Video-LLaVA-7B", "LanguageBind/Video-LLaVA-7B", "videollava"),
    PipelineModel("videochat2-7b", "VideoChat2-7B", "OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf", "videochat2"),
    PipelineModel("video-chatgpt-7b", "Video-ChatGPT-7B", "MBZUAI/Video-ChatGPT-7B", "videochatgpt"),
    PipelineModel("minigpt4-video", "MiniGPT4-Video", "Vision-CAIR/MiniGPT4-Video", "minigpt4video"),
    PipelineModel("videollama2-7b", "VideoLLaMA2-7B", "DAMO-NLP-SG/VideoLLaMA2.1-7B-16F", "videollama2"),
    PipelineModel("internvl3.5-8b", "InternVL3.5-8B", "OpenGVLab/InternVL3_5-8B", "internvl35"),
    PipelineModel(
        "qwen2.5vl-7b",
        "Qwen2.5VL-7B",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen25vl",
    ),
    PipelineModel(
        "qwen3.5-9b",
        "Qwen3.5-9B",
        "Qwen/Qwen3.5-9B",
        "qwen35vl",
    ),
    PipelineModel("videollama3-7b", "VideoLLaMA3-7B", "DAMO-NLP-SG/VideoLLaMA3-7B", "videollama3"),
    PipelineModel("mplug-owl3-7b", "mPLUG-Owl3-7B", "mPLUG/mPLUG-Owl3-7B-241101", "mplugowl3"),
    PipelineModel("tarsier2-7b", "Tarsier2-7B", "omni-research/Tarsier2-7b-0115", "tarsier2"),
]


def model_dir(model_id: str) -> Path:
    candidates = [
        MODELS_ROOT / model_id.replace("/", "__"),
        MODELS_ROOT / model_id.replace("/", "_"),
        MODELS_ROOT.joinpath(*[part for part in model_id.split("/") if part]),
        MODELS_ROOT / model_id.split("/")[-1],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def output_model_dir(entry: PipelineModel) -> Path:
    return OUTPUT_ROOT / entry.key.replace("-", "_")


def state_path(entry: PipelineModel) -> Path:
    path = STATE_ROOT / f"{entry.key}.json"
    if _get_chunk_suffix():
        path = STATE_ROOT / f"{entry.key}{_get_chunk_suffix()}.json"
    return path


def load_state(entry: PipelineModel) -> dict[str, Any]:
    path = state_path(entry)
    if not path.exists():
        return {
            "model": asdict(entry),
            "status": "waiting_for_download",
            "attempts": 0,
            "run_id": None,
            "videos": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(entry: PipelineModel, state: dict[str, Any]) -> None:
    path = state_path(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(message, flush=True)


def completed_model_ids() -> set[str]:
    completed: set[str] = set()
    for entry in MODELS:
        if is_model_snapshot_complete(model_dir(entry.model_id)):
            completed.add(entry.model_id)

    patterns = [
        re.compile(r"Fast download complete: (.+?) ->"),
        re.compile(r"模型下载完成: (.+?) ->"),
    ]
    for log_path in MODELS_ROOT.glob("download_all_benchmark_models*.log"):
        if not log_path.is_file():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in patterns:
            for match in pattern.findall(text):
                completed.add(match.strip())
    return completed


def _video_key(video_path: Path) -> str:
    try:
        rel = video_path.relative_to(VIDEO_FILES_ROOT)
    except ValueError:
        rel = video_path
    return str(rel)


def projection_path_for(model_root: Path) -> Path | None:
    candidates = sorted(model_root.rglob("*video_chatgpt*.bin"))
    if candidates:
        return candidates[0]
    return None
    candidates = sorted(model_root.rglob("*video_chatgpt*.bin"))
    if candidates:
        return candidates[0]
    return None


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


def _bert_cache_exists() -> bool:
    return any([
        (MODELS_ROOT / "bert-base-uncased").exists(),
        (MODELS_ROOT / ".hf_cache" / "models--bert-base-uncased").exists(),
    ])


def runtime_blocked_reason(entry: PipelineModel) -> str | None:
    if entry.blocked_reason:
        return entry.blocked_reason
    if entry.backend == "qwen35vl" and not qwen35_runtime_available():
        return (
            "Requires a recent Transformers build with Qwen3.5 support. "
            "Install transformers 5.4.0+ under lifebench/.vendor/qwen35_shim or upgrade the runtime environment."
        )
    if entry.backend == "videochat2":
        mistral_dir = find_existing_model_dir([
            "Mistral-7B-Instruct-v0.2",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ])
        if mistral_dir is None:
            return "VideoChat2 requires a local Mistral-7B-Instruct-v0.2 directory under models/."
        if not _bert_cache_exists():
            return "VideoChat2 requires a local bert-base-uncased cache under models/."
    if entry.backend == "minigpt4video":
        mistral_dir = find_existing_model_dir([
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Mistral-7B-Instruct-v0.2",
        ])
        if mistral_dir is None:
            return "MiniGPT4-Video requires a local Mistral-7B-Instruct-v0.2 directory under models/."
        if not _bert_cache_exists():
            return "MiniGPT4-Video requires a local bert-base-uncased cache under models/."
        if not (MODELS_ROOT / "eva_vit_g.pth").exists():
            return "MiniGPT4-Video requires eva_vit_g.pth under models/."
        model_path = model_dir(entry.model_id)
        ckpt_path = model_path / "checkpoints" / "video_mistral_checkpoint_last.pth"
        if not ckpt_path.exists():
            return f"MiniGPT4-Video requires checkpoint at {ckpt_path}."
    return None


def all_dimensions_done(video_state: dict[str, Any]) -> bool:
    dims = video_state.get("dimensions", {})
    return all(dims.get(d, {}).get("status") == "success" for d in DIMENSIONS)


def any_dimension_has_error(video_state: dict[str, Any]) -> bool:
    dims = video_state.get("dimensions", {})
    return any(dims.get(d, {}).get("status") == "error" for d in DIMENSIONS)


def terminal_state(state: dict[str, Any]) -> bool:
    if state.get("status") == "blocked":
        return True
    return (
        all(all_dimensions_done(video_state) for video_state in state.get("videos", {}).values())
        and bool(state["videos"])
    )


def merge_responses(dim_responses: dict[str, str]) -> str:
    merged: dict[str, Any] = {}
    for dim in DIMENSIONS:
        response_text = dim_responses.get(dim, "")
        if not response_text:
            continue
        cleaned = response_text.strip()
        cleaned = cleaned.replace("\\_", "_")
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                start = cleaned.index("{")
                end = cleaned.rindex("}") + 1
                parsed = json.loads(cleaned[start:end])
            except (ValueError, json.JSONDecodeError):
                continue
        if isinstance(parsed, dict):
            merged.update(parsed)
    return json.dumps(merged, ensure_ascii=False)


def run_one_inference(entry: PipelineModel, video_path: Path, prompt_file: Path, run_id: str) -> tuple[int, str, str]:
    cmd = [
        str(Path(__file__).resolve().parent / "run_infer.sh"),
        "--backend",
        entry.backend,
        "--model-id",
        entry.model_id,
        "--video-path",
        str(video_path),
        "--prompt-file",
        str(prompt_file),
        "--print-json",
    ]

    if entry.backend == "videochatgpt":
        projection_path = projection_path_for(model_dir(entry.model_id))
        if projection_path is not None:
            cmd.extend(["--projection-path", str(projection_path)])

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def extract_raw_response(infer_stdout: str) -> str:
    try:
        result = json.loads(infer_stdout)
        if isinstance(result, dict):
            return result.get("response", infer_stdout)
    except (json.JSONDecodeError, TypeError):
        pass
    return infer_stdout


def process_entry(entry: PipelineModel, downloaded: set[str], now: float) -> bool:
    state = load_state(entry)
    state["model"] = asdict(entry)

    if entry.model_id not in downloaded:
        state["status"] = "waiting_for_download"
        save_state(entry, state)
        return False

    blocked_reason = runtime_blocked_reason(entry)
    if entry.backend is None or blocked_reason is not None:
        detail = blocked_reason or entry.blocked_reason or "Backend is not wired in this environment yet."
        state["status"] = "blocked"
        state["blocked_reason"] = detail
        log(f"[blocked] {entry.label}: {detail}")
        save_state(entry, state)
        return True

    state.setdefault("videos", {})
    if not state.get("run_id"):
        state["run_id"] = run_timestamp()

    all_videos_done = True
    attempted_any = False

    for video_path in VIDEO_FILES:
        vkey = _video_key(video_path)
        video_state = state["videos"].setdefault(vkey, {})
        # migrate old-style key (just filename) to new-style key (relative path)
        old_key = video_path.name
        if old_key != vkey and old_key in state["videos"] and old_key not in state["videos"]:
            # happens after migration: old key already migrated
            pass
        if old_key != vkey and old_key in state["videos"] and not video_state:
            state["videos"][vkey] = state["videos"].pop(old_key)
            video_state = state["videos"][vkey]
        dims = video_state.setdefault("dimensions", {})

        if all_dimensions_done(video_state):
            continue

        if any_dimension_has_error(video_state):
            # skip only if ALL errored dimensions are still within retry cooldown
            errored_dims = [d for d in DIMENSIONS if dims.get(d, {}).get("status") == "error"]
            if errored_dims and all(
                now - float(dims[d].get("last_attempt_at", 0)) < RETRY_SECONDS
                for d in errored_dims
            ):
                all_videos_done = False
                continue
            # otherwise fall through to retry

        video_all_done = True
        for dim in DIMENSIONS:
            dim_state = dims.get(dim, {})
            if dim_state.get("status") == "success":
                continue
            if now - float(dim_state.get("last_attempt_at", 0)) < RETRY_SECONDS:
                video_all_done = False
                continue

            prompt_file = PROMPT_FILES[dim]
            if not prompt_file.exists():
                log(f"[missing_prompt] {entry.label} / {dim}: {prompt_file} not found, skipping")
                dims[dim] = {"status": "error", "error": f"prompt file not found: {prompt_file}"}
                save_state(entry, state)
                continue

            attempted_any = True
            state["attempts"] = int(state.get("attempts", 0)) + 1
            state["status"] = "running_inference"
            dim_state["last_attempt_at"] = now
            dims[dim] = dim_state
            save_state(entry, state)

            log(f"[infer] {entry.label} / {dim} -> {vkey}")
            try:
                returncode, stdout, stderr = _run_session_inference(entry, video_path, prompt_file)
            except Exception:
                import traceback as _traceback
                returncode = 1
                stdout = ""
                stderr = _traceback.format_exc()

            if returncode == 0:
                raw_response = extract_raw_response(stdout)
                dims[dim] = {
                    "status": "success",
                    "last_attempt_at": now,
                    "response": raw_response,
                }
                log(f"[success] {entry.label} / {dim} -> {vkey}")
            else:
                dims[dim] = {
                    "status": "error",
                    "last_attempt_at": now,
                    "error": (stderr or stdout).strip()[-4000:],
                }
                video_all_done = False
                log(f"[error] {entry.label} / {dim} -> {vkey}: {(stderr or stdout).strip()[:300]}")
            save_state(entry, state)

            if returncode != 0:
                video_all_done = False
                break

        if video_all_done and all_dimensions_done(video_state):
            dim_responses = {d: dims[d].get("response", "") for d in DIMENSIONS}
            merged = merge_responses(dim_responses)
            output_dir = output_model_dir(entry) / str(state["run_id"])
            output_dir.mkdir(parents=True, exist_ok=True)
            vkey_rel = Path(vkey)
            if vkey_rel.parent != Path("."):
                (output_dir / vkey_rel.parent).mkdir(parents=True, exist_ok=True)
                merged_path = output_dir / vkey_rel.parent / f"{video_path.stem}.json"
            else:
                merged_path = output_dir / f"{video_path.stem}.json"
            merged_payload = {
                "backend": entry.backend,
                "model_id": entry.model_id,
                "video_path": str(video_path.resolve()),
                "video_key": vkey,
                "response": merged,
            }
            merged_path.write_text(
                json.dumps(merged_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            log(f"[merged] {entry.label} -> {vkey}")
        elif not video_all_done:
            all_videos_done = False

    if all_videos_done and state["videos"]:
        state["status"] = "completed"
        log(f"[completed] {entry.label}")
        _teardown_session()
    elif attempted_any:
        state["status"] = "running_inference"
    else:
        state["status"] = "waiting_for_retry"
    save_state(entry, state)
    return terminal_state(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LifeBench multi-dimension benchmark inference pipeline."
    )
    parser.add_argument(
        "--videos-dir",
        default=str(PROJECT_ROOT / "data"),
        help="Directory containing .mp4 video files (searched recursively). Default: project_root/data",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "benchmark_inference"),
        help="Root directory for inference output. Default: outputs/benchmark_inference",
    )
    parser.add_argument(
        "--prompts-dir",
        default=str(OFFICIAL_EVAL_DIR),
        help="Directory containing dimension prompt.txt subdirectories. Default: evaluation/official_evaluation",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device ID. Sets CUDA_VISIBLE_DEVICES for this process. Default: all visible GPUs",
    )
    parser.add_argument(
        "--model-keys",
        nargs="*",
        default=None,
        help="Only process these model keys (e.g. 'qwen2.5vl-7b videollama3-7b'). Default: all models",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="When set with --total-chunks, only process videos where index %% total == chunk_index",
    )
    parser.add_argument(
        "--total-chunks",
        type=int,
        default=None,
        help="Number of parallel chunks for video distribution",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    apply_settings(args)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        log(f"Device:   CUDA_VISIBLE_DEVICES={args.device}")

    entries = [m for m in MODELS if not args.model_keys or m.key in args.model_keys]
    if not entries:
        log("[error] No models match the specified --model-keys")
        return 1

    for dim in DIMENSIONS:
        prompt_file = PROMPT_FILES[dim]
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    log(f"Videos:   {len(VIDEO_FILES)} found")
    log(f"Output:   {OUTPUT_ROOT}")
    log(f"Prompts:  {args.prompts_dir}")
    log(f"Models:   {len(entries)} / {len(MODELS)} configured: {[e.key for e in entries]}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        downloaded = completed_model_ids()
        terminal_count = 0
        now = time.time()
        for entry in entries:
            if process_entry(entry, downloaded, now):
                terminal_count += 1
        if terminal_count == len(entries):
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

from common import (
    RISK_STATUS_NO_ANOMALY,
    RISK_STATUS_OCCURRED,
    RISK_STATUS_POTENTIAL,
    normalize_risk_status_label,
)
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
    session_args.max_new_tokens = 800
    session_args.fps = 1.0
    session_args.max_frames = 32
    session_args.merge_size = 2
    session_args.attn_implementation = "auto"
    session_args.use_flash_attn = False
    if entry.backend == "internvl35":
        session_args.max_frames = int(os.environ.get("LIFEBENCH_INTERNVL35_MAX_FRAMES", "8"))
    if entry.backend == "tarsier2":
        session_args.max_frames = int(os.environ.get("LIFEBENCH_TARSIER_MAX_FRAMES", "32"))
    if entry.backend == "videochat2":
        from compat.videochat2_once import VideoChat2Session
        _session_bundle = type('SessionBundle', (object,), {
            'backend_name': entry.backend,
            'model_id': entry.model_id,
            'model_path': model_path,
            'session': VideoChat2Session(str(model_path), num_segments=32),
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


def _run_session_inference(
    entry: PipelineModel,
    video_path: Path,
    prompt: str | Path,
) -> tuple[int, str, str]:
    bundle = _ensure_session(entry)
    if isinstance(prompt, Path):
        prompt = _load_prompt(None, str(prompt))
    video_str = str(video_path)
    backend = entry.backend

    if backend == "videollava":
        response = bundle.session.generate(video_str, prompt, 0.0, 256)
    elif backend == "videochat2":
        response = bundle.session.generate(video_str, prompt, 0.0, 256)
    elif backend == "minigpt4video":
        response = bundle.session.generate(video_str, prompt, 0.0, 800, 32)
    else:
        response = bundle.session.generate(video_str, prompt)

    return 0, response.strip(), ""

POLL_SECONDS = 60
RETRY_SECONDS = 30
MAX_DIMENSION_ATTEMPTS = 3

DIMENSIONS = ["perception", "cognition", "grounding", "planning"]
PROMPTS_DIR = EVALUATION_DIR / "official_evaluation" / "real_videos"
OFFICIAL_EVAL_DIR = PROMPTS_DIR
SINGLE_PROMPTS_DIR = EVALUATION_DIR / "scripts" / "prompts"

OUTPUT_ROOT: Path = PROJECT_ROOT / "outputs" / "benchmark_inference"
STATE_ROOT: Path = OUTPUT_ROOT / ".auto_infer_state"
VIDEO_FILES: list[Path] = sorted((PROJECT_ROOT / "data").glob("A*.mp4"))
PROMPT_FILES: dict[str, Path] = {dim: PROMPTS_DIR / dim / "prompt.txt" for dim in DIMENSIONS}
SINGLE_PROMPT_FILE: Path = SINGLE_PROMPTS_DIR / "infer_prompt_structured.txt"
VIDEO_FILES_ROOT: Path = PROJECT_ROOT / "data"
_chunk_index: int | None = None
_chunk_total: int | None = None
_inference_mode: str = "sequential"
# Dimensions actually inferred in the current run. gt_hint skips grounding.
ACTIVE_DIMENSIONS: list[str] = DIMENSIONS
GT_HINT_DIR: Path = PROJECT_ROOT / "data" / "public_data_release" / "annotations" / "real_videos"
_GT_HINT_INDEX: dict[str, dict[str, Any]] = {}

GROUNDING_FRAME_COUNTS: dict[str, int] = {
    "videollama2": 16,
    "qwen25vl": 32,
    "qwen35vl": 32,
    "internvl35": 32,
    "tarsier2": 32,
    "videollama3": 32,
    "videochat2": 32,
    "videochatgpt": 32,
}


def build_grounding_frame_metadata(video_path: Path, backend: str) -> dict[str, Any]:
    """Compute the per-video frame/time mapping used in the single-pass prompt."""
    import numpy as np

    frame_count = 0
    fps = 0.0
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(reader)
        fps = float(reader.get_avg_fps() or 0.0)
    except Exception:
        try:
            import av

            container = av.open(str(video_path))
            stream = container.streams.video[0]
            frame_count = int(stream.frames or 0)
            rate = stream.average_rate or stream.base_rate
            fps = float(rate or 0.0)
            if frame_count <= 0 and stream.duration and stream.time_base and fps > 0:
                frame_count = int(round(float(stream.duration * stream.time_base) * fps))
            container.close()
        except Exception:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            capture.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"Unable to read frame count/FPS for grounding: {video_path}")

    sample_count = min(GROUNDING_FRAME_COUNTS.get(backend, 32), frame_count)
    if backend == "videochat2":
        segment_size = float(frame_count - 1) / sample_count
        frame_indices = [
            int(segment_size / 2 + round(segment_size * index))
            for index in range(sample_count)
        ]
    elif backend == "internvl35":
        segment_size = float(frame_count - 1) / sample_count
        frame_indices = [
            int(segment_size / 2 + round(segment_size * index))
            for index in range(sample_count)
        ]
    else:
        frame_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist()
    return {
        "backend": backend,
        "frame_count": frame_count,
        "fps": round(fps, 6),
        "sample_count": sample_count,
        "frame_indices": frame_indices,
        "frame_timestamps": [round(index / fps, 4) for index in frame_indices],
    }


def append_grounding_frame_context(prompt: str, metadata: dict[str, Any]) -> str:
    mapping = "\n".join(
        f"Frame {index}: {timestamp:.4f} seconds"
        for index, timestamp in enumerate(metadata["frame_timestamps"], start=1)
    )
    return (
        f"{prompt.rstrip()}\n\n"
        "本次视频输入帧的时间映射如下。Frame 编号从 1 开始，必须只输出 frame_spans，"
        "不要猜测或输出 time_spans 的秒数：\n"
        f"{mapping}"
    )


def _gt_hint_prefer(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    def key(entry: dict[str, Any]) -> tuple[int, bool]:
        status = normalize_risk_status_label(entry.get("risk_status", ""))
        non_normal = status in (RISK_STATUS_POTENTIAL, RISK_STATUS_OCCURRED)
        return (1 if non_normal else 0, bool(entry.get("time_spans")))
    if existing is None:
        return candidate
    if key(candidate) > key(existing):
        return candidate
    return existing


def load_gt_hint_index(gt_dir: Path) -> dict[str, dict[str, Any]]:
    """Index grounding_gt.json entries by video_id, preferring entries that
    carry a non-normal risk status with time_spans."""
    index: dict[str, dict[str, Any]] = {}
    for gt_file in sorted(gt_dir.rglob("grounding_gt.json")):
        try:
            items = json.loads(gt_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            video_id = item.get("video_id", "")
            if not video_id:
                continue
            index[video_id] = _gt_hint_prefer(index.get(video_id), item)
    return index


def match_gt_hint_entry(
    video_path: Path, index: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Match a video file to its GT entry using the same id conventions as the
    released real_videos layout (GT video_id = video path minus status dir)."""
    if not index:
        return None
    try:
        rel = video_path.relative_to(VIDEO_FILES_ROOT)
    except ValueError:
        rel = video_path
    stem = str(rel).removesuffix(".mp4")
    if stem in index:
        return index[stem]
    parts = stem.split("/")
    if len(parts) > 1:
        stripped = "/".join(parts[1:])
        if stripped in index:
            return index[stripped]
    basename = parts[-1]
    basename_matches = [
        entry for video_id, entry in index.items() if video_id.endswith("/" + basename)
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def append_gt_interval_hint(
    prompt: str, metadata: dict[str, Any], spans: list[list[float]]
) -> str:
    """Append the GT risk interval (as seconds and nearest sampled frame ids)
    to a per-dimension prompt."""
    timestamps = metadata["frame_timestamps"]
    sample_count = metadata["sample_count"]
    mapping = "\n".join(
        f"Frame {index}: {timestamp:.4f} seconds"
        for index, timestamp in enumerate(timestamps, start=1)
    )
    lines: list[str] = []
    for span in spans:
        try:
            start, end = float(span[0]), float(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        start_frame = min(range(sample_count), key=lambda i: abs(timestamps[i] - start))
        end_frame = min(range(sample_count), key=lambda i: abs(timestamps[i] - end))
        lines.append(
            f"- 时间区间 {start:.2f}s ~ {end:.2f}s：对应最近的采样帧为 Frame {start_frame + 1} 至 Frame {end_frame + 1}"
        )
    if not lines:
        return prompt
    return (
        f"{prompt.rstrip()}\n\n"
        "补充信息（该视频的风险发生时间区间，来自人工标注，请据此重点分析对应帧）：\n"
        + "\n".join(lines)
        + "\n\n本次视频输入帧的时间映射如下，Frame 编号从 1 开始：\n"
        + mapping
    )


def convert_frame_spans_to_time_spans(response: str, metadata: dict[str, Any]) -> str:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(parsed, dict):
        return response

    timestamps = metadata["frame_timestamps"]
    converted: list[list[float]] = []
    raw_spans = parsed.get("frame_spans") or []
    if isinstance(raw_spans, list):
        for item in raw_spans:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                start_frame = max(1, min(len(timestamps), int(item[0])))
                end_frame = max(1, min(len(timestamps), int(item[1])))
            except (TypeError, ValueError):
                continue
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame
            start_time = timestamps[start_frame - 1]
            end_time = timestamps[end_frame - 1]
            converted.append([math.floor(start_time), math.ceil(end_time)])
    parsed["time_spans"] = converted
    parsed["grounding_frame_metadata"] = metadata
    return json.dumps(parsed, ensure_ascii=False)


def _get_chunk_suffix() -> str:
    if _chunk_index is not None and _chunk_total is not None:
        return f"_chunk{_chunk_index}of{_chunk_total}"
    return ""


def apply_settings(args: argparse.Namespace) -> None:
    global OUTPUT_ROOT, STATE_ROOT, VIDEO_FILES, PROMPT_FILES, SINGLE_PROMPT_FILE, VIDEO_FILES_ROOT, _chunk_index, _chunk_total, _inference_mode, ACTIVE_DIMENSIONS, GT_HINT_DIR, _GT_HINT_INDEX
    videos_dir = Path(args.videos_dir).resolve()
    OUTPUT_ROOT = Path(args.output_dir).resolve()
    STATE_ROOT = OUTPUT_ROOT / ".auto_infer_state"
    prompts_dir = Path(args.prompts_dir).resolve()
    PROMPT_FILES = {dim: prompts_dir / dim / "prompt.txt" for dim in DIMENSIONS}
    SINGLE_PROMPT_FILE = Path(args.single_prompt_file).resolve()
    VIDEO_FILES_ROOT = videos_dir
    _chunk_index = args.chunk_index
    _chunk_total = args.total_chunks
    _inference_mode = args.inference_mode
    ACTIVE_DIMENSIONS = DIMENSIONS
    if _inference_mode == "gt_hint":
        ACTIVE_DIMENSIONS = [dim for dim in DIMENSIONS if dim != "grounding"]
    GT_HINT_DIR = Path(args.gt_hint_dir).resolve()

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

    if os.environ.get("LIFEBENCH_REBALANCE_PENDING") == "1":
        model_key = os.environ.get("LIFEBENCH_REBALANCE_MODEL_KEY", "")
        completed_keys: set[str] = set()
        if model_key and STATE_ROOT.exists():
            for state_path in STATE_ROOT.rglob("*.json"):
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if state.get("model", {}).get("key") != model_key:
                    continue
                for video_key, video_state in state.get("videos", {}).items():
                    dimensions = video_state.get("dimensions", {})
                    if all(
                        dimensions.get(dimension, {}).get("status") in {"success", "skipped"}
                        for dimension in DIMENSIONS
                    ):
                        completed_keys.add(video_key)
        VIDEO_FILES = [
            video_path for video_path in VIDEO_FILES
            if _video_key(video_path) not in completed_keys
        ]
        if _chunk_index is not None and _chunk_total is not None:
            log(
                f"[rebalance] {model_key}: {len(completed_keys)} completed, "
                f"{len(VIDEO_FILES)} pending across {_chunk_total} workers"
            )

    # chunk-based video splitting
    if _chunk_index is not None and _chunk_total is not None:
        VIDEO_FILES = [vp for i, vp in enumerate(VIDEO_FILES) if i % _chunk_total == _chunk_index]

    skip_file = PROJECT_ROOT / "data" / "public_data_release" / "real_videos_infer_skip.txt"
    skip_keys = set()
    if skip_file.exists():
        skip_keys.update(
            line.strip().removesuffix(".mp4")
            for line in skip_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    global_skip_file = os.environ.get("LIFEBENCH_GLOBAL_SKIP_FILE", "")
    if not global_skip_file and videos_dir.name == "generated_videos":
        global_skip_file = str(
            videos_dir.parent / "prediction" / videos_dir.name / "generated_video_infer_skip.txt"
        )
    if global_skip_file:
        global_skip_path = Path(global_skip_file)
        if global_skip_path.exists():
            skip_keys.update(
                line.strip().removesuffix(".mp4")
                for line in global_skip_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
    skip_keys.update(
        key.strip().removesuffix(".mp4")
        for key in os.environ.get("LIFEBENCH_SKIP_VIDEO_KEYS", "").split(",")
        if key.strip()
    )
    if skip_keys:
        VIDEO_FILES = [
            video_path
            for video_path in VIDEO_FILES
            if _video_key(video_path).removesuffix(".mp4") not in skip_keys
        ]

    if _inference_mode == "gt_hint":
        index = load_gt_hint_index(GT_HINT_DIR)
        kept: list[Path] = []
        for video_path in VIDEO_FILES:
            entry = match_gt_hint_entry(video_path, index)
            if entry is None:
                continue
            status = normalize_risk_status_label(entry.get("risk_status", ""))
            if status not in (RISK_STATUS_POTENTIAL, RISK_STATUS_OCCURRED):
                continue
            if not entry.get("time_spans"):
                continue
            kept.append(video_path)
        VIDEO_FILES = kept
        _GT_HINT_INDEX.clear()
        for video_path in VIDEO_FILES:
            entry = match_gt_hint_entry(video_path, index)
            if entry is not None:
                _GT_HINT_INDEX[_video_key(video_path)] = entry
        log(
            f"[gt_hint] GT dir: {GT_HINT_DIR}; "
            f"{len(VIDEO_FILES)} abnormal/risk_only videos with GT time_spans"
        )

    if not VIDEO_FILES and os.environ.get("LIFEBENCH_REBALANCE_PENDING") != "1":
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


STATE_DIR_NAMES = {
    "internvl3.5-8b": "internvl35",
    "qwen2.5vl-7b": "qwen25vl",
    "qwen3.5-9b": "qwen35",
    "video-llava-7b": "video_llava",
    "video-chatgpt-7b": "videochatgpt",
    "videollama2-7b": "videollama2",
    "videollama3-7b": "videollama3",
    "mplug-owl3-7b": "mplug_owl3",
    "minigpt4-video": "minigpt4",
    "tarsier2-7b": "tarsier2",
}


def state_path(entry: PipelineModel) -> Path:
    model_state_root = STATE_ROOT / STATE_DIR_NAMES.get(entry.key, entry.key)
    path = model_state_root / f"{entry.key}.json"
    if _get_chunk_suffix():
        path = model_state_root / f"{entry.key}{_get_chunk_suffix()}.json"
    return path


def load_state(entry: PipelineModel) -> dict[str, Any]:
    path = state_path(entry)
    initial_state = {
        "model": asdict(entry),
        "status": "waiting_for_download",
        "attempts": 0,
        "run_id": None,
        "videos": {},
    }
    if os.environ.get("LIFEBENCH_FORCE_REINFER") == "1":
        return initial_state
    if not path.exists() or path.stat().st_size == 0:
        return initial_state
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log(f"[state] invalid JSON in {path}; starting from a fresh state")
        return initial_state


def save_state(entry: PipelineModel, state: dict[str, Any]) -> None:
    path = state_path(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


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
    return all(dims.get(d, {}).get("status") in {"success", "skipped"} for d in ACTIVE_DIMENSIONS)


def any_dimension_has_error(video_state: dict[str, Any]) -> bool:
    dims = video_state.get("dimensions", {})
    return any(dims.get(d, {}).get("status") == "error" for d in ACTIVE_DIMENSIONS)


def skip_exhausted_dimensions(video_state: dict[str, Any], inference_mode: str) -> bool:
    dims = video_state.setdefault("dimensions", {})
    exhausted = [
        dim for dim in ACTIVE_DIMENSIONS
        if dims.get(dim, {}).get("status") == "error"
        and int(dims.get(dim, {}).get("attempt_count", 0)) >= MAX_DIMENSION_ATTEMPTS
    ]
    if not exhausted:
        return False

    for dim in ACTIVE_DIMENSIONS:
        dim_state = dims.get(dim, {})
        if dim_state.get("status") == "error":
            dims[dim] = {
                **dim_state,
                "status": "skipped",
                "skip_reason": f"failed after {MAX_DIMENSION_ATTEMPTS} attempts",
            }
        elif inference_mode == "single" and dim_state.get("status") != "success":
            dims[dim] = {
                "status": "skipped",
                "skip_reason": "single-pass response failed in another dimension",
            }
    return True


def terminal_state(state: dict[str, Any]) -> bool:
    if state.get("status") == "blocked":
        return True
    skip_file = PROJECT_ROOT / "data" / "public_data_release" / "real_videos_infer_skip.txt"
    skip_keys = set()
    if skip_file.exists():
        skip_keys.update(
            line.strip().removesuffix(".mp4")
            for line in skip_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    global_skip_file = os.environ.get("LIFEBENCH_GLOBAL_SKIP_FILE", "")
    if not global_skip_file and VIDEO_FILES_ROOT.name == "generated_videos":
        global_skip_file = str(
            VIDEO_FILES_ROOT.parent / "prediction" / VIDEO_FILES_ROOT.name / "generated_video_infer_skip.txt"
        )
    if global_skip_file:
        global_skip_path = Path(global_skip_file)
        if global_skip_path.exists():
            skip_keys.update(
                line.strip().removesuffix(".mp4")
                for line in global_skip_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
    skip_keys.update(
        key.strip().removesuffix(".mp4")
        for key in os.environ.get("LIFEBENCH_SKIP_VIDEO_KEYS", "").split(",")
        if key.strip()
    )
    video_states = [
        video_state
        for video_key, video_state in state.get("videos", {}).items()
        if video_key.removesuffix(".mp4") not in skip_keys
    ]
    return (
        all(all_dimensions_done(video_state) for video_state in video_states)
        and bool(video_states)
    )


def merge_responses(dim_responses: dict[str, str]) -> str:
    merged: dict[str, Any] = {}
    for dim in ACTIVE_DIMENSIONS:
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


def prompt_with_previous_results(
    prompt: str,
    current_dim: str,
    dimensions: dict[str, dict[str, Any]],
) -> str:
    current_index = DIMENSIONS.index(current_dim)
    previous_results: list[str] = []
    for dim in DIMENSIONS[:current_index]:
        response = dimensions.get(dim, {}).get("response", "")
        if response:
            previous_results.append(f"[{dim}]\n{response.strip()}")

    if not previous_results:
        return prompt

    context = "\n\n".join(previous_results)
    return (
        f"{prompt}\n\n"
        "以下是同一视频已经完成的前序分析结果。请将其作为当前分析的依据，"
        "不要重新推翻前序的 risk_status；只输出当前维度要求的字段。\n"
        "--- 前序分析结果 ---\n"
        f"{context}\n"
        "--- 前序分析结果结束 ---"
    )


def build_single_pass_prompt() -> str:
    return """你是视频安全分析专家。请完整观看视频，只输出一个 JSON 对象，不要输出解释、前缀、后缀或 Markdown。

请一次性完成四个维度的分析，并严格输出以下全部字段：
- risk_status: normal / risk_only / abnormal
- risk_type: 跌倒失稳、动物攻击、碰撞砸伤、用电安全、高温火源、人际冲突、中毒误食、锐器危险、陌生盗窃/入侵、溺水、自然灾害
- risk_sources: string 数组
- abnormal_actions: string 数组
- affected_objects: string 数组
- risk_description: string
- consequence_understanding: string
- causal_chain: string
- causal_chain_text: string
- event_summary: string
        - frame_spans: [[start_frame, end_frame]] 数组，帧号从 1 开始；程序会转换为 time_spans
- solution: string，包含“对人的解决方案：”“对危险源的解决方案：”“整体的解决方案：”三段

判断规则：normal 表示无异常事件且无明显风险征兆；risk_only 表示尚未发生异常但存在明显风险征兆；abnormal 表示异常事件已经发生。normal 时除 risk_status 外，其余内容按字段类型输出空值；risk_only 和 abnormal 时填写所有相关字段。"""


SINGLE_PASS_FIELDS: dict[str, tuple[str, ...]] = {
    "perception": ("risk_status", "risk_type", "risk_sources", "abnormal_actions", "affected_objects"),
    "cognition": ("risk_description", "consequence_understanding", "causal_chain", "causal_chain_text", "event_summary"),
    "grounding": ("risk_status", "frame_spans"),
    "planning": ("risk_status", "solution"),
}


def split_single_pass_response(response: str) -> dict[str, str] | None:
    cleaned = response.strip().replace("\\_", "_")
    # strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # try direct parse
    for attempt in (cleaned, re.sub(r"'''+|“|”", '"', cleaned)):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return _build_split_result(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # try to extract JSON object from text
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
        extracted = cleaned[start:end]
        parsed = json.loads(extracted)
        if isinstance(parsed, dict):
            return _build_split_result(parsed)
    except (ValueError, json.JSONDecodeError):
        pass

    # try json_repair library
    try:
        from json_repair import repair_json  # type: ignore[import-untyped]
    except ImportError:
        pass
    else:
        for candidate in (cleaned,):
            try:
                start = candidate.index("{")
                end = candidate.rindex("}") + 1
                extracted = candidate[start:end]
            except ValueError:
                extracted = candidate
            try:
                repaired = repair_json(extracted)
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return _build_split_result(parsed)
            except Exception:
                pass

    # Some VLMs stop during a long JSON string. Recover the fields that were
    # emitted so one malformed video does not retry forever and block the run.
    try:
        from lifebench_infer import parse_structured_response_text

        recovered = parse_structured_response_text(cleaned)
        if isinstance(recovered, dict):
            return _build_split_result(recovered)
    except Exception:
        pass

    return None


def _build_split_result(parsed: dict) -> dict[str, str]:
    # normalize common MiniGPT4 field name typos
    _FIELD_ALIASES: dict[str, str] = {
        "risksources": "risk_sources",
        "risk_sourcess": "risk_sources",
        "abnormal_actions": "abnormal_actions",  # identity, removes spaces
        "ab normal_actions": "abnormal_actions",
        "affectedobjects": "affected_objects",
        "affected_objects": "affected_objects",
        "risk_description": "risk_description",
        "risk_descriptions": "risk_description",
        "consequence_understanding": "consequence_understanding",
        "consequence_understandings": "consequence_understanding",
        "causal_chain": "causal_chain",
        "causalchain": "causal_chain",
        "causal_chains": "causal_chain",
        "causal_chain_text": "causal_chain_text",
        "causalchaint ext": "causal_chain_text",
        "casualchaintxt": "causal_chain_text",
        "casual_chaintxt": "causal_chain_text",
        "event_summary": "event_summary",
        "event_summaries": "event_summary",
        "time_spans": "time_spans",
        "timespans": "time_spans",
        "frame_spans": "frame_spans",
        "framespans": "frame_spans",
        "solution": "solution",
    }
    canonical: dict[str, object] = {}
    for key, value in parsed.items():
        clean_key = key.strip().replace(" ", "").replace("_", "").lower()
        mapped = None
        # exact alias match first
        if key in _FIELD_ALIASES:
            mapped = _FIELD_ALIASES[key]
        else:
            # fuzzy: strip all spaces/underscores for comparison
            clean_key = key.strip().replace(" ", "").replace("_", "").lower()
            for alias, target in _FIELD_ALIASES.items():
                alias_clean = alias.strip().replace(" ", "").replace("_", "").lower()
                if clean_key == alias_clean:
                    mapped = target
                    break
        if mapped is not None:
            canonical[mapped] = value
        else:
            canonical[key] = value
    return {
        dim: json.dumps(
            {field: canonical[field] for field in fields if field in canonical},
            ensure_ascii=False,
        )
        for dim, fields in SINGLE_PASS_FIELDS.items()
    }


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

    if (
        os.environ.get("LIFEBENCH_SKIP_UNRESOLVED_MODEL_KEY") == entry.key
        and not VIDEO_FILES
    ):
        for video_state in state["videos"].values():
            dimensions = video_state.setdefault("dimensions", {})
            if all(
                dimensions.get(dimension, {}).get("status") in {"success", "skipped"}
                for dimension in DIMENSIONS
            ):
                continue
            for dimension in DIMENSIONS:
                dimensions[dimension] = {
                    "status": "skipped",
                    "skip_reason": "manually skipped after prolonged inference stall",
                }
        save_state(entry, state)

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

        if skip_exhausted_dimensions(video_state, _inference_mode):
            save_state(entry, state)

        if all_dimensions_done(video_state):
            continue

        if os.environ.get("LIFEBENCH_SKIP_UNRESOLVED_MODEL_KEY") == entry.key:
            log(f"[skip-unresolved] {entry.label} -> {vkey}")
            for dimension in DIMENSIONS:
                dims[dimension] = {
                    "status": "skipped",
                    "skip_reason": "manually skipped after prolonged inference stall",
                }
            save_state(entry, state)
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
        grounding_metadata = None
        gt_hint_entry: dict[str, Any] | None = None
        if _inference_mode in ("single", "gt_hint"):
            try:
                grounding_metadata = build_grounding_frame_metadata(video_path, entry.backend)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                log(f"[unreadable] {entry.label} -> {vkey}: {error_text}")
                for dimension in DIMENSIONS:
                    dims[dimension] = {
                        "status": "skipped",
                        "skip_reason": "video could not be decoded",
                        "error": error_text,
                    }
                video_state["unreadable_video"] = True
                save_state(entry, state)
                continue
            if _inference_mode == "gt_hint":
                gt_hint_entry = _GT_HINT_INDEX.get(vkey)
                if gt_hint_entry is None:
                    log(f"[gt_hint] {entry.label} -> {vkey}: no GT hint entry; skipping")
                    for dimension in ACTIVE_DIMENSIONS:
                        dims[dimension] = {
                            "status": "skipped",
                            "skip_reason": "no GT risk interval hint",
                        }
                    save_state(entry, state)
                    continue
        if _inference_mode == "single":
            active_dimensions = [DIMENSIONS[0]]
        else:
            active_dimensions = ACTIVE_DIMENSIONS
        for dim in active_dimensions:
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
            dim_state["attempt_count"] = int(dim_state.get("attempt_count", 0)) + 1
            dim_state["last_attempt_at"] = now
            dims[dim] = dim_state
            save_state(entry, state)

            base_prompt = prompt_file.read_text(encoding="utf-8")
            if _inference_mode == "single":
                prompt = SINGLE_PROMPT_FILE.read_text(encoding="utf-8")
                prompt = append_grounding_frame_context(prompt, grounding_metadata)
            elif _inference_mode == "gt_hint":
                prompt = append_gt_interval_hint(
                    base_prompt, grounding_metadata, gt_hint_entry.get("time_spans") or []
                )
                prompt = prompt_with_previous_results(prompt, dim, dims)
            elif _inference_mode == "independent":
                prompt = base_prompt
            else:
                prompt = prompt_with_previous_results(base_prompt, dim, dims)
            log(f"[infer] {entry.label} / {dim} -> {vkey}")
            try:
                returncode, stdout, stderr = _run_session_inference(entry, video_path, prompt)
            except Exception:
                import traceback as _traceback
                returncode = 1
                stdout = ""
                stderr = _traceback.format_exc()

            if returncode == 0:
                raw_response = extract_raw_response(stdout)
                dims[dim] = {
                    "status": "success",
                    "attempt_count": int(dim_state.get("attempt_count", 0)),
                    "last_attempt_at": now,
                    "response": raw_response,
                }
                log(f"[success] {entry.label} / {dim} -> {vkey}")
            else:
                dims[dim] = {
                    "status": "error",
                    "attempt_count": int(dim_state.get("attempt_count", 0)),
                    "last_attempt_at": now,
                    "error": (stderr or stdout).strip()[-4000:],
                }
                video_all_done = False
                log(f"[error] {entry.label} / {dim} -> {vkey}: {(stderr or stdout).strip()[:300]}")
            save_state(entry, state)

            if returncode != 0:
                video_all_done = False
                break

        if _inference_mode == "single" and video_all_done:
            single_response = dims.get("perception", {}).get("response", "")
            split_responses = split_single_pass_response(single_response)
            if split_responses is None:
                video_all_done = False
                dims["perception"] = {
                    "status": "error",
                    "attempt_count": int(dims.get("perception", {}).get("attempt_count", 0)),
                    "last_attempt_at": now,
                    "response": single_response,
                    "error": "single-pass response was not a JSON object",
                }
                save_state(entry, state)
            else:
                if grounding_metadata is not None and split_responses.get("grounding"):
                    split_responses["grounding"] = convert_frame_spans_to_time_spans(
                        split_responses["grounding"], grounding_metadata
                    )
                for split_dim, split_response in split_responses.items():
                    dims[split_dim] = {
                        "status": "success",
                        "last_attempt_at": now,
                        "response": split_response,
                    }
                save_state(entry, state)

        if video_all_done and all_dimensions_done(video_state):
            dim_responses = {d: dims[d].get("response", "") for d in ACTIVE_DIMENSIONS}
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
            if grounding_metadata is not None:
                merged_payload["grounding_frame_metadata"] = grounding_metadata
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
    parser.add_argument(
        "--inference-mode",
        choices=("independent", "sequential", "single", "gt_hint"),
        default="sequential",
        help="independent: original four calls; sequential: pass prior results; "
             "single: one call for all fields; "
             "gt_hint: inject the GT risk interval as sampled-frame ids into "
             "perception/cognition/planning prompts and skip grounding.",
    )
    parser.add_argument(
        "--single-prompt-file",
        default=str(SINGLE_PROMPTS_DIR / "infer_prompt_structured.txt"),
        help="Prompt used by single mode. Defaults to evaluation/scripts/prompts/infer_prompt_structured.txt.",
    )
    parser.add_argument(
        "--gt-hint-dir",
        default=str(PROJECT_ROOT / "data" / "public_data_release" / "annotations" / "real_videos"),
        help="Directory containing grounding_gt.json entries used by gt_hint mode "
             "(risk interval hints). Defaults to the released real_videos annotations.",
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

    for dim in ACTIVE_DIMENSIONS:
        prompt_file = PROMPT_FILES[dim]
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if args.inference_mode == "single" and not SINGLE_PROMPT_FILE.exists():
        raise FileNotFoundError(f"Single-pass prompt file not found: {SINGLE_PROMPT_FILE}")

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

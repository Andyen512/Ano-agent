#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from batch_multi_view_risk_review import (
    DEFAULT_OUTPUT_ROOT,
    PUBLIC_DATA_ROOT,
    VIDEO_EXTENSIONS,
    scan_videos,
    task_output_root,
    valid_review_summary,
)
from multi_view_risk_review import (
    MODEL_POOL,
    ModelSpec,
    PerspectiveRule,
    build_prompt,
    build_subprocess_env,
    make_agent_slug,
    parse_keep_decision,
    parse_rule_file,
    summarize_results,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RULE_PATH = SCRIPT_DIR / "rule.md"
DEFAULT_BACKEND_ORDER = ("qwen35vl", "mplugowl3", "tarsier2", "internvl35", "videollama3")
EXTRA_WORKER_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(model_id="openbmb/MiniCPM-V-4_5", backend="minicpmv45"),
)


@dataclass(frozen=True)
class VideoTask:
    index: int
    total: int
    video_path: Path
    relative_path: Path
    output_root: Path


@dataclass
class VideoState:
    task: VideoTask
    assignments: list[tuple[PerspectiveRule, ModelSpec]]
    agent_results: list[dict[str, Any]] = field(default_factory=list)
    scheduled: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)
    failed: bool = False


@dataclass
class WorkerSlot:
    worker_id: str
    backend: str
    gpu_id: str
    input_queue: mp.Queue
    process: mp.Process
    inflight: int = 0


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent multi-model public_data_filter batch runner. Each worker loads one "
            "model once and continuously consumes per-video review jobs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=PUBLIC_DATA_ROOT)
    parser.add_argument("--datasets", nargs="*", help="Optional dataset subdirectories under --data-root.")
    parser.add_argument("--output-root", type=Path, help="Defaults to public_data_filter/batch_outputs/<run-id>.")
    parser.add_argument("--run-id", default=utc_tag())
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--worker-plan",
        default="auto",
        help=(
            "Worker allocation, e.g. qwen35vl:2,mplugowl3:2,tarsier2:2,internvl35:1,videollama3:1. "
            "Default auto allocates one worker per model, then duplicates the slowest models."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-text")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--tarsier-config", default=str(PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-agent-outputs", action="store_true")
    parser.add_argument("--early-stop-majority", action="store_true")
    parser.add_argument(
        "--replace-tarsier2-with-minicpmv45",
        action="store_true",
        help="Use MiniCPM-V-4.5 in the five-model pool instead of Tarsier2.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--max-inflight-videos", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_gpu_ids(raw: str) -> list[str]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not ids:
        raise ValueError("At least one GPU id must be provided via --gpus.")
    return ids


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    return (DEFAULT_OUTPUT_ROOT / args.run_id).resolve()


def model_by_backend() -> dict[str, ModelSpec]:
    return {model.backend: model for model in (*MODEL_POOL, *EXTRA_WORKER_MODELS)}


def parse_worker_plan(raw: str, gpu_ids: list[str], backend_order: tuple[str, ...] = DEFAULT_BACKEND_ORDER) -> list[str]:
    available = model_by_backend()
    if raw != "auto":
        backends: list[str] = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                backend, count_raw = chunk.split(":", 1)
                count = int(count_raw)
            else:
                backend, count = chunk, 1
            backend = backend.strip()
            if backend not in available:
                raise ValueError(f"Unknown backend in --worker-plan: {backend}")
            backends.extend([backend] * count)
        if len(backends) != len(gpu_ids):
            raise ValueError(f"--worker-plan defines {len(backends)} workers but --gpus defines {len(gpu_ids)} GPUs.")
        return backends

    backends = [backend for backend in backend_order if backend in available]
    if len(gpu_ids) < len(backends):
        raise ValueError(f"Need at least {len(backends)} GPUs for the five-model persistent runner.")
    index = 0
    while len(backends) < len(gpu_ids):
        backends.append(backend_order[index % len(backend_order)])
        index += 1
    return backends


def effective_model_pool(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    if not args.replace_tarsier2_with_minicpmv45:
        return MODEL_POOL
    return tuple(
        ModelSpec(model_id="openbmb/MiniCPM-V-4_5", backend="minicpmv45")
        if model.backend == "tarsier2"
        else model
        for model in MODEL_POOL
    )


def effective_backend_order(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.replace_tarsier2_with_minicpmv45:
        return DEFAULT_BACKEND_ORDER
    return tuple("minicpmv45" if backend == "tarsier2" else backend for backend in DEFAULT_BACKEND_ORDER)


def assign_models_from_pool(
    rules: list[PerspectiveRule],
    seed: int | None,
    model_pool: tuple[ModelSpec, ...],
) -> list[tuple[PerspectiveRule, ModelSpec]]:
    if len(rules) > len(model_pool):
        raise ValueError(
            f"Need at least {len(rules)} distinct models for the configured perspectives, "
            f"but only {len(model_pool)} models are available."
        )
    shuffled_models = list(model_pool)
    random.Random(seed).shuffle(shuffled_models)
    return [(rule, model) for rule, model in zip(rules, shuffled_models)]


def build_tasks(args: argparse.Namespace, data_root: Path, output_root: Path) -> tuple[list[VideoTask], list[str]]:
    videos = scan_videos(data_root, args.datasets)
    if args.limit is not None:
        videos = videos[: max(0, args.limit)]
    tasks: list[VideoTask] = []
    skipped: list[str] = []
    total = len(videos)
    for index, video_path in enumerate(videos, start=1):
        review_root = task_output_root(output_root, data_root, video_path)
        summary_path = review_root / "review_summary.json"
        if args.resume and valid_review_summary(summary_path, video_path):
            skipped.append(str(video_path))
            continue
        tasks.append(
            VideoTask(
                index=index,
                total=total,
                video_path=video_path,
                relative_path=video_path.relative_to(data_root),
                output_root=review_root,
            )
        )
    return tasks, skipped


def generation_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "merge_size": args.merge_size,
    }


def load_reusable_agent_result(
    args: argparse.Namespace,
    state: VideoState,
    assignment_index: int,
    prompt_path: Path,
    output_json: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any] | None:
    rule, model = state.assignments[assignment_index]
    if not output_json.exists():
        return None
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("backend") != model.backend:
        return None
    if payload.get("model_id") != model.model_id:
        return None
    if payload.get("video_path") != str(state.task.video_path):
        return None
    expected_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if str(payload.get("prompt", "")).strip() != expected_prompt:
        return None
    generation = payload.get("generation")
    if not isinstance(generation, dict):
        return None
    for key, expected in generation_payload(args).items():
        if generation.get(key) != expected:
            return None
    response = str(payload.get("response", "")).strip()
    if not response:
        return None
    parsed_decision = parse_keep_decision(response)
    if parsed_decision.get("keep") is None:
        return None
    return {
        "perspective": {
            "index": rule.index,
            "name": rule.name,
            "prompt": rule.prompt,
        },
        "model": {
            "model_id": model.model_id,
            "backend": model.backend,
        },
        "prompt_file": str(prompt_path),
        "output_json": str(output_json),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "worker_id": None,
        "gpu_id": None,
        "returncode": 0,
        "elapsed_seconds": 0.0,
        "reused_agent_output": True,
        "response": response,
        "parsed_decision": parsed_decision,
        "runner_payload": payload,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def worker_args_from_payload(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        video_path=Path(payload["video_path"]),
        temperature=payload["temperature"],
        top_p=payload["top_p"],
        max_new_tokens=payload["max_new_tokens"],
        fps=payload["fps"],
        max_frames=payload["max_frames"],
        merge_size=payload["merge_size"],
        attn_implementation=payload["attn_implementation"],
        device_map=payload["device_map"],
        use_flash_attn=payload["use_flash_attn"],
        tarsier_config=payload["tarsier_config"],
        output_json=payload["output_json"],
        model_id=payload["model_id"],
    )


def build_format_retry_prompt(original_prompt: str) -> str:
    return (
        "You must inspect the provided video and classify it using the exact schema below. "
        "Do not write a general video description. Do not add markdown. "
        "Start your answer with 'Risk presence:'.\n\n"
        "Risk presence: Yes / No\n"
        "Level 1 scene: dining room / kitchen / study / balcony / living room / bathroom / yard / other concise scene keyword\n"
        "Level 2 subject: child / older adult / young adult / middle-aged adult / other concise subject keyword\n"
        "Level 3 risk type: fall/instability / heat/fire source / collision/crush injury / sharp-object danger / electrical safety / poisoning/accidental ingestion / interpersonal conflict / other concise risk keyword / None\n"
        "Risk description: if Risk presence is Yes, 1-3 sentences; otherwise None\n"
        "Risk time interval: if Risk presence is Yes, [start,end] in seconds or Unknown; otherwise None\n"
        "Solution for person: if Risk presence is Yes, concrete action; otherwise None\n"
        "Solution for hazard source: if Risk presence is Yes, concrete action; otherwise None\n"
        "Solution to prevent recurrence: if Risk presence is Yes, prevention measure; otherwise None\n"
        "Normal video description: if Risk presence is No, 1-3 sentences; otherwise None\n\n"
        "Use the review perspective and criteria from the original task below, but output only the schema above.\n\n"
        f"Original task:\n{original_prompt}"
    )


class Tarsier2Engine:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        import yaml

        sys.path.insert(0, str(PROJECT_ROOT / "code" / "Tarsier2-7B"))
        tarsier_vendor = PROJECT_ROOT / ".vendor" / "tarsier_flashattn"
        if tarsier_vendor.exists():
            sys.path.insert(0, str(tarsier_vendor))
        from lifebench_infer import module_available
        from tasks.utils import load_model_and_processor

        if not module_available("flash_attn"):
            raise RuntimeError("Tarsier2 requires flash_attn in this checkout.")
        frame_budget = max(1, min(args.max_frames, 32))
        data_config = yaml.safe_load(Path(args.tarsier_config).read_text(encoding="utf-8"))
        data_config["n_frames"] = frame_budget
        data_config["max_n_frames"] = frame_budget
        video_sampling_strategy = dict(data_config.get("video_sampling_strategy") or {})
        video_sampling_strategy["use_multi_images_for_video"] = False
        data_config["video_sampling_strategy"] = video_sampling_strategy
        self.model, self.processor = load_model_and_processor(str(model_path), data_config=data_config)

    def infer(self, args: SimpleNamespace, prompt: str) -> str:
        from tasks.inference_quick_start import process_one

        generate_kwargs = {
            "do_sample": args.temperature > 0,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "use_cache": True,
        }
        return process_one(self.model, self.processor, prompt, str(args.video_path), generate_kwargs).strip()


class VideoLLaMA3Engine:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        sys.path.insert(0, str(PROJECT_ROOT / "code" / "VideoLLaMA3-7B"))
        from lifebench_infer import choose_attn_implementation
        from videollama3 import disable_torch_init

        disable_torch_init()
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "max_tokens"):
            self.processor.image_processor.max_tokens = 512
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            attn_implementation=choose_attn_implementation(args.attn_implementation),
        )

    def infer(self, args: SimpleNamespace, prompt: str) -> str:
        import torch
        from lifebench_infer import controlled_video_fps, output_looks_invalid

        effective_max_frames = min(args.max_frames, 32)
        effective_fps = controlled_video_fps(str(args.video_path), frame_budget=effective_max_frames, fps_cap=1.0)
        conversation = [
            {"role": "system", "content": "You are a careful video safety assessor."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": str(args.video_path.resolve()),
                            "fps": effective_fps,
                            "max_frames": effective_max_frames,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        target_device = "cuda" if args.device_map == "auto" else args.device_map
        inputs = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        generated_ids = self.model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[1]
        decode_ids = generated_ids[:, input_len:] if generated_ids.shape[1] > input_len else generated_ids
        response = self.processor.batch_decode(
            decode_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if output_looks_invalid(response):
            raise RuntimeError(f"VideoLLaMA3 produced an invalid response: {response[:120]!r}")
        return response


class InternVL35Engine:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        from lifebench_infer import load_internvl_model

        self.model, self.tokenizer, self.target_device = load_internvl_model(model_path, args)

    def infer(self, args: SimpleNamespace, prompt: str) -> str:
        import torch
        from lifebench_infer import load_internvl_video, normalize_model_output

        frame_count = max(1, min(args.max_frames, 32))
        pixel_values, num_patches_list = load_internvl_video(
            str(args.video_path),
            num_segments=frame_count,
            max_num=1,
        )
        pixel_values = pixel_values.to(torch.bfloat16).to(self.target_device)
        frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_config["temperature"] = args.temperature
            generation_config["top_p"] = args.top_p
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            frame_prefix + prompt,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        if isinstance(response, tuple):
            response = response[0]
        return normalize_model_output(response)


def install_qwen2_attention_compat() -> None:
    import torch
    import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
    from transformers.cache_utils import DynamicCache

    class CompatQwen2Attention(qwen2_modeling.Qwen2Attention):
        def __init__(self, config, layer_idx: int):
            super().__init__(config, layer_idx)
            self.rotary_emb = qwen2_modeling.Qwen2RotaryEmbedding(config)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ):
            if position_ids is None:
                seq_len = hidden_states.shape[1]
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            attn_output, attn_weights = super().forward(
                hidden_states=hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                output_attentions=output_attentions,
                **kwargs,
            )
            return attn_output, attn_weights, past_key_value if use_cache else None

    qwen2_modeling.Qwen2SdpaAttention = CompatQwen2Attention
    qwen2_modeling.Qwen2FlashAttention2 = CompatQwen2Attention
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


class MplugOwl3Engine:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        import torch
        from transformers import AutoModel, AutoTokenizer

        sys.path.insert(0, str(PROJECT_ROOT / "code" / "mPLUG-Owl3-7B" / "mPLUG-Owl3"))
        install_qwen2_attention_compat()
        self.target_device = "cuda" if args.device_map == "auto" else args.device_map
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            str(model_path),
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        self.model.eval()
        self.processor = self.model.init_processor(self.tokenizer)

    def infer(self, args: SimpleNamespace, prompt: str) -> str:
        from lifebench_infer import load_video_frames_simple, normalize_model_output

        video_frames = [load_video_frames_simple(str(args.video_path), max_num_frames=min(args.max_frames, 32))]
        messages = [
            {"role": "user", "content": f"<|video|>\n{prompt}"},
            {"role": "assistant", "content": ""},
        ]
        inputs = self.processor(messages, images=None, videos=video_frames)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.target_device)
        inputs.update(
            {
                "tokenizer": self.tokenizer,
                "max_new_tokens": args.max_new_tokens,
                "decode_text": True,
            }
        )
        output = self.model.generate(**inputs)
        return normalize_model_output(output)


class Qwen35VLEngine:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        import importlib.metadata as importlib_metadata
        import importlib.util

        import torch

        vendor_dir = PROJECT_ROOT / ".vendor" / "qwen35_shim"
        if vendor_dir.exists():
            sys.path.insert(0, str(vendor_dir))

        original_find_spec = importlib.util.find_spec
        original_metadata = importlib_metadata.metadata

        def patched_find_spec(name: str, *fn_args: Any, **fn_kwargs: Any):
            if name == "deepspeed":
                return None
            return original_find_spec(name, *fn_args, **fn_kwargs)

        def patched_metadata(name: str, *fn_args: Any, **fn_kwargs: Any):
            if name == "deepspeed":
                raise importlib_metadata.PackageNotFoundError(name)
            return original_metadata(name, *fn_args, **fn_kwargs)

        importlib.util.find_spec = patched_find_spec
        importlib_metadata.metadata = patched_metadata
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        finally:
            importlib.util.find_spec = original_find_spec
            importlib_metadata.metadata = original_metadata

        self.target_device = "cuda" if args.device_map == "auto" else args.device_map
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        if hasattr(self.processor, "video_processor") and hasattr(self.processor.video_processor, "fps"):
            self.processor.video_processor.fps = None
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.target_device)

    def infer(self, args: SimpleNamespace, prompt: str) -> str:
        from lifebench_infer import normalize_model_output, strip_qwen_thinking_content

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(args.video_path.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            processor_kwargs={"num_frames": max(1, min(args.max_frames, 32))},
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = inputs.to(self.target_device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p
        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[1]
        decode_ids = generated_ids[:, input_len:] if generated_ids.shape[1] > input_len else generated_ids
        response = self.processor.batch_decode(
            decode_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return strip_qwen_thinking_content(normalize_model_output(response))


class MiniCPMV45Engine:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        from lifebench_infer import load_minicpmv45_model

        self.model, self.tokenizer, self.target_device = load_minicpmv45_model(model_path, args)

    def infer(self, args: SimpleNamespace, prompt: str) -> str:
        from lifebench_infer import minicpmv45_chat

        return minicpmv45_chat(self.model, self.tokenizer, args, prompt)


ENGINE_BY_BACKEND = {
    "tarsier2": Tarsier2Engine,
    "videollama3": VideoLLaMA3Engine,
    "internvl35": InternVL35Engine,
    "mplugowl3": MplugOwl3Engine,
    "qwen35vl": Qwen35VLEngine,
    "minicpmv45": MiniCPMV45Engine,
}


def worker_main(
    worker_id: str,
    gpu_id: str,
    backend: str,
    model_id: str,
    args_payload: dict[str, Any],
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    log_path: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(EVALUATION_DIR))
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        from lifebench_infer import BACKENDS, build_result, resolve_model_path

        args = SimpleNamespace(**args_payload)
        model_path = resolve_model_path(None, model_id)
        engine = ENGINE_BY_BACKEND[backend](model_path, args)
        output_queue.put({"type": "ready", "worker_id": worker_id, "backend": backend, "gpu_id": gpu_id})
        while True:
            job = input_queue.get()
            if job is None:
                return
            started_at = time.time()
            infer_args = worker_args_from_payload(job)
            try:
                response = engine.infer(infer_args, job["prompt"])
                format_retry_used = False
                first_response = None
                if parse_keep_decision(response).get("keep") is None:
                    first_response = response
                    format_retry_used = True
                    response = engine.infer(infer_args, build_format_retry_prompt(job["prompt"]))
                elapsed = time.time() - started_at
                result = build_result(
                    infer_args,
                    BACKENDS[backend],
                    model_id,
                    model_path,
                    job["prompt"],
                    response,
                    elapsed,
                )
                if format_retry_used:
                    result["format_retry_used"] = True
                    result["first_response_before_format_retry"] = first_response
                output_json = Path(job["output_json"])
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                output_queue.put(
                    {
                        "type": "result",
                        "task_key": job["task_key"],
                        "assignment_index": job["assignment_index"],
                        "worker_id": worker_id,
                        "backend": backend,
                        "gpu_id": gpu_id,
                        "returncode": 0,
                        "elapsed_seconds": round(elapsed, 3),
                        "runner_payload": result,
                    }
                )
            except Exception as exc:
                output_queue.put(
                    {
                        "type": "result",
                        "task_key": job["task_key"],
                        "assignment_index": job["assignment_index"],
                        "worker_id": worker_id,
                        "backend": backend,
                        "gpu_id": gpu_id,
                        "returncode": 1,
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "error": str(exc),
                        "traceback": traceback.format_exc()[-4000:],
                    }
                )
    except Exception as exc:
        output_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "backend": backend,
                "gpu_id": gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
        )
    finally:
        log_file.close()


def agent_paths(run_dir: Path, rule: PerspectiveRule) -> tuple[Path, Path, Path, Path]:
    agent_slug = make_agent_slug(rule)
    prompt_path = run_dir / "prompts" / f"{agent_slug}.txt"
    output_json = run_dir / "agent_outputs" / f"{agent_slug}.json"
    stdout_path = run_dir / "logs" / f"{agent_slug}.stdout.log"
    stderr_path = run_dir / "logs" / f"{agent_slug}.stderr.log"
    return prompt_path, output_json, stdout_path, stderr_path


def build_agent_result(
    state: VideoState,
    assignment_index: int,
    message: dict[str, Any],
    prompt_path: Path,
    output_json: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    rule, model = state.assignments[assignment_index]
    base = {
        "perspective": {
            "index": rule.index,
            "name": rule.name,
            "prompt": rule.prompt,
        },
        "model": {
            "model_id": model.model_id,
            "backend": model.backend,
        },
        "prompt_file": str(prompt_path),
        "output_json": str(output_json),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "worker_id": message.get("worker_id"),
        "gpu_id": message.get("gpu_id"),
        "returncode": message["returncode"],
        "elapsed_seconds": message["elapsed_seconds"],
    }
    if message["returncode"] != 0:
        base["error"] = {
            "message": message.get("error", "Worker inference failed."),
            "traceback_tail": message.get("traceback", ""),
        }
        return base
    payload = message["runner_payload"]
    response = str(payload.get("response", "")).strip()
    base["response"] = response
    base["parsed_decision"] = parse_keep_decision(response)
    base["runner_payload"] = payload
    return base


def write_video_summary(
    args: argparse.Namespace,
    state: VideoState,
    started_at: float,
    model_pool: tuple[ModelSpec, ...],
) -> dict[str, Any]:
    manifest = {
        "video_path": str(state.task.video_path),
        "rule_path": str(RULE_PATH),
        "prompt_language": "en",
        "output_schema_version": "risk_full_en_v2",
        "candidate_text": args.candidate_text,
        "seed": args.seed,
        "persistent_runner": True,
        "reuse_agent_outputs": bool(args.reuse_agent_outputs),
        "early_stop_majority": bool(args.early_stop_majority),
        "model_pool": [
            {
                "model_id": item.model_id,
                "backend": item.backend,
            }
            for item in model_pool
        ],
        "assignments": [
            {
                "perspective": {
                    "index": rule.index,
                    "name": rule.name,
                    "prompt": rule.prompt,
                },
                "model": {
                    "model_id": model.model_id,
                    "backend": model.backend,
                },
            }
            for rule, model in state.assignments
        ],
    }
    state.agent_results.sort(key=lambda item: item["perspective"]["index"])
    final_payload = {
        "manifest": manifest,
        "agent_results": state.agent_results,
        "majority_vote": summarize_results(state.agent_results, expected_votes=len(state.assignments)),
    }
    write_json(state.task.output_root / "assignment_manifest.json", manifest)
    write_json(state.task.output_root / "review_summary.json", final_payload)
    return {
        "video_path": str(state.task.video_path),
        "video_relpath": str(state.task.relative_path),
        "output_root": str(state.task.output_root),
        "summary_json": str(state.task.output_root / "review_summary.json"),
        "majority_vote": final_payload["majority_vote"],
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


def can_stop(args: argparse.Namespace, state: VideoState) -> bool:
    if state.failed and not args.keep_going:
        return True
    if len(state.completed) >= len(state.assignments):
        return True
    if args.early_stop_majority:
        summary = summarize_results(state.agent_results, expected_votes=len(state.assignments))
        return summary.get("complete") is True
    return False


def next_assignment_indices(args: argparse.Namespace, state: VideoState) -> list[int]:
    unscheduled = [idx for idx in range(len(state.assignments)) if idx not in state.scheduled]
    if not unscheduled:
        return []
    if not args.early_stop_majority:
        return unscheduled
    if not state.scheduled:
        return unscheduled[:3]
    if all(idx in state.completed for idx in state.scheduled):
        return unscheduled[:1]
    return []


def choose_worker(workers: list[WorkerSlot], backend: str) -> WorkerSlot:
    candidates = [worker for worker in workers if worker.backend == backend]
    if not candidates:
        raise RuntimeError(f"No persistent worker available for backend {backend}")
    return min(candidates, key=lambda worker: worker.inflight)


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


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")
    if not RULE_PATH.exists():
        raise FileNotFoundError(f"Rule file does not exist: {RULE_PATH}")

    gpu_ids = parse_gpu_ids(args.gpus)
    model_pool = effective_model_pool(args)
    backend_order = effective_backend_order(args)
    worker_backends = parse_worker_plan(args.worker_plan, gpu_ids, backend_order=backend_order)
    output_root = resolve_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks, skipped = build_tasks(args, data_root, output_root)
    rules = parse_rule_file(RULE_PATH)
    assignments = assign_models_from_pool(rules, args.seed, model_pool)
    available_backends = {model.backend for _, model in assignments}
    for backend in available_backends:
        if backend not in worker_backends:
            raise ValueError(f"Worker plan does not include required backend: {backend}")

    manifest = {
        "run_id": args.run_id,
        "data_root": str(data_root),
        "datasets": args.datasets or [],
        "output_root": str(output_root),
        "prompt_language": "en",
        "output_schema_version": "risk_full_en_v2",
        "gpu_ids": gpu_ids,
        "worker_backends": worker_backends,
        "worker_plan": dict(Counter(worker_backends)),
        "seed": args.seed,
        "task_count": len(tasks),
        "skipped_existing_count": len(skipped),
        "reuse_agent_outputs": bool(args.reuse_agent_outputs),
        "early_stop_majority": bool(args.early_stop_majority),
        "replace_tarsier2_with_minicpmv45": bool(args.replace_tarsier2_with_minicpmv45),
        "model_pool": [
            {
                "model_id": item.model_id,
                "backend": item.backend,
            }
            for item in model_pool
        ],
        "persistent_runner": True,
        "dry_run": bool(args.dry_run),
    }
    write_json(output_root / "batch_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    # Match the single-video subprocess environment. In this conda env, torch may
    # need libittnotify preloaded before a spawned worker imports it.
    os.environ.update(build_subprocess_env())

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
        "tarsier_config": args.tarsier_config,
    }
    backend_models = model_by_backend()
    workers: list[WorkerSlot] = []
    worker_logs = output_root / "worker_logs"
    worker_logs.mkdir(parents=True, exist_ok=True)
    for index, (gpu_id, backend) in enumerate(zip(gpu_ids, worker_backends), start=1):
        input_queue: mp.Queue = ctx.Queue()
        worker_id = f"{index:02d}_{backend}_gpu{gpu_id}"
        process = ctx.Process(
            target=worker_main,
            args=(
                worker_id,
                gpu_id,
                backend,
                backend_models[backend].model_id,
                args_payload,
                input_queue,
                output_queue,
                str(worker_logs / f"{worker_id}.log"),
            ),
        )
        process.start()
        workers.append(
            WorkerSlot(
                worker_id=worker_id,
                backend=backend,
                gpu_id=gpu_id,
                input_queue=input_queue,
                process=process,
            )
        )

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
    active: dict[str, VideoState] = {}
    started_at: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    completed_count = 0

    def add_active_tasks() -> None:
        while pending and len(active) < max(1, args.max_inflight_videos):
            task = pending.popleft()
            key = str(task.relative_path)
            active[key] = VideoState(task=task, assignments=list(assignments))
            started_at[key] = time.time()

    def schedule_state(key: str, state: VideoState) -> None:
        if can_stop(args, state):
            return
        for assignment_index in next_assignment_indices(args, state):
            rule, model = state.assignments[assignment_index]
            prompt = build_prompt(rule, args.candidate_text)
            prompt_path, output_json, stdout_path, stderr_path = agent_paths(state.task.output_root, rule)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            if args.reuse_agent_outputs:
                reused = load_reusable_agent_result(args, state, assignment_index, prompt_path, output_json, stdout_path, stderr_path)
                if reused is not None:
                    state.agent_results.append(reused)
                    state.scheduled.add(assignment_index)
                    state.completed.add(assignment_index)
                    continue
            worker = choose_worker(workers, model.backend)
            job = {
                "task_key": key,
                "assignment_index": assignment_index,
                "video_path": str(state.task.video_path),
                "prompt": prompt,
                "prompt_file": str(prompt_path),
                "output_json": str(output_json),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "model_id": model.model_id,
                "backend": model.backend,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "fps": args.fps,
                "max_frames": args.max_frames,
                "merge_size": args.merge_size,
                "attn_implementation": args.attn_implementation,
                "device_map": args.device_map,
                "use_flash_attn": args.use_flash_attn,
                "tarsier_config": args.tarsier_config,
            }
            state.scheduled.add(assignment_index)
            worker.inflight += 1
            worker.input_queue.put(job)

    def schedule_all_ready() -> None:
        add_active_tasks()
        for key, state in list(active.items()):
            schedule_state(key, state)

    try:
        schedule_all_ready()
        while active:
            finished_now: list[str] = []
            for key, state in list(active.items()):
                if can_stop(args, state):
                    row = write_video_summary(args, state, started_at[key], model_pool)
                    rows.append(row)
                    finished_now.append(key)
            for key in finished_now:
                del active[key]
                del started_at[key]
                completed_count += 1
                if args.log_every > 0 and (completed_count % args.log_every == 0 or completed_count == len(tasks)):
                    last = rows[-1]
                    majority = last.get("majority_vote", {})
                    status = "keep" if majority.get("keep") is True else "drop" if majority.get("keep") is False else "unknown"
                    print(
                        f"[{completed_count}/{len(tasks)}] status={status} votes={majority.get('votes_cast')} {last['video_relpath']}",
                        file=sys.stderr,
                        flush=True,
                    )
            schedule_all_ready()
            if not active:
                continue

            try:
                message = output_queue.get(timeout=1.0)
            except queue.Empty:
                dead = [worker.worker_id for worker in workers if not worker.process.is_alive()]
                if dead:
                    raise RuntimeError(f"Worker process exited unexpectedly: {dead}")
                continue

            if message.get("type") == "worker_error":
                raise RuntimeError(
                    f"Worker {message['worker_id']} failed: {message.get('error')}\n{message.get('traceback', '')}"
                )
            if message.get("type") != "result":
                continue
            worker_id = message["worker_id"]
            for worker in workers:
                if worker.worker_id == worker_id:
                    worker.inflight = max(0, worker.inflight - 1)
                    break
            key = message["task_key"]
            state = active.get(key)
            if state is None:
                continue
            assignment_index = message["assignment_index"]
            rule, _model = state.assignments[assignment_index]
            prompt_path, output_json, stdout_path, stderr_path = agent_paths(state.task.output_root, rule)
            state.agent_results.append(
                build_agent_result(state, assignment_index, message, prompt_path, output_json, stdout_path, stderr_path)
            )
            state.completed.add(assignment_index)
            if message["returncode"] != 0:
                state.failed = True
                if not args.keep_going:
                    raise RuntimeError(
                        f"Agent failed for {state.task.relative_path}: {message.get('error')}\n{message.get('traceback', '')}"
                    )
            schedule_state(key, state)
    finally:
        for worker in workers:
            worker.input_queue.put(None)
        for worker in workers:
            worker.process.join(timeout=30)
            if worker.process.is_alive():
                worker.process.terminate()

    rows.sort(key=lambda item: item["video_relpath"])
    summary = {
        "manifest": manifest,
        "summary": summarize_batch(rows),
        "skipped_existing": skipped,
        "results": rows,
    }
    write_json(output_root / "batch_review_summary.json", summary)
    write_csv(output_root / "batch_review_summary.csv", rows)
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import importlib.util
import io
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from lifebench_infer import (
    BACKENDS,
    build_result,
    choose_attn_implementation,
    compact_structured_prompt_for_videochatgpt,
    controlled_video_fps,
    ensure_backend_on_path,
    load_prompt,
    load_internvl_model,
    load_internvl_video,
    load_video_frames_simple,
    module_available,
    normalize_model_output,
    output_looks_invalid,
    resolve_model_path,
    resolve_videochatgpt_base_model,
    strip_qwen_thinking_content,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a LifeBench backend once and run it over multiple videos.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--model-path")
    parser.add_argument("--model-id")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="auto")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument(
        "--tarsier-config",
        default=str(PROJECT_ROOT / "code" / "Tarsier2-7B" / "configs" / "tarser2_default_config.yaml"),
    )
    parser.add_argument("--projection-path")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos that already have output JSON files")
    return parser


def validate_task_manifest(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task manifest entry must be an object.")
        required = ("video_path", "output_json", "stdout_log", "stderr_log")
        missing = [key for key in required if not isinstance(task.get(key), str) or not str(task.get(key)).strip()]
        if missing:
            raise ValueError(f"Task manifest entry missing required fields: {missing}")
        validated.append({key: str(task[key]) for key in required})
    return validated


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clone_args_for_video(args: argparse.Namespace, video_path: str) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["video_path"] = video_path
    return argparse.Namespace(**payload)


@dataclass
class SessionBundle:
    backend_name: str
    model_id: str
    model_path: Path | None
    session: Any


class Tarsier2Session:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        backend = BACKENDS["tarsier2"]
        ensure_backend_on_path(backend)

        tarsier_vendor = PROJECT_ROOT / ".vendor" / "tarsier_flashattn"
        if (
            os.environ.get("LIFEBENCH_TARSIER_ATTN_IMPLEMENTATION", "eager") == "flash_attention_2"
            and tarsier_vendor.exists()
        ):
            vendor_path = str(tarsier_vendor)
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)

        from tasks.inference_quick_start import process_one
        from tasks.utils import load_model_and_processor
        import yaml

        data_config = yaml.safe_load(Path(args.tarsier_config).read_text(encoding="utf-8"))
        data_config["max_pixels"] = int(os.environ.get("LIFEBENCH_TARSIER_MAX_PIXELS", "50176"))
        self.model, self.processor = load_model_and_processor(str(model_path), data_config=data_config)
        self.process_one = process_one
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        generate_kwargs = {
            "do_sample": self.args.temperature > 0,
            "max_new_tokens": self.args.max_new_tokens,
            "top_p": self.args.top_p,
            "temperature": self.args.temperature,
            "use_cache": True,
        }
        return self.process_one(self.model, self.processor, prompt, video_path, generate_kwargs).strip()


class VideoLLaMA2Session:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        backend = BACKENDS["videollama2"]
        ensure_backend_on_path(backend)

        from videollama2 import mm_infer, model_init
        from videollama2.utils import disable_torch_init

        if args.use_flash_attn and not module_available("flash_attn"):
            raise RuntimeError("VideoLLaMA2 inference requested flash_attn, but the package is not installed.")

        disable_torch_init()
        self.model, self.processor, self.tokenizer = model_init(
            str(model_path),
            device_map=args.device_map,
            use_flash_attn=args.use_flash_attn and module_available("flash_attn"),
        )
        self.mm_infer = mm_infer
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        video_tensor = self.processor["video"](video_path)
        return self.mm_infer(
            video_tensor,
            prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            modal="video",
            do_sample=self.args.temperature > 0,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_new_tokens=self.args.max_new_tokens,
        ).strip()


class VideoLLaMA3Session:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        backend = BACKENDS["videollama3"]
        ensure_backend_on_path(backend)

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from videollama3 import disable_torch_init

        self.torch = torch
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
        self.target_device = "cuda" if args.device_map == "auto" else args.device_map
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        effective_max_frames = min(self.args.max_frames, 32)
        effective_fps = controlled_video_fps(
            video_path,
            frame_budget=effective_max_frames,
            fps_cap=1.0,
        )
        conversation = [
            {"role": "system", "content": "You are a careful video safety assessor."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": str(Path(video_path).expanduser().resolve()),
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
        inputs = {k: (v.to(self.target_device) if isinstance(v, self.torch.Tensor) else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], self.torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch.bfloat16)
        generated_ids = self.model.generate(
            **inputs,
            do_sample=self.args.temperature > 0,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_new_tokens=self.args.max_new_tokens,
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


class VideoCCAMSession:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        backend = BACKENDS["videoccam"]
        ensure_backend_on_path(backend)

        import torch
        from eval.utils import load_decord
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        self.torch = torch
        if args.device_map == "auto":
            device_map: str | dict[str, str] = {"": "cuda:0"}
        else:
            device_map = args.device_map
        self.load_decord = load_decord
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.image_processor = AutoImageProcessor.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=self.torch.bfloat16,
            device_map=device_map,
            attn_implementation=choose_attn_implementation(args.attn_implementation),
        )
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        messages = [[{"role": "user", "content": f"<video>\n{prompt}"}]]
        images = [self.load_decord(video_path, sample_type="uniform", num_frames=min(self.args.max_frames, 32))]
        response = self.model.chat(
            messages,
            images,
            self.tokenizer,
            self.image_processor,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=self.args.temperature > 0,
        )
        return normalize_model_output(response)


class InternVL35Session:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        import torch

        self.model, self.tokenizer, self.target_device = load_internvl_model(model_path, args)
        self.torch = torch
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        frame_count = max(1, min(self.args.max_frames, 32))
        pixel_values, num_patches_list = load_internvl_video(
            video_path,
            num_segments=frame_count,
            max_num=1,
        )
        pixel_values = pixel_values.to(self.torch.bfloat16).to(self.target_device)
        frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        generation_config = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            generation_config["temperature"] = self.args.temperature
            generation_config["top_p"] = self.args.top_p
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


class MPlugOwl3Session:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        backend = BACKENDS["mplugowl3"]
        ensure_backend_on_path(backend)

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            str(model_path),
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=self.torch.bfloat16,
            device_map=args.device_map,
        )
        self.model.eval()
        self.processor = self.model.init_processor(self.tokenizer)
        self.target_device = "cuda" if args.device_map == "auto" else args.device_map
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        video_frames = [load_video_frames_simple(video_path, max_num_frames=min(self.args.max_frames, 32))]
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
                "max_new_tokens": self.args.max_new_tokens,
                "decode_text": True,
            }
        )
        output = self.model.generate(**inputs)
        return normalize_model_output(output)


class VideoChatGPTSession:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        backend = BACKENDS["videochatgpt"]
        ensure_backend_on_path(backend)

        from video_chatgpt.eval.model_utils import initialize_model, load_video
        from video_chatgpt.inference import video_chatgpt_infer

        base_model_path = model_path
        projection_path = args.projection_path
        if not (model_path / "config.json").exists() and (model_path / "video_chatgpt-7B.bin").exists():
            base_model_path = resolve_videochatgpt_base_model(model_path)
            if projection_path is None:
                projection_path = str(model_path / "video_chatgpt-7B.bin")

        self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(
            str(base_model_path),
            projection_path,
        )
        self.load_video = load_video
        self.video_chatgpt_infer = video_chatgpt_infer
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        compact_prompt = compact_structured_prompt_for_videochatgpt(prompt)
        video_frames = self.load_video(video_path, num_frm=min(self.args.max_frames, 32))
        response = self.video_chatgpt_infer(
            video_frames,
            compact_prompt,
            "video-chatgpt_v1",
            self.model,
            self.vision_tower,
            self.tokenizer,
            self.image_processor,
            self.video_token_len,
            do_sample=self.args.temperature > 0,
            temperature=max(self.args.temperature, 0.0),
            max_new_tokens=min(self.args.max_new_tokens, 320),
        ).strip()
        if output_looks_invalid(response):
            raise RuntimeError(f"Video-ChatGPT produced an invalid response: {response[:120]!r}")
        return response


class Qwen25VLSession:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        import torch

        if not module_available("qwen_vl_utils"):
            raise RuntimeError("Qwen2.5-VL inference requires qwen_vl_utils, but it is not installed.")

        vendor_dir = PROJECT_ROOT / ".vendor" / "qwen251_shim"
        if vendor_dir.exists():
            vendor_path = str(vendor_dir)
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)

        from qwen_vl_utils import process_vision_info

        original_find_spec = importlib.util.find_spec
        original_metadata = importlib_metadata.metadata

        def patched_find_spec(name: str, *f_args: Any, **f_kwargs: Any):
            if name == "deepspeed":
                return None
            return original_find_spec(name, *f_args, **f_kwargs)

        def patched_metadata(name: str, *f_args: Any, **f_kwargs: Any):
            if name == "deepspeed":
                raise importlib_metadata.PackageNotFoundError(name)
            return original_metadata(name, *f_args, **f_kwargs)

        importlib.util.find_spec = patched_find_spec
        importlib_metadata.metadata = patched_metadata
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        finally:
            importlib.util.find_spec = original_find_spec
            importlib_metadata.metadata = original_metadata

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=self.torch.bfloat16,
        )
        self.target_device = "cuda" if args.device_map == "auto" else args.device_map
        self.model = self.model.to(self.target_device)
        self.process_vision_info = process_vision_info
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        frame_budget = min(self.args.max_frames, 32)
        effective_fps = controlled_video_fps(video_path, frame_budget=frame_budget, fps_cap=1.0)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{Path(video_path).expanduser().resolve()}",
                        "fps": effective_fps,
                        "max_pixels": 360 * 420,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.target_device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return normalize_model_output(response)


class Qwen35VLSession:
    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        import torch

        vendor_dir = PROJECT_ROOT / ".vendor" / "qwen35_shim"
        if vendor_dir.exists():
            vendor_path = str(vendor_dir)
            if vendor_path not in sys.path:
                sys.path.insert(0, vendor_path)

        original_find_spec = importlib.util.find_spec
        original_metadata = importlib_metadata.metadata

        def patched_find_spec(name: str, *f_args: Any, **f_kwargs: Any):
            if name == "deepspeed":
                return None
            return original_find_spec(name, *f_args, **f_kwargs)

        def patched_metadata(name: str, *f_args: Any, **f_kwargs: Any):
            if name == "deepspeed":
                raise importlib_metadata.PackageNotFoundError(name)
            return original_metadata(name, *f_args, **f_kwargs)

        importlib.util.find_spec = patched_find_spec
        importlib_metadata.metadata = patched_metadata
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        finally:
            importlib.util.find_spec = original_find_spec
            importlib_metadata.metadata = original_metadata

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        if hasattr(self.processor, "video_processor") and hasattr(self.processor.video_processor, "fps"):
            self.processor.video_processor.fps = None
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=self.torch.bfloat16,
        )
        self.target_device = "cuda" if args.device_map == "auto" else args.device_map
        self.model = self.model.to(self.target_device)
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(Path(video_path).expanduser().resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            processor_kwargs={"num_frames": max(1, min(self.args.max_frames, 32))},
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.target_device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "use_cache": True,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[1]
        decode_ids = generated_ids[:, input_len:] if generated_ids.shape[1] > input_len else generated_ids
        response = self.processor.batch_decode(
            decode_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return strip_qwen_thinking_content(normalize_model_output(response))


class CommercialSession:
    def __init__(self, args: argparse.Namespace) -> None:
        from llm import chat_completion, image_to_data_url

        self.chat_completion = chat_completion
        self.image_to_data_url = image_to_data_url
        self.args = args

    def generate(self, video_path: str, prompt: str) -> str:
        frame_budget = max(1, min(self.args.max_frames, 12))
        video_frames = load_video_frames_simple(video_path, max_num_frames=frame_budget)
        if not video_frames:
            raise RuntimeError(f"Unable to decode video frames for commercial backend: {video_path}")

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "These sampled frames are ordered chronologically from the same video. Use them to infer the full scene dynamics and answer the task below.",
            }
        ]
        for index, frame in enumerate(video_frames, start=1):
            content.append({"type": "text", "text": f"Frame {index}/{len(video_frames)}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.image_to_data_url(frame, format="JPEG", quality=85),
                        "detail": "low",
                    },
                }
            )
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": "You are a careful video safety assessor."},
            {"role": "user", "content": content},
        ]
        return self.chat_completion(
            messages,
            model=self.args.model_id or BACKENDS["commercial"].default_model_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.max_new_tokens,
        ).strip()


def create_session_bundle(args: argparse.Namespace) -> SessionBundle:
    backend = BACKENDS[args.backend]
    model_id = args.model_id or backend.default_model_id
    model_path = resolve_model_path(args.model_path, model_id) if backend.requires_local_model else None

    if args.backend == "tarsier2":
        session = Tarsier2Session(model_path, args)
    elif args.backend == "videollama2":
        session = VideoLLaMA2Session(model_path, args)
    elif args.backend == "videollama3":
        session = VideoLLaMA3Session(model_path, args)
    elif args.backend == "videoccam":
        session = VideoCCAMSession(model_path, args)
    elif args.backend == "internvl35":
        session = InternVL35Session(model_path, args)
    elif args.backend == "mplugowl3":
        session = MPlugOwl3Session(model_path, args)
    elif args.backend == "videollava":
        from compat.videollava_once import VideoLlavaSession

        session = VideoLlavaSession(str(model_path))
    elif args.backend == "videochat2":
        from compat.videochat2_once import VideoChat2Session

        session = VideoChat2Session(str(model_path))
    elif args.backend == "videochatgpt":
        session = VideoChatGPTSession(model_path, args)
    elif args.backend == "minigpt4video":
        from compat.minigpt4_video_once import MiniGPT4VideoSession

        session = MiniGPT4VideoSession()
    elif args.backend == "qwen25vl":
        session = Qwen25VLSession(model_path, args)
    elif args.backend == "qwen35vl":
        session = Qwen35VLSession(model_path, args)
    elif args.backend == "commercial":
        session = CommercialSession(args)
    else:
        raise KeyError(f"Unsupported backend: {args.backend}")
    return SessionBundle(args.backend, model_id, model_path, session)


def run_one_task(
    task: dict[str, str],
    args: argparse.Namespace,
    prompt: str,
    bundle: SessionBundle,
) -> dict[str, Any]:
    video_path = task["video_path"]
    output_json = Path(task["output_json"]).expanduser()
    stdout_log = Path(task["stdout_log"]).expanduser()
    stderr_log = Path(task["stderr_log"]).expanduser()

    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if getattr(args, "skip_existing", False) and (output_json.exists() or output_json.is_symlink()):
        return {
            "video": video_path,
            "video_id": Path(video_path).stem,
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "returncode": 0,
            "output_json": str(output_json),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "stdout_tail": "",
            "stderr_tail": "",
            "skipped": True,
        }

    if output_json.exists() or output_json.is_symlink():
        output_json.unlink()

    started_at = now_iso()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    response = ""
    returncode = 0
    started_clock = time.time()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            if bundle.backend_name == "videollava":
                response = bundle.session.generate(video_path, prompt, args.temperature, max(args.max_new_tokens, 256))
            elif bundle.backend_name == "videochat2":
                response = bundle.session.generate(video_path, prompt, args.temperature, max(args.max_new_tokens, 256))
            elif bundle.backend_name == "minigpt4video":
                response = bundle.session.generate(
                    video_path,
                    prompt,
                    args.temperature,
                    max(args.max_new_tokens, 512),
                    min(args.max_frames, 32),
                )
            else:
                response = bundle.session.generate(video_path, prompt)

        result = build_result(
            clone_args_for_video(args, video_path),
            BACKENDS[bundle.backend_name],
            bundle.model_id,
            bundle.model_path,
            prompt,
            response,
            time.time() - started_clock,
        )
        write_json(output_json, result)
    except Exception:
        returncode = 1
        if output_json.exists() or output_json.is_symlink():
            output_json.unlink()
        stderr_buffer.write(traceback.format_exc())

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    stdout_log.write_text(stdout_text, encoding="utf-8")
    stderr_log.write_text(stderr_text, encoding="utf-8")

    return {
        "video": video_path,
        "video_id": Path(video_path).stem,
        "started_at": started_at,
        "finished_at": now_iso(),
        "returncode": returncode,
        "output_json": str(output_json),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stdout_tail": stdout_text.strip()[-1000:],
        "stderr_tail": stderr_text.strip()[-1000:],
    }


def main() -> int:
    args = build_parser().parse_args()
    prompt = load_prompt(args.prompt, args.prompt_file)
    tasks = validate_task_manifest(json.loads(Path(args.task_manifest).read_text(encoding="utf-8")))
    bundle = create_session_bundle(args)

    records: list[dict[str, Any]] = []
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else None

    for task in tasks:
        record = run_one_task(task, args, prompt, bundle)
        records.append(record)
        if summary_path is not None:
            write_json(
                summary_path,
                {
                    "backend": bundle.backend_name,
                    "model_id": bundle.model_id,
                    "records": records,
                    "failed_count": sum(1 for item in records if item["returncode"] != 0),
                    "updated_at": now_iso(),
                },
            )

    failed_count = sum(1 for item in records if item["returncode"] != 0)
    summary = {
        "backend": bundle.backend_name,
        "model_id": bundle.model_id,
        "records": records,
        "failed_count": failed_count,
        "updated_at": now_iso(),
    }
    if summary_path is not None:
        write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

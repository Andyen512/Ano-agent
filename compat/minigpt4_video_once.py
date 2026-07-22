#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINIGPT4_ROOT = PROJECT_ROOT / "code" / "MiniGPT4-Video"
MODELS_ROOT = PROJECT_ROOT / "models"
VENDOR_COMMON = PROJECT_ROOT / ".vendor" / "common"
VENDOR_MINIGPT4_TRANSFORMERS = PROJECT_ROOT / ".vendor" / "minigpt4_transformers"
HF_CACHE = PROJECT_ROOT / "models" / ".hf_cache"
HF_HOME = PROJECT_ROOT / "models" / ".hf_home"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=32)
    return parser


def find_existing_model_dir(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        for name in (candidate.replace("/", "__"), candidate.replace("/", "_")):
            path = MODELS_ROOT / name
            if path.exists():
                return path.resolve()
    return None


def resolve_mistral_dir() -> Path:
    mistral = find_existing_model_dir(
        [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Mistral-7B-Instruct-v0.2",
        ]
    )
    if mistral is None:
        raise FileNotFoundError("Local Mistral-7B-Instruct-v0.2 directory not found under models/.")
    return mistral


def resolve_eva_vit() -> Path:
    candidates = [
        MODELS_ROOT / "eva_vit_g.pth",
        PROJECT_ROOT / "eva_vit_g.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Local eva_vit_g.pth not found.")


def resolve_ckpt() -> Path:
    candidate = MODELS_ROOT / "Vision-CAIR__MiniGPT4-Video" / "checkpoints" / "video_mistral_checkpoint_last.pth"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError("MiniGPT4-Video checkpoint video_mistral_checkpoint_last.pth not found.")


def prepare_cfg(temp_dir: Path, mistral_dir: Path, ckpt_path: Path) -> Path:
    template_path = MINIGPT4_ROOT / "test_configs" / "mistral_test_config.yaml"
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config["model"]["llama_model"] = str(mistral_dir)
    config["model"]["ckpt"] = str(ckpt_path)
    output_path = temp_dir / "mistral_local_test_config.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def prepare_torch_cache(temp_dir: Path, eva_vit_path: Path) -> Path:
    torch_home = temp_dir / "torch_home"
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / "eva_vit_g.pth"
    if not target.exists():
        target.symlink_to(eva_vit_path)
    return torch_home


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class MiniGPT4VideoSession:
    def __init__(self) -> None:
        mistral_dir = resolve_mistral_dir()
        eva_vit_path = resolve_eva_vit()
        ckpt_path = resolve_ckpt()

        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="minigpt4_video_")
        self.temp_dir = Path(self.temp_dir_obj.name)
        cfg_path = prepare_cfg(self.temp_dir, mistral_dir, ckpt_path)
        torch_home = prepare_torch_cache(self.temp_dir, eva_vit_path)

        os.environ["TORCH_HOME"] = str(torch_home)
        os.environ["HF_HOME"] = str(HF_HOME)
        os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
        os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
        os.environ["HF_TKN"] = os.environ.get("HF_TKN", "")

        sys.path.insert(0, str(VENDOR_MINIGPT4_TRANSFORMERS))
        sys.path.insert(0, str(VENDOR_COMMON))
        sys.path.insert(0, str(MINIGPT4_ROOT))

        from minigpt4.common.eval_utils import init_model
        from minigpt4.conversation.conversation import CONV_VISION
        from PIL import Image
        import cv2

        self.conv_template = CONV_VISION.copy()
        self.conv_template.system = ""
        self.Image = Image
        self.cv2 = cv2
        self.args = argparse.Namespace(
            cfg_path=str(cfg_path),
            ckpt=str(ckpt_path),
            lora_r=64,
            lora_alpha=16,
        )
        self.model, self.vis_processor, *_ = init_model(self.args)

    def _extract_video_info(self, video_path: str, max_images_length: int) -> tuple[int, float]:
        capture = self.cv2.VideoCapture(video_path)
        fps = float(capture.get(self.cv2.CAP_PROP_FPS) or 0.0)
        total_num_frames = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        if fps <= 0:
            fps = 1.0
        sampling_interval = max(1, total_num_frames // max_images_length) if total_num_frames > 0 else 1
        return sampling_interval, fps

    def _prepare_input(self, video_path: str, instruction: str, max_frames: int):
        sampling_interval, _ = self._extract_video_info(video_path, max_frames)
        capture = self.cv2.VideoCapture(video_path)
        images = []
        frame_count = 0
        img_placeholder = ""
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            if frame_count % sampling_interval == 0:
                frame = self.Image.fromarray(frame[:, :, ::-1])
                images.append(self.vis_processor(frame))
                img_placeholder += "<Img><ImageHere>"
            frame_count += 1
            if len(images) >= max_frames:
                break
        capture.release()
        if not images:
            raise RuntimeError(f"Unable to decode frames from video: {video_path}")
        return torch.stack(images), img_placeholder + "\n" + instruction

    def generate(self, video_path: str, prompt: str, temperature: float, max_new_tokens: int, max_frames: int) -> str:
        setup_seeds(50)
        prepared_images, prepared_instruction = self._prepare_input(video_path, prompt, max_frames)
        prepared_images = prepared_images.unsqueeze(0)
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], prepared_instruction)
        conv.append_message(conv.roles[1], None)
        rendered_prompt = [conv.get_prompt()]
        answers = self.model.generate(
            prepared_images,
            rendered_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.0),
            lengths=[prepared_images.shape[1]],
            num_beams=1,
        )
        return str(answers[0]).replace("\\_", "_").replace('\\"', '"').strip()


def main() -> int:
    args = build_parser().parse_args()
    session = MiniGPT4VideoSession()
    print(session.generate(args.video_path, args.prompt, args.temperature, args.max_new_tokens, args.max_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

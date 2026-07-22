#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASK_ANYTHING_ROOT = PROJECT_ROOT / "code" / "Ask-Anything" / "video_chat2"
VENDOR_COMMON = PROJECT_ROOT / ".vendor" / "common"
HF_CACHE = PROJECT_ROOT / "models" / ".hf_cache"
HF_HOME = PROJECT_ROOT / "models" / ".hf_home"
PROJECT_TMP = PROJECT_ROOT / ".tmp"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--hd-num", type=int, default=6)
    return parser


def prepare_env() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
    PROJECT_TMP.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(PROJECT_TMP))
    os.environ.setdefault("TMP", str(PROJECT_TMP))
    os.environ.setdefault("TEMP", str(PROJECT_TMP))
    sys.path.insert(0, str(VENDOR_COMMON))
    sys.path.insert(0, str(ASK_ANYTHING_ROOT))


def resolve_mistral_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "models" / "Mistral-7B-Instruct-v0.2",
        PROJECT_ROOT / "models" / "mistralai__Mistral-7B-Instruct-v0.2",
        PROJECT_ROOT / "models" / "mistralai__Mistral-7B-Instruct-v0.2_tokenizer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No local Mistral-7B-Instruct-v0.2 directory found for VideoChat2.")


def build_compat_model_dir(model_path: Path) -> Path:
    compat_root = Path(os.environ.get("TMPDIR") or PROJECT_TMP)
    compat_dir = compat_root / "videochat2_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    mistral_dir = resolve_mistral_dir()
    for source in model_path.iterdir():
        target = compat_dir / source.name
        if source.name == "config.json":
            config = json.loads(source.read_text(encoding="utf-8"))
            config["mistral_model_path"] = str(mistral_dir)
            config["use_flash_attention"] = False
            config["videochat2_model_path"] = ""
            target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            continue
        if target.exists() or target.is_symlink():
            continue
        os.symlink(source, target)
    return compat_dir


def load_vit_module(compat_dir: Path):
    spec = importlib.util.spec_from_file_location("videochat2_vit", compat_dir / "vit.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def get_index(num_frames: int, num_segments: int):
    import numpy as np

    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    return np.array([start + int(round(seg_size * idx)) for idx in range(num_segments)])


def load_video(video_path: str, num_segments: int, resolution: int, hd_num: int):
    from decord import VideoReader, cpu
    import decord
    from torchvision import transforms
    from dataset.hd_utils import HD_transform_no_padding

    decord.bridge.set_bridge("torch")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_indices = get_index(len(vr), num_segments)
    frames = vr.get_batch(frame_indices).permute(0, 3, 1, 2)
    frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform(frames)


def get_prompt(conv) -> str:
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv) -> str:
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            if message:
                ret += role + " " + message
            else:
                ret += role
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def ask(text, conv) -> None:
    conv.messages.append([conv.roles[0], text])


class VideoChat2Session:
    def __init__(self, model_path: str, num_segments: int = 8, resolution: int = 224, hd_num: int = 6) -> None:
        prepare_env()

        from transformers import AutoModel

        self.num_segments = num_segments
        self.resolution = resolution
        self.hd_num = hd_num

        resolved_model_path = Path(model_path).expanduser().resolve()
        compat_dir = build_compat_model_dir(resolved_model_path)
        vit_module = load_vit_module(compat_dir)

        self.model = AutoModel.from_pretrained(str(compat_dir), trust_remote_code=True, low_cpu_mem_usage=False).to("cuda:0")
        self.model.vision_encoder.encoder.pos_embed = vit_module.get_sinusoid_encoding_table(
            n_position=(resolution // 16) ** 2 * num_segments,
            d_hid=self.model.vision_encoder.encoder.pos_embed.shape[-1],
            cur_frame=num_segments,
        )

    def get_context_emb(self, conv, img_list):
        import torch

        prompt = get_prompt2(conv)
        prompt_segs = prompt.split("<VideoHere>") if "<VideoHere>" in prompt else prompt.split("<ImageHere>")
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of placeholders and embeddings."
        with torch.no_grad():
            seg_tokens = [
                self.model.mistral_tokenizer(seg, return_tensors="pt", add_special_tokens=index == 0).to("cuda:0").input_ids
                for index, seg in enumerate(prompt_segs)
            ]
            seg_embs = [self.model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        return torch.cat(mixed_embs, dim=1)

    def answer(self, conv, img_list, max_new_tokens: int, do_sample: bool, temperature: float):
        import torch

        conv.messages.append([conv.roles[1], None])
        embs = self.get_context_emb(conv, img_list)
        with torch.no_grad():
            outputs = self.model.mistral_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=0.9,
                temperature=max(temperature, 0.2) if do_sample else 1.0,
            )
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
        return output_text.split("</s>")[0].strip()

    def generate(self, video_path: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
        video = load_video(
            video_path,
            num_segments=self.num_segments,
            resolution=self.resolution,
            hd_num=self.hd_num,
        )
        t, c, h, w = video.shape
        video = video.reshape(1, t, c, h, w).to("cuda:0")

        import torch

        with torch.no_grad():
            video_emb, _, _ = self.model.encode_img(video, [""])
        video_list = [video_emb[0]]

        chat = type("ChatState", (), {"system": "", "roles": ("[INST]", "[/INST]"), "messages": [], "sep": ""})()
        chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
        ask(prompt, chat)
        return self.answer(chat, video_list, max_new_tokens, temperature > 0, temperature)


def main() -> int:
    args = build_parser().parse_args()
    session = VideoChat2Session(
        args.model_path,
        num_segments=args.num_segments,
        resolution=args.resolution,
        hd_num=args.hd_num,
    )
    print(session.generate(args.video_path, args.prompt, args.temperature, args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

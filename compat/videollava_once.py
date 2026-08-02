#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_attention_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_LLAVA_ROOT = PROJECT_ROOT / "code" / "Video-LLaVA-7B"


def install_safe_transformers_registration() -> None:
    original_config_register = AutoConfig.register
    original_model_register = AutoModelForCausalLM.register

    def safe_config_register(model_type, config, exist_ok=False):
        return original_config_register(model_type, config, exist_ok=True)

    def safe_model_register(config_class, model_class, exist_ok=False):
        return original_model_register(config_class, model_class, exist_ok=True)

    AutoConfig.register = safe_config_register
    AutoModelForCausalLM.register = safe_model_register


def install_transformers_compat() -> None:
    import transformers.models.bloom.modeling_bloom as modeling_bloom
    import transformers.models.opt.modeling_opt as modeling_opt

    if not hasattr(modeling_bloom, "_expand_mask"):
        def bloom_expand_mask(mask, tgt_length=None):
            return _prepare_4d_attention_mask(mask, torch.float32, tgt_len=tgt_length)
        modeling_bloom._expand_mask = bloom_expand_mask
    if not hasattr(modeling_bloom, "_make_causal_mask"):
        modeling_bloom._make_causal_mask = AttentionMaskConverter._make_causal_mask

    if not hasattr(modeling_opt, "_expand_mask"):
        def opt_expand_mask(mask, dtype, tgt_len=None):
            return _prepare_4d_attention_mask(mask, dtype, tgt_len=tgt_len)
        modeling_opt._expand_mask = opt_expand_mask
    if not hasattr(modeling_opt, "_make_causal_mask"):
        modeling_opt._make_causal_mask = AttentionMaskConverter._make_causal_mask


def install_torchvision_compat() -> None:
    # pytorchvideo 0.1.5 imports this module name, which was removed from
    # torchvision 0.18 while the underlying functional API stayed compatible.
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--model-base")
    return parser


class VideoLlavaSession:
    def __init__(self, model_path: str, model_base: str | None = None) -> None:
        install_safe_transformers_registration()
        install_transformers_compat()
        install_torchvision_compat()
        sys.path.insert(0, str(VIDEO_LLAVA_ROOT))

        from videollava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from videollava.conversation import SeparatorStyle, conv_templates
        from videollava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
        from videollava.model.builder import load_pretrained_model
        from videollava.utils import disable_torch_init

        disable_torch_init()

        self.default_image_token = DEFAULT_IMAGE_TOKEN
        self.default_im_end_token = DEFAULT_IM_END_TOKEN
        self.default_im_start_token = DEFAULT_IM_START_TOKEN
        self.image_token_index = IMAGE_TOKEN_INDEX
        self.separator_style = SeparatorStyle
        self.conv_templates = conv_templates
        self.keywords_stopping_criteria = KeywordsStoppingCriteria
        self.tokenizer_image_token = tokenizer_image_token

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path,
            model_base,
            self.model_name,
            device_map={"": "cuda:0"},
        )
        original_forward = self.model.forward

        def compat_forward(this, *f_args, **f_kwargs):
            f_kwargs.pop("cache_position", None)
            return original_forward(*f_args, **f_kwargs)

        self.model.forward = types.MethodType(compat_forward, self.model)
        self.model._validate_model_kwargs = lambda model_kwargs: None
        self.video_processor = self.processor["video"]

        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

    def generate(self, video_path: str, prompt: str, temperature: float, max_new_tokens: int) -> str:
        conv = self.conv_templates[self.conv_mode].copy()
        frame_tokens = [self.default_image_token] * self.model.get_video_tower().config.num_frames
        if getattr(self.model.config, "mm_use_im_start_end", False):
            prefix = "".join(self.default_im_start_token + token + self.default_im_end_token for token in frame_tokens)
        else:
            prefix = "".join(frame_tokens)
        conv.append_message(conv.roles[0], prefix + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        rendered_prompt = conv.get_prompt()

        video_tensor = self.video_processor(video_path, return_tensors="pt")["pixel_values"][0].to(
            self.model.device,
            dtype=torch.float16,
        )
        input_ids = self.tokenizer_image_token(
            rendered_prompt,
            self.tokenizer,
            self.image_token_index,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        stop_str = conv.sep if conv.sep_style != self.separator_style.TWO else conv.sep2
        stopping_criteria = self.keywords_stopping_criteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[video_tensor],
                do_sample=temperature > 0,
                temperature=max(temperature, 0.2) if temperature > 0 else 0.0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        text = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        return text.replace("\\_", "_")


def main() -> int:
    args = build_parser().parse_args()
    session = VideoLlavaSession(args.model_path, args.model_base)
    print(session.generate(args.video_path, args.prompt, args.temperature, args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

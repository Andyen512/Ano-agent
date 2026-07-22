from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkModel:
    name: str
    model_id: str
    backend: str | None
    notes: str = ""


BENCHMARK_MODELS: tuple[BenchmarkModel, ...] = (
    BenchmarkModel(
        name="Video-LLaVA-7B",
        model_id="LanguageBind/Video-LLaVA-7B",
        backend="videollava",
        notes="Repo code needs local compatibility fixes against the installed transformers version; the isolated adapter is wired in the unified runner.",
    ),
    BenchmarkModel(
        name="VideoChat2-7B",
        model_id="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
        backend="videochat2",
        notes="Requires local bert-base-uncased cache plus a lightweight Mistral tokenizer/config directory for the remote-code wrapper.",
    ),
    BenchmarkModel(
        name="Video-ChatGPT-7B",
        model_id="MBZUAI/Video-ChatGPT-7B",
        backend="videochatgpt",
        notes="The downloaded snapshot is only the projection/adaptor package; it still needs a full local LLaVA-Lightning-7B-v1-1 base checkpoint.",
    ),
    BenchmarkModel(
        name="MiniGPT4-Video",
        model_id="Vision-CAIR/MiniGPT4-Video",
        backend="minigpt4video",
        notes="Needs external base LLM weights (preferably local Mistral-7B-Instruct-v0.2), bert-base-uncased cache, and eva_vit_g.pth in addition to the downloaded .pth checkpoints.",
    ),
    BenchmarkModel(
        name="VideoLLaMA2-7B",
        model_id="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F",
        backend="videollama2",
    ),
    BenchmarkModel(
        name="InternVL3.5-8B",
        model_id="OpenGVLab/InternVL3_5-8B",
        backend="internvl35",
    ),
    BenchmarkModel(
        name="Qwen2.5VL-7B",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        backend="qwen25vl",
        notes="Requires qwen-vl-utils; the unified runner now includes a direct Qwen2.5-VL path.",
    ),
    BenchmarkModel(
        name="Qwen3.5-9B",
        model_id="Qwen/Qwen3.5-9B",
        backend="qwen35vl",
        notes="Requires a recent Transformers build with Qwen3.5 support, preferably under lifebench/.vendor/qwen35_shim (tested with transformers 5.4.0).",
    ),
    BenchmarkModel(
        name="VideoLLaMA3-7B",
        model_id="DAMO-NLP-SG/VideoLLaMA3-7B",
        backend="videollama3",
    ),
    BenchmarkModel(
        name="mPLUG-Owl3-7B",
        model_id="mPLUG/mPLUG-Owl3-7B-241101",
        backend="mplugowl3",
    ),
    BenchmarkModel(
        name="Tarsier2-7B",
        model_id="omni-research/Tarsier2-7b-0115",
        backend="tarsier2",
    ),
)


BY_NAME = {model.name: model for model in BENCHMARK_MODELS}
BY_MODEL_ID = {model.model_id: model for model in BENCHMARK_MODELS}

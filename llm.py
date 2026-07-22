from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


# DEFAULT_LLM_CONFIG = LLMConfig(
#     api_key="sk-65cd6d8e37366dc003116101e9eefdf8c049bc8712d75b6b6d1fd81c752cc4cc",
#     base_url="https://gmn.chuangzuoli.com/v1",
#     model="gpt-5.4",
# )

DEFAULT_LLM_CONFIG = LLMConfig(
    api_key="sk-41f87f38dbe530b92bed40c20a7b26f14f6980b3da056f2d33c12b571561b1f3",
    base_url="https://gmncode.cn/v1",
    model="gpt-5.4",
)


def get_default_llm_config() -> LLMConfig:
    return LLMConfig(
        api_key=os.environ.get("LIFEBENCH_LLM_API_KEY", DEFAULT_LLM_CONFIG.api_key),
        base_url=os.environ.get("LIFEBENCH_LLM_BASE_URL", DEFAULT_LLM_CONFIG.base_url),
        model=os.environ.get("LIFEBENCH_LLM_MODEL", DEFAULT_LLM_CONFIG.model),
    )


def create_client(config: LLMConfig | None = None) -> OpenAI:
    config = config or get_default_llm_config()
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_to_data_url(image: Any, *, format: str = "JPEG", quality: int = 85) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    mime_type = f"image/{format.lower()}"
    if format.upper() == "JPEG":
        mime_type = "image/jpeg"
    return image_bytes_to_data_url(buffer.getvalue(), mime_type=mime_type)


def file_to_data_url(path: str | os.PathLike[str], mime_type: str) -> str:
    payload = Path(path).read_bytes()
    return image_bytes_to_data_url(payload, mime_type=mime_type)


def response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    config: LLMConfig | None = None,
    **kwargs: Any,
) -> str:
    config = config or get_default_llm_config()
    client = create_client(config)
    response = client.chat.completions.create(
        model=model or config.model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response_text(response.choices[0].message.content)


if __name__ == "__main__":
    content = chat_completion(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好，帮我介绍一下 transformer。"},
        ]
    )
    print(content)

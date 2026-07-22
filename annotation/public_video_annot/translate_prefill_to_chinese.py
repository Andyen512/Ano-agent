#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from app import APP_DIR, PROJECT_ROOT, init_db, MODEL_OUTPUTS_ROOTS


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen__Qwen3-4B-Instruct"
AGENT_TEXT_FIELDS = (
    "description",
    "solution_for_person",
    "solution_for_hazard_source",
    "solution_to_prevent_recurrence",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate model prefill free-text fields to Chinese for annotators.")
    parser.add_argument("--db", type=Path, default=APP_DIR / "data" / "app.db")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", help="Only translate one dataset, for example web_h264.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-english", action="store_true", help="Only re-translate entries where description is English (not Chinese)")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser


def normalize_none(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "nan", "n/a", "na"}:
        return "无"
    if text.lower() == "unknown":
        return "未知"
    return text


def load_agent_outputs(video_key: str) -> list[dict]:
    """Load agent outputs from review_summary.json for a video."""
    # 处理 video_key 映射: web_h264_segments/xxx -> web_h264/xxx
    actual_video_key = video_key
    if video_key.startswith("web_h264_segments/"):
        actual_video_key = "web_h264/" + video_key[len("web_h264_segments/"):]
    
    for reviews_dir in MODEL_OUTPUTS_ROOTS:
        if not reviews_dir.exists():
            continue
        for dataset_dir in reviews_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            if not actual_video_key.startswith(dataset_dir.name):
                continue
            rest_path = actual_video_key[len(dataset_dir.name)+1:]
            parts = rest_path.split('/')
            last = parts[-1]
            for ext in ('.mp4', '.avi', '.mkv', '.mov', '.webm'):
                if last.lower().endswith(ext):
                    last = last[:-len(ext)]
                    break
            parts[-1] = last
            review_dir = dataset_dir.joinpath(*parts)
            review_file = review_dir / "review_summary.repaired.json"
            if not review_file.exists():
                review_file = review_dir / "review_summary.json"
            if not review_file.exists():
                continue
            try:
                with open(review_file) as f:
                    data = json.load(f)
                agents = []
                for agent in data.get("agent_results", []):
                    pd = agent.get("parsed_decision", {})
                    desc = pd.get("risk_description") or pd.get("normal_video_description", "")
                    sol_person = pd.get("solution_for_person", "")
                    sol_hazard = pd.get("solution_for_hazard_source", "")
                    sol_prev = pd.get("solution_to_prevent_recurrence", "")
                    model_backend = agent.get("model", {}).get("backend", "unknown")
                    agents.append({
                        "model_backend": model_backend,
                        "description": desc,
                        "solution_for_person": sol_person,
                        "solution_for_hazard_source": sol_hazard,
                        "solution_to_prevent_recurrence": sol_prev,
                    })
                return agents
            except Exception as e:
                print(f"Error loading agent outputs for {video_key}: {e}")
                continue
    return []


class QwenTranslator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.max_new_tokens = args.max_new_tokens

    def render_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你只输出合法 JSON，不要输出解释、Markdown 或思考过程。"},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def translate_text(self, text: str) -> str:
        """Translate a single text field to Chinese."""
        if not text or text in ("无", "未知", "None"):
            return text or "无"
        
        prompt = f"Translate to Chinese: {text}"
        rendered = self.render_prompt(prompt)
        inputs = self.tokenizer([rendered], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        
        # 清理响应，移除可能的前缀
        if response.startswith("system") or response.startswith("user"):
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith("system") and not line.startswith("user") and not line.startswith("Translate"):
                    response = line
                    break
        
        return normalize_none(response)

    def translate_agent(self, agent: dict) -> dict[str, str]:
        """Translate all fields for a single agent."""
        result = {}
        for field in AGENT_TEXT_FIELDS:
            value = agent.get(field, "")
            if not value or value in ("无", "未知"):
                result[field] = "无"
            else:
                result[field] = self.translate_text(value)
        return result


def videos_to_translate(conn: sqlite3.Connection, args: argparse.Namespace) -> list[str]:
    """Get list of video_keys that need translation."""
    query = """
        SELECT DISTINCT v.video_key
        FROM videos v
        WHERE v.dataset = ?
    """
    params: list[Any] = [args.dataset]
    
    if args.retry_english:
        # Only re-translate where description has no Chinese characters
        query += """
            AND v.video_key IN (
                SELECT video_key FROM model_prediction_zh_agents 
                WHERE description_zh NOT GLOB '*[一-龥]*' 
                AND description_zh IS NOT NULL AND description_zh != '' AND description_zh != '无'
            )
        """
    elif not args.overwrite:
        query += " AND v.video_key NOT IN (SELECT DISTINCT video_key FROM model_prediction_zh_agents)"
    
    query += " ORDER BY v.video_key"
    if args.limit is not None:
        query += " LIMIT ?"
        params.append(args.limit)
    
    return [row["video_key"] for row in conn.execute(query, params).fetchall()]


def upsert_agent_translation(conn: sqlite3.Connection, video_key: str, agent_idx: int, model_backend: str, payload: dict[str, str]) -> None:
    conn.execute(
        """
        INSERT INTO model_prediction_zh_agents (
            video_key, agent_idx, model_backend, description_zh, solution_for_person_zh,
            solution_for_hazard_source_zh, solution_to_prevent_recurrence_zh, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(video_key, agent_idx) DO UPDATE SET
            model_backend=excluded.model_backend,
            description_zh=excluded.description_zh,
            solution_for_person_zh=excluded.solution_for_person_zh,
            solution_for_hazard_source_zh=excluded.solution_for_hazard_source_zh,
            solution_to_prevent_recurrence_zh=excluded.solution_to_prevent_recurrence_zh,
            updated_at=datetime('now')
        """,
        (
            video_key,
            agent_idx,
            model_backend,
            payload["description"],
            payload["solution_for_person"],
            payload["solution_for_hazard_source"],
            payload["solution_to_prevent_recurrence"],
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    init_db()
    conn = sqlite3.connect(args.db, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    
    if not args.dataset:
        print("Error: --dataset is required")
        return 1
    
    videos = videos_to_translate(conn, args)
    print(f"videos_to_translate={len(videos)}", flush=True)
    
    if not videos:
        print("No videos to translate")
        conn.close()
        return 0
    
    translator = QwenTranslator(args)
    total_agents = 0
    
    for video_idx, video_key in enumerate(videos, start=1):
        agents = load_agent_outputs(video_key)
        if not agents:
            print(f"[{video_idx}/{len(videos)}] {video_key} - no agent outputs found", flush=True)
            continue
        
        for agent_idx, agent in enumerate(agents):
            payload = translator.translate_agent(agent)
            upsert_agent_translation(conn, video_key, agent_idx, agent["model_backend"], payload)
            total_agents += 1
        
        conn.commit()
        print(f"[{video_idx}/{len(videos)}] {video_key} - {len(agents)} agents translated", flush=True)
    
    conn.close()
    print(f"\nDone! Translated {total_agents} agent outputs for {len(videos)} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

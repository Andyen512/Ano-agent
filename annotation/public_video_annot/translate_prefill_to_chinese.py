#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from app import APP_DIR, PROJECT_ROOT, init_db


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen__Qwen3-8B-Base"
TEXT_FIELDS = (
    "description",
    "risk_localization",
    "solution_for_person",
    "solution_for_hazard_source",
    "solution_prevent_recurrence",
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
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser


def normalize_none(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "nan", "n/a", "na"}:
        return "无"
    if text.lower() == "unknown":
        return "未知"
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


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

    def translate_row(self, row: sqlite3.Row) -> dict[str, str]:
        source = {field: normalize_none(row[field]) for field in TEXT_FIELDS}
        prompt = (
            "请把下面视频安全标注中的自由文本字段翻译成简体中文。\n"
            "要求：\n"
            "1. 只翻译字段值，不新增事实。\n"
            "2. Risk localization 如果是 None 输出“无”；如果是 Unknown 输出“未知”；如果是 [1,3] 这类时间段，保持原格式。\n"
            "3. 如果原文是 None/null/空，输出“无”。\n"
            "4. 输出 JSON，key 必须保持为英文原 key。\n\n"
            f"{json.dumps(source, ensure_ascii=False)}"
        )
        rendered = self.render_prompt(prompt)
        inputs = self.tokenizer([rendered], return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        payload = extract_json_object(response)
        return {field: normalize_none(payload.get(field)) for field in TEXT_FIELDS}


def rows_to_translate(conn: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    query = """
        SELECT p.video_key, p.description, p.risk_localization, p.solution_for_person,
               p.solution_for_hazard_source, p.solution_prevent_recurrence, v.dataset
        FROM model_predictions p
        JOIN videos v ON v.video_key = p.video_key
    """
    clauses = []
    params: list[Any] = []
    if args.dataset:
        clauses.append("v.dataset = ?")
        params.append(args.dataset)
    if not args.overwrite:
        clauses.append("p.video_key NOT IN (SELECT video_key FROM model_prediction_zh)")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY COALESCE(p.yes_count, 0) DESC, v.dataset, p.video_key"
    if args.limit is not None:
        query += " LIMIT ?"
        params.append(args.limit)
    return conn.execute(query, params).fetchall()


def upsert_translation(conn: sqlite3.Connection, video_key: str, payload: dict[str, str]) -> None:
    conn.execute(
        """
        INSERT INTO model_prediction_zh (
            video_key, description, risk_localization, solution_for_person,
            solution_for_hazard_source, solution_prevent_recurrence, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(video_key) DO UPDATE SET
            description=excluded.description,
            risk_localization=excluded.risk_localization,
            solution_for_person=excluded.solution_for_person,
            solution_for_hazard_source=excluded.solution_for_hazard_source,
            solution_prevent_recurrence=excluded.solution_prevent_recurrence,
            updated_at=datetime('now')
        """,
        (
            video_key,
            payload["description"],
            payload["risk_localization"],
            payload["solution_for_person"],
            payload["solution_for_hazard_source"],
            payload["solution_prevent_recurrence"],
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    init_db()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = rows_to_translate(conn, args)
    print(f"rows_to_translate={len(rows)}", flush=True)
    translator = QwenTranslator(args)
    for index, row in enumerate(rows, start=1):
        payload = translator.translate_row(row)
        upsert_translation(conn, row["video_key"], payload)
        conn.commit()
        print(f"[{index}/{len(rows)}] {row['video_key']}", flush=True)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

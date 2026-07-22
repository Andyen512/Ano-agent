#!/usr/bin/env python3
"""Translate original_prompt in agents_json to Chinese for generated videos."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from app import APP_DIR, init_db


DEFAULT_MODEL_PATH = APP_DIR.parents[1] / "models" / "Qwen__Qwen3-4B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate original_prompt to Chinese.")
    parser.add_argument("--db", type=Path, default=APP_DIR / "data" / "app.db")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", type=int, help="Use specific GPU")
    parser.add_argument("--output", type=Path, default=Path("/home/caiqingyuan/code/lifebench/data/generated_videos/translations.json"), help="Save translations to JSON file instead of DB")
    return parser


class QwenTranslator:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.torch_dtype]
        device_map = args.device_map
        if args.gpu is not None:
            device_map = f"cuda:{args.gpu}"
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True, local_files_only=True)
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def translate(self, text: str) -> str:
        if not text:
            return ""
        messages = [
            {"role": "system", "content": "你只输出翻译结果，不要输出任何解释、Markdown 或思考过程。"},
            {"role": "user", "content": f"将以下英文翻译为中文：\n{text}"},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([rendered], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        prompts = []
        for text in texts:
            if not text:
                prompts.append("")
                continue
            messages = [
                {"role": "system", "content": "你只输出翻译结果，不要输出任何解释、Markdown 或思考过程。"},
                {"role": "user", "content": f"将以下英文翻译为中文：\n{text}"},
            ]
            try:
                rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(rendered)

        non_empty = [(i, p) for i, p in enumerate(prompts) if p]
        if not non_empty:
            return [""] * len(texts)

        batch_inputs = self.tokenizer([p for _, p in non_empty], return_tensors="pt", padding=True).to(self.model.device)
        batch_outputs = self.model.generate(**batch_inputs, max_new_tokens=1024, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)

        results = [""] * len(texts)
        for j, (idx, _) in enumerate(non_empty):
            new_ids = batch_outputs[j][batch_inputs["input_ids"].shape[-1]:]
            results[idx] = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return results


def main() -> int:
    args = build_parser().parse_args()
    init_db()
    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")

    query = """
        SELECT video_key, agents_json FROM model_predictions
        WHERE video_key LIKE 'generated_videos/%'
        AND agents_json IS NOT NULL AND agents_json != ''
    """
    if not args.overwrite:
        query += " AND (agents_json NOT LIKE '%original_prompt_zh%' OR agents_json LIKE '%original_prompt_zh\": \"\"%' OR agents_json LIKE '%original_prompt_zh\":\"\"%')"
    if args.limit:
        query += f" LIMIT {args.limit}"

    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} videos to translate")

    if not rows:
        conn.close()
        return 0

    translator = QwenTranslator(args)
    batch_size = args.batch_size
    done = 0
    results = []

    # Collect all items to translate
    items = []
    for row in rows:
        video_key = row["video_key"]
        try:
            agents_data = json.loads(row["agents_json"])
        except:
            continue
        original_prompt = agents_data.get("original_prompt", "")
        if not original_prompt:
            continue
        existing_zh = agents_data.get("original_prompt_zh", "")
        if existing_zh and not args.overwrite:
            continue
        items.append((video_key, agents_data, original_prompt))

    print(f"Items to translate: {len(items)}")

    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        prompts = [item[2] for item in batch]
        translations = translator.translate_batch(prompts)

        for (video_key, agents_data, _), zh in zip(batch, translations):
            agents_data["original_prompt_zh"] = zh
            results.append({"video_key": video_key, "agents_json": json.dumps(agents_data, ensure_ascii=False)})
            done += 1

        print(f"[{done}/{len(items)}] batch {i//batch_size+1}", flush=True)

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} translations to {args.output}")
    else:
        # Save to DB
        for r in results:
            conn.execute(
                "UPDATE model_predictions SET agents_json = ? WHERE video_key = ?",
                (r["agents_json"], r["video_key"]),
            )
        conn.commit()
        print(f"\nDone! Translated {done} original prompts")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

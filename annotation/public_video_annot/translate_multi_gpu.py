#!/usr/bin/env python3
"""Translate original_prompt using multiple GPUs in parallel."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
from pathlib import Path
from typing import Any

from app import APP_DIR, init_db


DEFAULT_MODEL_PATH = APP_DIR.parents[1] / "models" / "Qwen__Qwen3-4B-Instruct"
OUTPUT_FILE = Path("/home/caiqingyuan/code/lifebench/data/generated_videos/translations.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate original_prompt with multi-GPU.")
    parser.add_argument("--db", type=Path, default=APP_DIR / "data" / "app.db")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated GPU IDs")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return parser


class QwenTranslator:
    def __init__(self, model_path, torch_dtype, gpu):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=f"cuda:{gpu}", trust_remote_code=True, local_files_only=True)
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

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


def worker(gpu_id, items, model_path, torch_dtype, batch_size, output_file):
    """Worker process for a single GPU."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting with {len(items)} items", flush=True)
    translator = QwenTranslator(model_path, torch_dtype, 0)  # gpu=0 because CUDA_VISIBLE_DEVICES is set
    
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        prompts = [item[2] for item in batch]
        translations = translator.translate_batch(prompts)
        
        for (video_key, agents_data, _), zh in zip(batch, translations):
            agents_data["original_prompt_zh"] = zh
            results.append({"video_key": video_key, "agents_json": json.dumps(agents_data, ensure_ascii=False)})
        
        print(f"[GPU {gpu_id}] [{i+len(batch)}/{len(items)}]", flush=True)
    
    # Save to temp file
    temp_file = output_file.parent / f"translations_gpu{gpu_id}.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    
    print(f"[GPU {gpu_id}] Done! Saved {len(results)} to {temp_file}", flush=True)
    return len(results)


def main() -> int:
    args = build_parser().parse_args()
    init_db()
    conn = sqlite3.connect(args.db, timeout=60)
    conn.row_factory = sqlite3.Row

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
    conn.close()
    print(f"Found {len(rows)} videos to translate")

    if not rows:
        return 0

    # Collect items
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

    # Split items across GPUs
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    chunk_size = len(items) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = start + chunk_size if i < num_gpus - 1 else len(items)
        chunks.append(items[start:end])

    # Start workers
    processes = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        p = mp.Process(target=worker, args=(gpu_id, chunk, args.model_path, args.torch_dtype, args.batch_size, args.output))
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    # Merge results
    all_results = []
    for gpu_id in gpu_ids:
        temp_file = args.output.parent / f"translations_gpu{gpu_id}.json"
        if temp_file.exists():
            with open(temp_file, "r", encoding="utf-8") as f:
                all_results.extend(json.load(f))
            temp_file.unlink()

    # Save merged results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(all_results)} translations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

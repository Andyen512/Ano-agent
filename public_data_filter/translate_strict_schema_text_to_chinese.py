#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ['DS_BUILD_OPS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_JSONL = SCRIPT_DIR / "batch_outputs" / "persistent_full" / "strict_english_schema.jsonl"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_outputs" / "persistent_full"
DEFAULT_OUTPUT_STEM = "strict_schema_chinese_text"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen__Qwen3-8B"

TRANSLATE_FIELDS = (
    "Normal-video description or risk description",
    "Solutions-For person",
    "Solutions-For hazard source",
    "Solutions-Prevent recurrence",
)
RISK_LOCALIZATION_FIELD = "Risk localization"


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Translate the free-text fields in strict_english_schema.jsonl into Chinese with Qwen. "
            "Risk and Level 1/2/3 taxonomy fields are kept unchanged."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument(
        "--gpus",
        help=(
            "Comma-separated GPU ids for one-command parallel launch, for example 0,1,2,3,4,5,6,7. "
            "When set without --merge-shards, this script starts one shard worker per GPU and merges outputs."
        ),
    )
    parser.add_argument("--progress-poll-seconds", type=int, default=30)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser


def output_stem(base_stem: str, shard_index: int, num_shards: int) -> str:
    if num_shards <= 1:
        return base_stem
    return f"{base_stem}.shard{shard_index:02d}of{num_shards:02d}"


def output_paths(output_root: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        output_root / f"{stem}.jsonl",
        output_root / f"{stem}.json",
        output_root / f"{stem}.csv",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(row.get("video_relpath")): row for row in rows}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fieldnames = ["video_relpath"]
    else:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key, "") for key in fieldnames})


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "nan", "n/a", "na"}:
        return "无"
    if text.lower() == "unknown":
        return "未知"
    return text


def is_none_like(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "none", "null", "nan", "n/a", "na"}


def has_chinese(value: Any) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value or ""))


def clean_translation(value: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:text|txt|zh|中文)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(译文|翻译|中文翻译|结果)\s*[:：]\s*", "", text).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            payload, _end = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"No JSON object found in model output: {text[:200]}")


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
        self.cache: dict[str, str] = {}

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

    def translate_text(self, value: str, max_retries: int = 3) -> str:
        value = str(value or "").strip()
        cache_key = re.sub(r"\s+", " ", value)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if is_none_like(value):
            self.cache[cache_key] = "无"
            return "无"
        if has_chinese(value):
            self.cache[cache_key] = value
            return value

        messages = [
            {"role": "system", "content": "你是专业翻译。只输出简体中文译文，不要解释，不要输出原文，不要使用 JSON。"},
            {"role": "user", "content": f"请将下面英文翻译成简体中文，只输出译文：\n{value}"},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([rendered], return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        translated = None
        for attempt in range(max_retries):
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            response = clean_translation(response)
            if response and has_chinese(response):
                translated = response
                break
            if attempt < max_retries - 1:
                print(f"[RETRY {attempt + 1}/{max_retries}] text", flush=True)

        result = translated if translated else value
        self.cache[cache_key] = result
        return result

    def translate(self, source: dict[str, str]) -> dict[str, str]:
        results = {}
        for field in TRANSLATE_FIELDS:
            results[field] = self.translate_text(source.get(field, ""))
        return results


def translated_row(row: dict[str, Any], translator: QwenTranslator, model_path: Path) -> dict[str, Any]:
    source = {field: normalize_text(row.get(field)) for field in TRANSLATE_FIELDS}
    zh = translator.translate(source)
    out = dict(row)
    out["Normal-video description or risk description zh"] = zh["Normal-video description or risk description"]
    out["Risk localization zh"] = str(row.get(RISK_LOCALIZATION_FIELD) or "").strip()
    out["Solutions-For person zh"] = zh["Solutions-For person"]
    out["Solutions-For hazard source zh"] = zh["Solutions-For hazard source"]
    out["Solutions-Prevent recurrence zh"] = zh["Solutions-Prevent recurrence"]
    out["translation_model"] = str(model_path)
    out["translated_at"] = utc_tag()
    return out


def seed_translation_cache(rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping = {
        "Normal-video description or risk description": "Normal-video description or risk description zh",
        "Solutions-For person": "Solutions-For person zh",
        "Solutions-For hazard source": "Solutions-For hazard source zh",
        "Solutions-Prevent recurrence": "Solutions-Prevent recurrence zh",
    }
    cache: dict[str, str] = {}
    for row in rows:
        for src_field, zh_field in mapping.items():
            source = str(row.get(src_field) or "").strip()
            translated = str(row.get(zh_field) or "").strip()
            if source and translated and has_chinese(translated):
                cache[re.sub(r"\s+", " ", source)] = translated
    return cache


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def has_chinese(text: str) -> bool:
        return any('\u4e00' <= c <= '\u9fff' for c in str(text or ""))
    return {
        "processed_videos": len(rows),
        "risk_counts": dict(Counter(row.get("Risk") for row in rows)),
        "rows_with_chinese_description": sum(has_chinese(row.get("Normal-video description or risk description zh")) for row in rows),
    }


def normalize_output_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["Risk localization zh"] = str(out.get(RISK_LOCALIZATION_FIELD) or "").strip()
    return out


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]], stem: str, merged_from: list[str] | None = None) -> None:
    rows = [normalize_output_row(row) for row in rows]
    jsonl_path, json_path, csv_path = output_paths(args.output_root, stem)
    payload = {
        "manifest": {
            "input_jsonl": str(args.input_jsonl),
            "output_root": str(args.output_root),
            "output_stem": stem,
            "model_path": str(args.model_path),
            "generated_at": utc_tag(),
            "merged_from": merged_from or [],
            "translated_fields": list(TRANSLATE_FIELDS),
            "copied_fields": [RISK_LOCALIZATION_FIELD],
            "note": "Risk, Risk localization, and Level 1/2/3 taxonomy fields are preserved; zh text fields contain Chinese translations for annotators.",
        },
        "summary": summarize(rows),
        "results": rows,
    }
    write_jsonl(jsonl_path, rows)
    write_json(json_path, payload)
    write_csv(csv_path, rows)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(str(jsonl_path), flush=True)
    print(str(json_path), flush=True)
    print(str(csv_path), flush=True)


def merge_shards(args: argparse.Namespace) -> int:
    rows_by_video: dict[str, dict[str, Any]] = {}
    merged_from: list[str] = []
    missing: list[str] = []
    for shard_index in range(args.num_shards):
        stem = output_stem(args.output_stem, shard_index, args.num_shards)
        jsonl_path, _json_path, _csv_path = output_paths(args.output_root, stem)
        if not jsonl_path.exists():
            missing.append(str(jsonl_path))
            continue
        merged_from.append(str(jsonl_path))
        rows_by_video.update(existing_rows(jsonl_path))
    if missing:
        raise FileNotFoundError("Missing shard outputs: " + ", ".join(missing[:5]))
    rows = [rows_by_video[key] for key in sorted(rows_by_video)]
    write_outputs(args, rows, args.output_stem, merged_from)
    return 0


def split_gpu_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def common_child_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--input-jsonl",
        str(args.input_jsonl),
        "--output-root",
        str(args.output_root),
        "--output-stem",
        args.output_stem,
        "--model-path",
        str(args.model_path),
        "--torch-dtype",
        args.torch_dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.resume:
        command.append("--resume")
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def launch_parallel(args: argparse.Namespace, gpu_ids: list[str]) -> int:
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    base = [sys.executable, str(Path(__file__).resolve()), *common_child_args(args)]
    processes: list[tuple[int, str, subprocess.Popen[Any], Any]] = []

    print(f"[launcher] starting {num_shards} shard workers for {args.output_stem}", flush=True)
    for shard_index, gpu in enumerate(gpu_ids):
        label = f"shard{shard_index:02d}of{num_shards:02d}"
        log_path = log_dir / f"{args.output_stem}.{label}.log"
        command = [
            *base,
            "--device-map",
            "cuda:0",
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_index),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["DS_BUILD_OPS"] = "0"
        log_handle = log_path.open("w", encoding="utf-8")
        print(f"[launcher] shard {shard_index}/{num_shards - 1} uses GPU {gpu} -> {log_path}", flush=True)
        proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT)
        processes.append((shard_index, gpu, proc, log_handle))

    last_done = -1
    while True:
        alive = False
        done_rows = 0
        for shard_index, _gpu, proc, _log_handle in processes:
            if proc.poll() is None:
                alive = True
            shard_stem = output_stem(args.output_stem, shard_index, num_shards)
            shard_jsonl, _json_path, _csv_path = output_paths(args.output_root, shard_stem)
            if shard_jsonl.exists():
                with shard_jsonl.open("r", encoding="utf-8") as handle:
                    done_rows += sum(1 for _ in handle)
        if done_rows != last_done:
            print(f"[launcher] completed rows: {done_rows}", flush=True)
            last_done = done_rows
        if not alive:
            break
        time.sleep(max(1, args.progress_poll_seconds))

    status = 0
    for shard_index, _gpu, proc, log_handle in processes:
        returncode = proc.wait()
        log_handle.close()
        if returncode != 0:
            label = f"shard{shard_index:02d}of{num_shards:02d}"
            print(
                f"[launcher] shard {shard_index} failed with code {returncode}; "
                f"inspect {log_dir / f'{args.output_stem}.{label}.log'}",
                file=sys.stderr,
                flush=True,
            )
            status = returncode or 1
    if status != 0:
        return status

    print("[launcher] merging shard outputs", flush=True)
    merge_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *common_child_args(args),
        "--num-shards",
        str(num_shards),
        "--merge-shards",
    ]
    return subprocess.call(merge_command, cwd=PROJECT_ROOT)


def main() -> int:
    args = build_parser().parse_args()
    args.input_jsonl = args.input_jsonl.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()

    gpu_ids = split_gpu_ids(args.gpus)
    if gpu_ids and not args.merge_shards:
        return launch_parallel(args, gpu_ids)

    if args.merge_shards:
        return merge_shards(args)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    rows = read_jsonl(args.input_jsonl)
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    stem = output_stem(args.output_stem, args.shard_index, args.num_shards)
    jsonl_path, _json_path, _csv_path = output_paths(args.output_root, stem)
    done = existing_rows(jsonl_path) if args.resume else {}
    rows_by_video = dict(done)
    initial_cache = seed_translation_cache(list(done.values()))

    translator: QwenTranslator | None = None
    started = time.time()
    skipped = 0
    for index, row in enumerate(rows, start=1):
        key = str(row.get("video_relpath"))
        if key in done:
            continue
        if translator is None:
            translator = QwenTranslator(args)
            translator.cache.update(initial_cache)
        try:
            out = translated_row(row, translator, args.model_path)
        except (ValueError, json.JSONDecodeError) as exc:
            skipped += 1
            print(f"[{index}/{len(rows)}] SKIP {key}: {exc}", flush=True)
            continue
        rows_by_video[key] = out
        translated_so_far = [rows_by_video[str(item.get("video_relpath"))] for item in rows if str(item.get("video_relpath")) in rows_by_video]
        write_jsonl(jsonl_path, translated_so_far)
        print(f"[{index}/{len(rows)}] {key}", flush=True)

    translated = [rows_by_video[str(row.get("video_relpath"))] for row in rows if str(row.get("video_relpath")) in rows_by_video]
    write_outputs(args, translated, stem)
    print(f"elapsed_seconds={time.time() - started:.1f}, skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RULE_PATH = SCRIPT_DIR / "rule.md"
LIFEBENCH_INFER = PROJECT_ROOT / "evaluation" / "lifebench_infer.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
ENV_ROOT_CANDIDATES = (
    Path("/workspace/anaconda3/envs/lifebench-vlm"),
    Path("/data_4/liuyuan/anaconda3/envs/lifebench-vlm"),
)


@dataclass(frozen=True)
class PerspectiveRule:
    index: int
    name: str
    prompt: str


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str


MODEL_POOL: tuple[ModelSpec, ...] = (
    ModelSpec(model_id="Qwen/Qwen3.5-9B", backend="qwen35vl"),
    ModelSpec(model_id="omni-research/Tarsier2-7b-0115", backend="tarsier2"),
    ModelSpec(model_id="mPLUG/mPLUG-Owl3-7B-241101", backend="mplugowl3"),
    ModelSpec(model_id="DAMO-NLP-SG/VideoLLaMA3-7B", backend="videollama3"),
    ModelSpec(model_id="OpenGVLab/InternVL3_5-8B", backend="internvl35"),
)


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a five-perspective risk review over one video by randomly assigning the "
            "selected five benchmark models to the five perspectives from public_data_filter/rule.md."
        )
    )
    parser.add_argument("--video-path", required=True, type=Path, help="Video path to review.")
    parser.add_argument(
        "--rule-path",
        type=Path,
        default=RULE_PATH,
        help="Markdown file that defines the five perspective prompts.",
    )
    parser.add_argument(
        "--candidate-text",
        help=(
            "Optional candidate behavior text. When omitted, each agent first identifies the "
            "main candidate behavior from the video and then judges whether to retain it."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output directory. Defaults to public_data_filter/outputs/<run-id>.",
    )
    parser.add_argument("--run-id", default=utc_tag(), help="Run id used when --output-root is omitted.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible model-to-perspective assignment.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--merge-size", type=int, default=2)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument(
        "--python-executable",
        type=Path,
        help=(
            "Python executable used to launch lifebench_infer.py. When omitted, the script "
            "tries the lifebench-vlm environment first, then falls back to the current interpreter."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Continue collecting agent outputs after an invocation failure. By default the "
            "workflow aborts on the first failed agent because the five-vote procedure is incomplete."
        ),
    )
    parser.add_argument(
        "--reuse-agent-outputs",
        action="store_true",
        help="Reuse existing per-agent output JSON files in --output-root instead of rerunning those agents.",
    )
    parser.add_argument(
        "--early-stop-majority",
        action="store_true",
        help="Stop after a 3-of-5 keep/drop majority is already determined.",
    )
    return parser


def parse_rule_file(path: Path) -> list[PerspectiveRule]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rules: list[PerspectiveRule] = []
    current_index: int | None = None
    current_name: str | None = None
    current_prompt_lines: list[str] = []
    in_prompt = False

    def flush() -> None:
        nonlocal current_index, current_name, current_prompt_lines, in_prompt
        if current_index is None or current_name is None:
            current_prompt_lines = []
            in_prompt = False
            return
        prompt = "\n".join(line.rstrip() for line in current_prompt_lines).strip()
        if not prompt:
            raise ValueError(f"Missing prompt body for perspective {current_index}. {current_name}")
        rules.append(PerspectiveRule(index=current_index, name=current_name, prompt=prompt))
        current_index = None
        current_name = None
        current_prompt_lines = []
        in_prompt = False

    for raw_line in lines:
        line = raw_line.rstrip()
        header_match = re.match(r"^\s*(\d+)\.\s+(.+?)\s*$", line)
        if header_match:
            flush()
            current_index = int(header_match.group(1))
            current_name = header_match.group(2).strip()
            continue
        if current_index is None:
            continue
        if not in_prompt:
            if re.match(r"^\s*Prompt\s*[：:]\s*$", line):
                in_prompt = True
            continue
        current_prompt_lines.append(line)

    flush()
    if len(rules) != 5:
        raise ValueError(f"Expected exactly 5 perspectives in {path}, found {len(rules)}.")
    return sorted(rules, key=lambda item: item.index)


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root.expanduser().resolve()
    return (DEFAULT_OUTPUT_ROOT / args.run_id).resolve()


def discover_env_root() -> Path | None:
    raw_env_root = os.environ.get("ENV_ROOT")
    if raw_env_root:
        candidate = Path(raw_env_root).expanduser()
        if candidate.is_dir():
            return candidate.resolve()
    for candidate in ENV_ROOT_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_python_executable(args: argparse.Namespace) -> Path:
    if args.python_executable is not None:
        return args.python_executable.expanduser().resolve()
    env_root = discover_env_root()
    if env_root is not None:
        candidate = env_root / "bin" / "python"
        if candidate.exists():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env_root = discover_env_root()
    if env_root is None:
        return env

    env["PATH"] = f"{env_root / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONNOUSERSITE"] = "1"
    project_tmp_dir = PROJECT_ROOT / ".tmp"
    project_tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(project_tmp_dir)
    env["TMP"] = str(project_tmp_dir)
    env["TEMP"] = str(project_tmp_dir)

    shim_lib = env_root / "lib" / "libittnotify.so"
    if shim_lib.exists():
        ld_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{shim_lib}:{ld_preload}" if ld_preload else str(shim_lib)
    return env


def assign_models(rules: list[PerspectiveRule], seed: int | None) -> list[tuple[PerspectiveRule, ModelSpec]]:
    if len(rules) > len(MODEL_POOL):
        raise ValueError(
            f"Need at least {len(rules)} distinct models for the configured perspectives, "
            f"but only {len(MODEL_POOL)} models are available."
        )
    randomizer = random.Random(seed)
    shuffled_models = list(MODEL_POOL)
    randomizer.shuffle(shuffled_models)
    return [(rule, model) for rule, model in zip(rules, shuffled_models)]


def build_prompt(rule: PerspectiveRule, candidate_text: str | None) -> str:
    task_lines = [
        "Review the input video and first decide whether the video contains a risk.",
        "Base your judgment only on the video content. Do not assume facts outside the video, and do not invent extra events.",
    ]
    if candidate_text:
        task_lines.append(f"The candidate behavior to review is: {candidate_text.strip()}")
        task_lines.append("Judge only whether this candidate behavior creates a risk.")
    else:
        task_lines.append("Judge the main visible behavior and scene in the video.")
    task_lines.append("You must answer based on the provided video input. Do not say that you cannot watch or inspect the video.")
    task_lines.append("Output exactly the following English-labeled fields, with no markdown and no extra text:")
    task_lines.append("Risk presence: Yes / No")
    task_lines.append("Level 1 scene: output one scene keyword. Prefer one of: dining room, kitchen, study, balcony, living room, bathroom, yard. If none fits, output your own concise scene keyword.")
    task_lines.append("Level 2 subject: output one subject keyword. Prefer one of: child, older adult, young adult, middle-aged adult. If none fits, output your own concise subject keyword.")
    task_lines.append("Level 3 risk type: if Risk presence is Yes, output one risk keyword. Prefer one of: fall/instability, heat/fire source, collision/crush injury, sharp-object danger, electrical safety, poisoning/accidental ingestion, interpersonal conflict. If none fits, output your own concise risk keyword. If Risk presence is No, output None.")
    task_lines.append("Risk description: if Risk presence is Yes, describe the risk in 1-3 sentences. If Risk presence is No, output None.")
    task_lines.append("Risk time interval: if Risk presence is Yes, localize the risk as seconds like [start,end]. If there are multiple intervals, separate them with semicolons. If the exact interval is unclear, output Unknown. If Risk presence is No, output None.")
    task_lines.append("Solution for person: if Risk presence is Yes, give a concrete action for the person or caregiver. If Risk presence is No, output None.")
    task_lines.append("Solution for hazard source: if Risk presence is Yes, give a concrete action for the hazard source or environment. If Risk presence is No, output None.")
    task_lines.append("Solution to prevent recurrence: if Risk presence is Yes, give a prevention measure to reduce recurrence. If Risk presence is No, output None.")
    task_lines.append("Normal video description: if Risk presence is No, describe the normal video content in 1-3 sentences. If Risk presence is Yes, output None.")
    return f"{rule.prompt.strip()}\n\nTask addendum:\n" + "\n".join(task_lines)


def normalize_field_label(label: str) -> str:
    compact = re.sub(r"[\s_\-]+", "", label.strip().lower())
    aliases = {
        "candidatebehavior": "candidate_behavior",
        "behavior": "candidate_behavior",
        "keepasriskanomaly": "keep",
        "keep": "keep",
        "retain": "keep",
        "whetherkeep": "keep",
        "riskpresence": "risk_presence",
        "hasrisk": "risk_presence",
        "risk": "risk_presence",
        "level1scene": "level1_scene",
        "level1": "level1_scene",
        "scene": "level1_scene",
        "level2subject": "level2_subject",
        "level2": "level2_subject",
        "subject": "level2_subject",
        "level3risktype": "level3_risk_type",
        "level3risk": "level3_risk_type",
        "level3": "level3_risk_type",
        "risktype": "level3_risk_type",
        "riskcategory": "level3_risk_type",
        "riskdescription": "risk_description",
        "riskdesc": "risk_description",
        "risktimeinterval": "risk_time_interval",
        "risktime": "risk_time_interval",
        "timeinterval": "risk_time_interval",
        "timespan": "risk_time_interval",
        "time": "risk_time_interval",
        "solutionforperson": "solution_for_person",
        "personsolution": "solution_for_person",
        "solutionforhazardsource": "solution_for_hazard_source",
        "hazardsourcesolution": "solution_for_hazard_source",
        "solutiontopreventrecurrence": "solution_to_prevent_recurrence",
        "preventrecurrencesolution": "solution_to_prevent_recurrence",
        "recurrenceprevention": "solution_to_prevent_recurrence",
        "normalvideodescription": "normal_video_description",
        "normaldescription": "normal_video_description",
        "reason": "reason",
        "rationale": "reason",
        "explanation": "reason",
    }
    return aliases.get(compact, compact)


def parse_labeled_field(response: str, label: str) -> str:
    target_label = normalize_field_label(label)
    known_labels = {
        "candidate_behavior",
        "keep",
        "risk_presence",
        "level1_scene",
        "level2_subject",
        "level3_risk_type",
        "risk_description",
        "risk_time_interval",
        "solution_for_person",
        "solution_for_hazard_source",
        "solution_to_prevent_recurrence",
        "normal_video_description",
        "reason",
    }
    lines = response.splitlines()
    collected: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        label_match = re.match(r"^([^：:]+)\s*[：:]\s*(.*)$", stripped)
        if not capturing:
            if not label_match:
                continue
            current_label = normalize_field_label(label_match.group(1))
            if current_label != target_label:
                continue
            capturing = True
            collected.append(label_match.group(2).strip())
            continue
        if label_match and normalize_field_label(label_match.group(1)) in known_labels:
            break
        collected.append(stripped)
    return "\n".join(part for part in collected if part).strip()


def parse_keep_decision(response: str) -> dict[str, Any]:
    candidate_behavior = parse_labeled_field(response, "Candidate behavior")
    risk_presence_raw = parse_labeled_field(response, "Risk presence")
    keep_raw = risk_presence_raw or parse_labeled_field(response, "Keep as risk anomaly")
    level1_scene = parse_labeled_field(response, "Level 1 scene")
    level2_subject = parse_labeled_field(response, "Level 2 subject")
    level3_risk_type = parse_labeled_field(response, "Level 3 risk type")
    risk_description = parse_labeled_field(response, "Risk description")
    risk_time_interval = parse_labeled_field(response, "Risk time interval")
    solution_for_person = parse_labeled_field(response, "Solution for person")
    solution_for_hazard_source = parse_labeled_field(response, "Solution for hazard source")
    solution_to_prevent_recurrence = parse_labeled_field(response, "Solution to prevent recurrence")
    normal_video_description = parse_labeled_field(response, "Normal video description")
    reason = parse_labeled_field(response, "Reason")
    normalized = re.sub(r"[\s\.;,。；，]+", "", keep_raw).lower()
    keep: bool | None
    if normalized.startswith("yes") or normalized.startswith("true") or normalized.startswith("keep") or normalized.startswith("risk"):
        keep = True
    elif normalized.startswith("no") or normalized.startswith("false") or normalized.startswith("donotkeep") or normalized.startswith("notkeep") or normalized.startswith("normal"):
        keep = False
    else:
        keep = None
    return {
        "candidate_behavior": candidate_behavior,
        "risk_presence_raw": risk_presence_raw,
        "risk_presence": "Yes" if keep is True else "No" if keep is False else None,
        "keep_raw": keep_raw,
        "keep": keep,
        "level1_scene": level1_scene,
        "level2_subject": level2_subject,
        "level3_risk_type": level3_risk_type,
        "risk_description": risk_description,
        "risk_time_interval": risk_time_interval,
        "solution_for_person": solution_for_person,
        "solution_for_hazard_source": solution_for_hazard_source,
        "solution_to_prevent_recurrence": solution_to_prevent_recurrence,
        "normal_video_description": normal_video_description,
        "reason": reason,
    }


def make_agent_slug(rule: PerspectiveRule) -> str:
    safe_name = re.sub(r"\s+", "_", rule.name.strip())
    return f"{rule.index:02d}_{safe_name}"


def build_infer_command(
    args: argparse.Namespace,
    assignment: tuple[PerspectiveRule, ModelSpec],
    prompt_path: Path,
    output_json: Path,
) -> list[str]:
    rule, model = assignment
    del rule
    command = [
        str(resolve_python_executable(args)),
        str(LIFEBENCH_INFER),
        "--backend",
        model.backend,
        "--model-id",
        model.model_id,
        "--video-path",
        str(args.video_path.resolve()),
        "--prompt-file",
        str(prompt_path),
        "--output-json",
        str(output_json),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--fps",
        str(args.fps),
        "--max-frames",
        str(args.max_frames),
        "--merge-size",
        str(args.merge_size),
        "--attn-implementation",
        args.attn_implementation,
        "--device-map",
        args.device_map,
    ]
    if args.use_flash_attn:
        command.append("--use-flash-attn")
    return command


def run_agent(
    args: argparse.Namespace,
    assignment: tuple[PerspectiveRule, ModelSpec],
    run_dir: Path,
) -> dict[str, Any]:
    rule, model = assignment
    agent_slug = make_agent_slug(rule)
    prompt = build_prompt(rule, args.candidate_text)
    prompt_path = run_dir / "prompts" / f"{agent_slug}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    output_json = run_dir / "agent_outputs" / f"{agent_slug}.json"
    stdout_path = run_dir / "logs" / f"{agent_slug}.stdout.log"
    stderr_path = run_dir / "logs" / f"{agent_slug}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_infer_command(args, assignment, prompt_path, output_json)
    if args.reuse_agent_outputs and output_json.exists():
        reused_result = load_reusable_agent_result(
            args=args,
            assignment=assignment,
            prompt_path=prompt_path,
            output_json=output_json,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command,
        )
        if reused_result is not None:
            return reused_result

    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=build_subprocess_env(),
    )
    elapsed = time.time() - started_at
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    result: dict[str, Any] = {
        "perspective": asdict(rule),
        "model": asdict(model),
        "prompt_file": str(prompt_path),
        "output_json": str(output_json),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
    }

    if completed.returncode != 0:
        result["error"] = {
            "message": f"Agent failed with exit code {completed.returncode}.",
            "stderr_tail": completed.stderr[-2000:],
        }
        return result

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    response = str(payload.get("response", "")).strip()
    parsed_decision = parse_keep_decision(response)
    result["response"] = response
    result["parsed_decision"] = parsed_decision
    result["runner_payload"] = payload
    return result


def load_reusable_agent_result(
    args: argparse.Namespace,
    assignment: tuple[PerspectiveRule, ModelSpec],
    prompt_path: Path,
    output_json: Path,
    stdout_path: Path,
    stderr_path: Path,
    command: list[str],
) -> dict[str, Any] | None:
    rule, model = assignment
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("backend") != model.backend:
        return None
    if payload.get("model_id") != model.model_id:
        return None
    if payload.get("video_path") != str(args.video_path):
        return None
    expected_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if str(payload.get("prompt", "")).strip() != expected_prompt:
        return None
    generation = payload.get("generation")
    if not isinstance(generation, dict):
        return None
    expected_generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "merge_size": args.merge_size,
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            return None
    response = str(payload.get("response", "")).strip()
    if not response:
        return None
    parsed_decision = parse_keep_decision(response)
    if parsed_decision.get("keep") is None:
        return None
    return {
        "perspective": asdict(rule),
        "model": asdict(model),
        "prompt_file": str(prompt_path),
        "output_json": str(output_json),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": command,
        "returncode": 0,
        "elapsed_seconds": 0.0,
        "reused_agent_output": True,
        "response": response,
        "parsed_decision": parsed_decision,
        "runner_payload": payload,
    }


def normalize_keyword_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() in {"none", "n/a", "na", "not applicable", "null"}:
        return ""
    return text


def aggregate_keep_keywords(keep_results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "level1_scene": "level1_scene_keywords",
        "level2_subject": "level2_subject_keywords",
        "level3_risk_type": "level3_risk_type_keywords",
    }
    payload: dict[str, Any] = {}
    for parsed_field, output_field in fields.items():
        counter = Counter(
            keyword
            for item in keep_results
            for keyword in [normalize_keyword_value(item.get("parsed_decision", {}).get(parsed_field))]
            if keyword
        )
        payload[output_field] = [
            {"keyword": keyword, "count": count}
            for keyword, count in counter.most_common()
        ]
        payload[output_field.replace("_keywords", "_top")] = counter.most_common(1)[0][0] if counter else None
    return payload


def aggregate_normal_keywords(drop_results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "level1_scene": "normal_level1_scene_keywords",
        "level2_subject": "normal_level2_subject_keywords",
    }
    payload: dict[str, Any] = {}
    for parsed_field, output_field in fields.items():
        counter = Counter(
            keyword
            for item in drop_results
            for keyword in [normalize_keyword_value(item.get("parsed_decision", {}).get(parsed_field))]
            if keyword
        )
        payload[output_field] = [
            {"keyword": keyword, "count": count}
            for keyword, count in counter.most_common()
        ]
        payload[output_field.replace("_keywords", "_top")] = counter.most_common(1)[0][0] if counter else None
    descriptions = [
        normalize_keyword_value(item.get("parsed_decision", {}).get("normal_video_description"))
        for item in drop_results
    ]
    payload["normal_video_descriptions"] = [description for description in descriptions if description]
    return payload


def summarize_results(agent_results: list[dict[str, Any]], expected_votes: int = 5) -> dict[str, Any]:
    successful = [item for item in agent_results if item.get("returncode") == 0]
    failed = [item for item in agent_results if item.get("returncode") != 0]
    undecidable = [item for item in successful if item.get("parsed_decision", {}).get("keep") is None]
    keep_threshold = expected_votes // 2 + 1
    if failed:
        return {
            "complete": False,
            "keep": None,
            "error": "At least one agent invocation failed; the five-vote review is incomplete.",
            "failed_agents": [item["perspective"]["name"] for item in failed],
        }
    if undecidable:
        return {
            "complete": False,
            "keep": None,
            "risk_presence": None,
            "error": "At least one successful agent did not produce a parseable Risk presence result.",
            "undecidable_agents": [item["perspective"]["name"] for item in undecidable],
        }

    keep_results = [item for item in successful if item["parsed_decision"]["keep"] is True]
    drop_results = [item for item in successful if item["parsed_decision"]["keep"] is False]
    if len(keep_results) >= keep_threshold:
        keep = True
    elif len(drop_results) >= keep_threshold:
        keep = False
    elif len(successful) < expected_votes:
        return {
            "complete": False,
            "keep": None,
            "risk_presence": None,
            "error": "The review stopped before a five-vote majority was determined.",
            "yes_count": len(keep_results),
            "no_count": len(drop_results),
            "threshold": keep_threshold,
            "votes_cast": len(successful),
            "expected_votes": expected_votes,
        }
    else:
        keep = False
    summary = {
        "complete": True,
        "keep": keep,
        "risk_presence": "Yes" if keep else "No",
        "yes_count": len(keep_results),
        "no_count": len(drop_results),
        "threshold": keep_threshold,
        "votes_cast": len(successful),
        "expected_votes": expected_votes,
        "kept_by_perspectives": [item["perspective"]["name"] for item in keep_results],
        "rejected_by_perspectives": [item["perspective"]["name"] for item in drop_results],
    }
    if keep:
        summary.update(aggregate_keep_keywords(keep_results))
    else:
        summary.update(aggregate_normal_keywords(drop_results))
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.video_path = args.video_path.expanduser().resolve()
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {args.video_path}")
    if not LIFEBENCH_INFER.exists():
        raise FileNotFoundError(f"Unified inference entrypoint not found: {LIFEBENCH_INFER}")

    rules = parse_rule_file(args.rule_path.expanduser().resolve())
    assignments = assign_models(rules, args.seed)
    run_dir = resolve_output_root(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "video_path": str(args.video_path),
        "rule_path": str(args.rule_path.expanduser().resolve()),
        "prompt_language": "en",
        "output_schema_version": "risk_full_en_v2",
        "candidate_text": args.candidate_text,
        "seed": args.seed,
        "reuse_agent_outputs": bool(args.reuse_agent_outputs),
        "early_stop_majority": bool(args.early_stop_majority),
        "model_pool": [asdict(item) for item in MODEL_POOL],
        "assignments": [
            {
                "perspective": asdict(rule),
                "model": asdict(model),
            }
            for rule, model in assignments
        ],
    }
    write_json(run_dir / "assignment_manifest.json", manifest)

    agent_results: list[dict[str, Any]] = []
    for assignment in assignments:
        result = run_agent(args, assignment, run_dir)
        agent_results.append(result)
        if result.get("returncode") != 0 and not args.keep_going:
            summary = summarize_results(agent_results, expected_votes=len(assignments))
            final_payload = {
                "manifest": manifest,
                "agent_results": agent_results,
                "majority_vote": summary,
            }
            write_json(run_dir / "review_summary.json", final_payload)
            raise RuntimeError(
                f"Agent {result['perspective']['name']} failed. "
                f"See {result['stderr_log']} for details."
            )
        if args.early_stop_majority:
            summary = summarize_results(agent_results, expected_votes=len(assignments))
            if summary.get("complete") is True:
                break

    final_payload = {
        "manifest": manifest,
        "agent_results": agent_results,
        "majority_vote": summarize_results(agent_results, expected_votes=len(assignments)),
    }
    write_json(run_dir / "review_summary.json", final_payload)
    print(json.dumps(final_payload["majority_vote"], ensure_ascii=False, indent=2))
    print(f"Saved detailed outputs to: {run_dir}")
    return 0


def run_cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())

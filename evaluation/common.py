from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("lifebench.judge")

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for judge mode only
    OpenAI = None


SAFE_ACTION_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("stir fry", "cooking", "cook", "mix"), ("stir fry", "cooking", "cook", "prepare food", "mix", "stir")),
    (("dry hair", "shower area", "bath"), ("dry hair", "shower", "bath", "bathing", "out of bath")),
    (("jog", "run", "pass"), ("jog", "run", "running", "pass", "pass by", "go through")),
    (("get up", "walk to door", "walk"), ("get up", "stand up", "walk to door", "walk", "walking")),
)

LOCATION_ALIASES: dict[str, tuple[str, ...]] = {
    "kitchen": ("kitchen", "stove", "kitchen area"),
    "bathroom": ("bathroom", "shower room", "restroom", "bath"),
    "hallway": ("hallway", "corridor", "passage", "hallway area"),
    "bedroom": ("bedroom", "room", "bedroom area"),
}

SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "adult": ("adult", "adults", "person", "man", "woman", "male", "female"),
    "middle-aged": ("middle-aged", "middle-aged adult", "adult", "person", "man", "woman"),
    "child": ("child", "children", "kid", "kids", "toddler", "little boy", "little girl"),
}

RISK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scald", ("scald", "scalded", "burn", "hot liquid", "high-temperature liquid", "boiling")),
    ("electric", ("electric spark", "spark", "short circuit", "electrical leak", "crackling", "burning smell")),
    ("fire", ("fire", "flame", "open flame", "flames", "fire hazard", "ignite", "ignited", "combustion", "flaring")),
    ("smoke", ("smoke", "smoke rising", "smoke plume", "light smoke", "gray smoke", "overheating", "dry burn")),
    ("breakage", ("break", "broken", "shards", "crack", "debris", "ceramic fragment")),
    ("falling_object", ("slide off", "fall", "drop", "tumble", "descend", "tip over", "tilt")),
    ("pinch", ("pinch", "pinched", "being pinched", "pinched finger", "indentation", "swelling", "whitening")),
    ("trip", ("trip", "tripped", "wire", "toy", "floor mat", "clutter")),
    ("slip", ("slip", "slipping", "slippery", "water stain", "puddle")),
    ("collision", ("kick", "bump", "collide", "hit", "bed edge", "clutter", "stumble")),
    ("fall", ("fall", "fall down", "lose balance", "on the ground")),
)

SCENARIO_RISK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scald", ("hot soup", "hot water", "hot liquid", "high-temperature liquid", "hot water kettle", "hot drink cup")),
    ("electric", ("outlet", "power strip", "power cord", "charger", "electric spark", "spark", "short circuit")),
    ("fire", ("flame", "fire", "ignite", "fire hazard", "open flame", "candle", "tissue")),
    ("smoke", ("smoke", "dry burn", "overheating")),
    ("breakage", ("glass cup", "glass bowl", "glass vase", "break", "broken", "shards")),
    ("falling_object", ("tv sliding from stand", "shelf items falling", "flower pot falling from edge")),
    ("pinch", ("pinched finger", "door gap", "drawer rebound", "cabinet door")),
    ("trip", ("trip", "mat edge", "scattered clutter", "tripped by wire")),
    ("slip", ("slip", "slippery", "wet", "puddle", "water stain")),
    ("fall", ("fall", "fall down", "lose balance", "on the ground")),
)

SCENARIO_SIGNAL_PATTERNS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("tv", "stand"), ("tv", "stand", "slide", "slip off", "fall", "crack", "shards")),
    (("shelf items",), ("shelf", "items", "fall", "drop", "crash")),
    (("flower pot",), ("flower pot", "shake", "slide", "tip over", "fall", "break", "soil", "ceramic fragment")),
    (("glass cup",), ("glass cup", "fall", "break", "shards", "reflect")),
    (("glass bowl",), ("glass bowl", "fall", "break", "shards")),
    (("glass vase",), ("glass vase", "tip over", "break", "shards")),
    (("outlet",), ("outlet", "spark", "electric spark", "crackling", "burning smell", "smoke")),
    (("power cord",), ("power cord", "spark", "electric spark", "charred", "smoke thread", "insulation")),
    (("power strip",), ("power strip", "overheat", "smoke", "current")),
    (("charger",), ("charger", "short circuit", "spark", "smoke")),
    (("appliance", "smoke"), ("appliance", "overheat", "smoke", "smoke plume", "vent")),
    (("dry burn",), ("dry burn", "smoke", "pot", "cough", "burnt smell")),
    (("candle",), ("candle", "flame", "burn", "scorch mark", "blue smoke")),
    (("tissue", "flame"), ("tissue", "flame", "burn", "gray smoke")),
    (("stove", "flame"), ("stove", "flame", "flaring", "fire hazard", "heat radiation")),
    (("hot soup",), ("hot soup", "spill", "splash", "steam", "hot vapor")),
    (("hot water kettle",), ("hot water kettle", "hot water", "spill", "red hot", "hot vapor")),
    (("hot drink cup",), ("hot drink cup", "knock over", "hot liquid", "spill", "scald")),
    (("drawer",), ("drawer", "rebound", "pinch", "indentation", "swelling", "redness")),
    (("cabinet door",), ("cabinet door", "pinch", "indentation", "redness", "door gap")),
    (("door gap",), ("door gap", "pinch", "indentation", "whitening", "swelling")),
    (("floor mat",), ("floor mat", "trip", "curl up", "tripped")),
    (("clutter",), ("clutter", "toy", "obstacle", "trip", "scatter")),
    (("puddle",), ("puddle", "slip", "water stain", "slippery")),
    (("water stain",), ("water stain", "slip", "slippery", "wet")),
)

RISK_ONSET_HINTS: tuple[str, ...] = (
    "start",
    "suddenly",
    "accidentally",
    "instantly",
    "emitting",
    "burst",
    "flash",
    "flare up",
    "flaring",
    "ignite",
    "ignited",
    "combustion",
    "slide",
    "slip",
    "slip off",
    "fall",
    "drop",
    "tip over",
    "tilt",
    "knock over",
    "spill",
    "splash",
    "splatter",
    "break",
    "broken",
    "pinched",
    "rebound",
    "trip",
    "slipping",
    "fall down",
    "lose balance",
    "smoke",
    "spark",
    "electric spark",
    "flame",
)

RISK_LABELS_EN: dict[str, str] = {
    "scald": "Scald",
    "electric": "Electrical Risk",
    "fire": "Fire",
    "smoke": "Smoke Risk",
    "breakage": "Breakage/Cut",
    "falling_object": "Falling Object Strike",
    "pinch": "Pinch Injury",
    "slip": "Slip",
    "trip": "Trip",
    "collision": "Collision Injury",
    "fall": "Fall Down",
    "unknown": "Unknown Risk",
}

RISK_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "heat_source": (
        "heat_source",
        "heat source",
        "scald",
        "scalded",
        "hot liquid",
        "high-temperature liquid",
        "burn",
        "flame",
        "fire",
        "fire hazard",
    ),
    "fall_instability": ("fall_instability", "fall instability", "slip", "slipping", "fall", "fall down", "lose balance", "on the ground", "trip"),
    "collision_or_struck": (
        "collision_or_struck",
        "collision or struck",
        "collision",
        "bump",
        "hit",
        "struck",
        "falling object strike",
        "tv sliding",
        "flower pot falling",
    ),
    "sharp_object": ("sharp_object", "sharp object", "breakage/cut", "glass shards", "shards", "sharp", "cut", "crack"),
    "electrical_safety": ("electrical_safety", "electrical safety", "electrical risk", "electrical leak", "short circuit", "electric spark", "overheat smoke", "smoke"),
    "pinch_injury": ("pinch_injury", "pinch injury", "pinch", "pinched", "pinched finger", "door gap pinch", "drawer rebound"),
    "falling_object": ("falling_object", "falling object strike", "falling object", "drop strike", "high drop", "slide off strike"),
    "fire_or_smoke": ("fire_or_smoke", "fire or smoke", "fire", "flame", "fire hazard", "flames", "smoke", "smoke plume"),
    "unknown": ("unknown", "unknown risk"),
}

LEGACY_RISK_TYPE_TO_DATASET: dict[str, str] = {
    "scald": "heat_source",
    "fire": "heat_source",
    "smoke": "fire_or_smoke",
    "electric": "electrical_safety",
    "breakage": "sharp_object",
    "falling_object": "collision_or_struck",
    "pinch": "pinch_injury",
    "trip": "fall_instability",
    "slip": "fall_instability",
    "collision": "collision_or_struck",
    "fall": "fall_instability",
    "unknown": "unknown",
}

SOLUTION_SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("person_solution", "Solution for the person:"),
    ("hazard_solution", "Solution for the hazard source:"),
    ("prevention_solution", "Follow-up prevention solution:"),
)

GROUND_TRUTH_SOLUTION_SECTION_LABEL_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("person_solution", ("Solution for the person:", "Solution for the person", "对人的解决方案：")),
    ("hazard_solution", ("Solution for the hazard source:", "Solution for the hazard source", "对危险源的解决方案：")),
    (
        "prevention_solution",
        ("Follow-up prevention solution:", "Overall solution:", "Follow-up prevention solution", "后续防止危险复发的解决方案："),
    ),
)

PREDICTION_SOLUTION_SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("person_solution", "Solution for the person:"),
    ("person_solution", "对人的解决方案："),
    ("hazard_solution", "Solution for the hazard source:"),
    ("hazard_solution", "对危险源的解决方案："),
    ("overall_solution", "Overall solution:"),
    ("overall_solution", "整体的解决方案："),
    ("prevention_solution", "Follow-up prevention solution:"),
    ("prevention_solution", "后续防止危险复发的解决方案："),
)

NEGATIVE_RISK_PATTERNS: tuple[str, ...] = (
    "no anomaly",
    "no abnormality",
    "no anomaly detected",
    "no anomaly will occur",
    "no abnormal event occurred",
)

GENERAL_RISK_HINTS: tuple[str, ...] = (
    "suddenly",
    "flame",
    "fire",
    "scald",
    "scalded",
    "burn",
    "hot liquid",
    "high-temperature liquid",
    "boiling",
    "splash",
    "spill",
    "open flame",
    "combustion",
    "smoke",
    "smoke plume",
    "spark",
    "electric spark",
    "short circuit",
    "break",
    "shards",
    "slip off",
    "fall",
    "drop",
    "pinch",
    "pinched",
    "fall down",
    "slip",
    "trip",
    "lose balance",
    "bump",
    "collide",
    "kick",
    "injury",
    "pain",
    "call for help",
)

STOPWORDS: set[str] = {
    "according to the video",
    "in the video",
    "may subsequently",
    "may appear",
    "abnormal behavior",
    "anomaly detection",
    "anomaly prediction",
    "can",
    "therefore",
    "exist",
    "given",
    "output",
    "if",
    "in order to",
    "already occurred",
    "by observing",
    "infer",
    "requires the model",
}

RISK_STATUS_NO_ANOMALY = "normal"
RISK_STATUS_POTENTIAL = "risk_only"
RISK_STATUS_OCCURRED = "abnormal"

RISK_STATUS_ALIASES: dict[str, str] = {
    "noanomaly": RISK_STATUS_NO_ANOMALY,
    "normal": RISK_STATUS_NO_ANOMALY,
    "safe": RISK_STATUS_NO_ANOMALY,
    "norisk": RISK_STATUS_NO_ANOMALY,
    "riskonly": RISK_STATUS_POTENTIAL,
    "risk_only": RISK_STATUS_POTENTIAL,
    "potentialanomaly": RISK_STATUS_POTENTIAL,
    "potentialrisk": RISK_STATUS_POTENTIAL,
    "abnormal": RISK_STATUS_OCCURRED,
    "anomalyoccurred": RISK_STATUS_OCCURRED,
    "occurred": RISK_STATUS_OCCURRED,
}

GROUND_TRUTH_OCCURRED_HINTS: tuple[str, ...] = (
    "[abnormalresult]",
    "abnormal result",
    "fall down",
    "fall",
    "slip",
    "on the ground",
    "bump",
    "collide",
    "struck",
    "pinch",
    "pinched",
    "fire",
    "flame",
    "spark",
    "smoke",
    "smoke plume",
    "break",
    "shards",
    "drop",
    "fall",
    "slip off",
    "spill",
    "splash",
    "splatter",
    "scald",
    "injury",
    "redness",
    "cough",
    "swallow",
    "ingest",
    "land",
    "crack",
)

GROUND_TRUTH_POTENTIAL_HINTS: tuple[str, ...] = (
    "[abnormalsign]",
    "abnormal sign",
    "may subsequently",
    "may develop",
    "may occur",
    "not yet",
    "has not",
    "has not truly",
    "has not reached",
    "nearly",
    "about to",
    "preparing",
    "reaching for",
    "approaching",
    "near",
    "withinreach",
    "notyet",
    "aboutto",
)

TIMELINE_SEGMENT_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "normal": ("normal behavior", "normal"),
    "sign": ("abnormal sign", "risk sign", "warning sign"),
    "result": ("abnormal result", "abnormal outcome", "result"),
}


@dataclass(frozen=True)
class GroundTruthEntry:
    video_id: str
    duration_sec: float
    location: str
    subject: str
    safe_segments: list[dict[str, Any]]
    risk_segments: list[dict[str, Any]]
    summary_gt: dict[str, Any]
    safe_keyword_groups: list[list[str]]
    risk_keyword_groups: list[list[str]]
    video_keyword_groups: list[list[str]]
    solution_sections: dict[str, str]
    source_text: str
    risk_sources: tuple[str, ...] = ()
    abnormal_actions: tuple[str, ...] = ()
    affected_objects: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()
    causal_chain: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"summary_gt": self.summary_gt}

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id, "duration_sec": self.duration_sec,
            "location": self.location, "subject": self.subject,
            "safe_segments": self.safe_segments, "risk_segments": self.risk_segments,
            "summary_gt": self.summary_gt,
            "safe_keyword_groups": self.safe_keyword_groups,
            "risk_keyword_groups": self.risk_keyword_groups,
            "video_keyword_groups": self.video_keyword_groups,
            "solution_sections": self.solution_sections, "source_text": self.source_text,
            "risk_sources": list(self.risk_sources),
            "abnormal_actions": list(self.abnormal_actions),
            "affected_objects": list(self.affected_objects),
            "consequences": list(self.consequences),
            "causal_chain": list(self.causal_chain),
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict[str, Any]) -> "GroundTruthEntry":
        return cls(
            video_id=str(Path(str(payload.get("video_id", ""))).with_suffix("")),
            duration_sec=float(payload.get("duration_sec", 0.0)),
            location=str(payload.get("location", "")),
            subject=str(payload.get("subject", "")),
            safe_segments=list(payload.get("safe_segments", [])),
            risk_segments=list(payload.get("risk_segments", [])),
            summary_gt=dict(payload.get("summary_gt", {})),
            safe_keyword_groups=list(payload.get("safe_keyword_groups", [])),
            risk_keyword_groups=list(payload.get("risk_keyword_groups", [])),
            video_keyword_groups=list(payload.get("video_keyword_groups", [])),
            solution_sections=dict(payload.get("solution_sections", {})),
            source_text=str(payload.get("source_text", "")),
            risk_sources=tuple(payload.get("risk_sources", [])),
            abnormal_actions=tuple(payload.get("abnormal_actions", [])),
            affected_objects=tuple(payload.get("affected_objects", [])),
            consequences=tuple(payload.get("consequences", [])),
            causal_chain=tuple(dict(item) for item in payload.get("causal_chain", [])),
        )


def normalize_risk_type_label(label: str | None) -> str:
    if not label:
        return ""
    compact = compact_text(label).replace("risk type", "").replace("risk", "").strip("::,.; ")
    if not compact:
        return ""
    for canonical, aliases in RISK_TYPE_ALIASES.items():
        candidates = (canonical, *aliases)
        if any(alias and alias in compact for alias in candidates):
            return canonical
    return compact


def risk_type_keyword_groups(label: str) -> list[list[str]]:
    normalized = normalize_risk_type_label(label)
    if not normalized:
        return []
    aliases = RISK_TYPE_ALIASES.get(normalized, ())
    return dedupe_groups([[normalized, *aliases]])


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", "", text)


def split_sentences(text: str | None) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[.;\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def compact_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\\_", "_").replace("\\n", "\n")
    return re.sub(r"\s+", " ", text).strip()


def format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def safe_stem_from_video_path(video_path: str | None) -> str | None:
    if not video_path:
        return None
    return Path(video_path).stem


def normalize_video_id_text(value: str) -> str:
    text = value.strip()
    return text[:-4] if text.lower().endswith(".mp4") else text


_SYMLINK_CACHE: dict[str, str] = {}


def _build_symlink_reverse_map() -> dict[str, str]:
    """Map resolved directory names back to symlink names under public_data/."""
    if _SYMLINK_CACHE:
        return _SYMLINK_CACHE
    try:
        base = Path(__file__).resolve().parent.parent / "data" / "public_data"
        if not base.exists():
            return _SYMLINK_CACHE
        for entry in base.iterdir():
            if entry.is_symlink():
                try:
                    resolved = entry.resolve()
                    # Map the full resolved relative path to symlink name
                    resolved_rel = resolved.relative_to(base)
                    _SYMLINK_CACHE[str(resolved_rel)] = entry.name
                except (OSError, ValueError):
                    pass
    except OSError:
        pass
    return _SYMLINK_CACHE


def match_video_id_from_path(video_path: str | None) -> str | None:
    if not video_path:
        return None
    p = Path(video_path)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "public_data" and i + 1 < len(parts):
            relative = Path(*parts[i + 1:])
            result = str(relative.with_suffix(""))
            # Map resolved symlink path back to symlink name
            reverse = _build_symlink_reverse_map()
            for resolved_rel, symlink_name in reverse.items():
                if result.startswith(resolved_rel + "/"):
                    result = symlink_name + result[len(resolved_rel):]
                    break
                elif result == resolved_rel:
                    result = symlink_name
                    break
            return result
    return p.stem


def normalize_risk_status_label(label: str | None) -> str:
    if not label:
        return ""
    raw = compact_text(label)
    if not raw:
        return ""
    normalized = re.sub(r"[\s_\-]+", "", raw).lower()
    return RISK_STATUS_ALIASES.get(normalized, raw if raw in RISK_STATUS_ALIASES.values() else "")


def infer_risk_status_from_ground_truth(
    video_id: str,
    status_hint: str,
    timeline_text: str,
    risk_segments: list[dict[str, Any]],
) -> str:
    explicit = normalize_risk_status_label(status_hint)
    if explicit:
        return explicit
    for suffix in ("risk_only", "normal", "abnormal"):
        if video_id.endswith(f"_{suffix}") or video_id == suffix:
            explicit = normalize_risk_status_label(suffix)
            if explicit:
                return explicit
    if not risk_segments:
        return RISK_STATUS_NO_ANOMALY

    normalized_timeline = normalize_text(timeline_text)
    if "[abnormalresult]" in normalized_timeline:
        return RISK_STATUS_OCCURRED
    if "[abnormalsign]" in normalized_timeline and "[abnormalresult]" not in normalized_timeline:
        return RISK_STATUS_POTENTIAL

    risk_text = compact_text(" ".join(segment["description"] for segment in risk_segments))
    normalized_risk_text = normalize_text(risk_text)
    if any(normalize_text(hint) in normalized_risk_text for hint in GROUND_TRUTH_OCCURRED_HINTS):
        return RISK_STATUS_OCCURRED
    if any(normalize_text(hint) in normalized_risk_text for hint in GROUND_TRUTH_POTENTIAL_HINTS):
        return RISK_STATUS_POTENTIAL
    return RISK_STATUS_OCCURRED


def align_summary_to_risk_status(summary: dict[str, Any]) -> dict[str, Any]:
    status = normalize_risk_status_label(str(summary.get("risk_status", "")))
    if not status:
        legacy_has_risk = summary.get("has_risk")
        if legacy_has_risk is not None:
            status = RISK_STATUS_OCCURRED if bool(legacy_has_risk) else RISK_STATUS_NO_ANOMALY
    if not status:
        time_spans = summary.get("time_spans") or []
        if time_spans:
            status = RISK_STATUS_OCCURRED
        elif any(compact_text(str(summary.get(field, ""))) for field in ("risk_type", "risk_description", "solution")):
            status = RISK_STATUS_POTENTIAL
        else:
            status = RISK_STATUS_NO_ANOMALY

    aligned = {
        "risk_status": status,
        "has_risk": status != RISK_STATUS_NO_ANOMALY,
        "risk_type": "",
        "video_description": "",
        "safe_segment_desc": "",
        "risk_description": "",
        "solution": "",
        "time_spans": [],
    }
    if status == RISK_STATUS_NO_ANOMALY:
        video_description = compact_text(str(summary.get("video_description", ""))) or compact_text(
            str(summary.get("safe_segment_desc", ""))
        )
        aligned["video_description"] = video_description
        aligned["safe_segment_desc"] = video_description
        return aligned

    aligned["risk_type"] = normalize_risk_type_label(str(summary.get("risk_type", "")))
    aligned["risk_description"] = compact_text(str(summary.get("risk_description", "")))
    aligned["solution"] = compact_text(str(summary.get("solution", "")))
    if status != RISK_STATUS_NO_ANOMALY:
        aligned["time_spans"] = merge_spans(
            [
                [float(item[0]), float(item[1])]
                for item in (summary.get("time_spans") or [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ]
        )
    return aligned


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def extract_keywords(text: str) -> list[str]:
    chunks = re.findall(r"[A-Za-z0-9]+", text)
    results: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk in STOPWORDS:
            continue
        if chunk.isdigit():
            continue
        if chunk not in seen:
            results.append(chunk)
            seen.add(chunk)
    return results


def location_keywords(location: str) -> list[str]:
    prefix = location.split("(", 1)[0].strip()
    aliases = LOCATION_ALIASES.get(prefix, (prefix,))
    return [word for word in aliases if word]


def subject_keywords(subject: str) -> list[str]:
    prefix = re.split(r"[(]", subject, maxsplit=1)[0].strip()
    for key, aliases in SUBJECT_ALIASES.items():
        if key in prefix:
            return [word for word in aliases if word]
    keywords = extract_keywords(prefix)
    return keywords if keywords else [prefix]


def safe_action_keywords(safe_text: str) -> list[str]:
    for patterns, aliases in SAFE_ACTION_GROUPS:
        if any(pattern in safe_text for pattern in patterns):
            return list(aliases)
    keywords = extract_keywords(safe_text)
    return keywords[:4]


def is_scald_risk_text(text: str) -> bool:
    heat_terms = (
        "hot soup",
        "hot water",
        "hot liquid",
        "high-temperature liquid",
        "boiling",
        "scald",
        "scalded",
        "burn",
        "hot vapor",
        "steam",
    )
    spill_terms = (
        "splash",
        "spill",
        "spilled",
        "splatter",
        "splashed",
        "knock over",
        "tip over",
        "spread",
        "diffuse",
        "wet",
    )
    injury_terms = (
        "scald",
        "scalded",
        "burn",
        "redness",
        "pain",
        "splashed on",
    )
    has_heat = any(term in text for term in heat_terms)
    has_spill = any(term in text for term in spill_terms)
    has_injury = any(term in text for term in injury_terms)
    return has_heat and (has_spill or has_injury)


def infer_risk_type_from_scenario(text: str) -> str:
    if not text:
        return "unknown"
    if is_scald_risk_text(text):
        return "scald"
    for risk_type, patterns in SCENARIO_RISK_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return risk_type
    return infer_risk_type(text)


def infer_risk_type(text: str) -> str:
    if is_scald_risk_text(text):
        return "scald"
    for risk_type, patterns in RISK_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return risk_type
    return "unknown"


def scenario_signal_keywords(scenario_text: str, risk_type: str) -> list[str]:
    keywords: list[str] = []
    for fragments, signals in SCENARIO_SIGNAL_PATTERNS:
        if all(fragment in scenario_text for fragment in fragments):
            keywords.extend(signals)
    if risk_type != "unknown":
        for current_type, patterns in RISK_PATTERNS:
            if current_type == risk_type:
                keywords.extend(patterns)
                break
    anchors = (
        "tv",
        "stand",
        "shelf",
        "items",
        "flower pot",
        "glass cup",
        "glass bowl",
        "glass vase",
        "outlet",
        "power strip",
        "power cord",
        "charger",
        "candle",
        "tissue",
        "stove",
        "flame",
        "smoke",
        "drawer",
        "cabinet door",
        "door gap",
        "floor mat",
        "clutter",
        "hot soup",
        "hot water kettle",
        "hot drink cup",
        "hot liquid",
    )
    for anchor in anchors:
        if anchor in scenario_text:
            keywords.append(anchor)
    return list(dict.fromkeys(keyword for keyword in keywords if keyword))


def find_risk_start_index(segments: list[dict[str, Any]], scenario_text: str) -> int | None:
    if not segments:
        return None

    scenario_risk_type = infer_risk_type_from_scenario(scenario_text)
    scenario_keywords = scenario_signal_keywords(scenario_text, scenario_risk_type)

    for index, segment in enumerate(segments):
        description = segment["description"]
        if infer_risk_type(description) != "unknown":
            return index
        if scenario_keywords and any(keyword in description for keyword in scenario_keywords):
            if any(hint in description for hint in RISK_ONSET_HINTS):
                return index

    if not scenario_text:
        return None
    if len(segments) == 1:
        return 0
    for index, segment in enumerate(segments[1:], start=1):
        if scenario_keywords and any(keyword in segment["description"] for keyword in scenario_keywords):
            return index
    return 1


def risk_keyword_groups(risk_segments: list[dict[str, Any]], summary_risk_type: str) -> list[list[str]]:
    groups: list[list[str]] = []
    combined = " ".join(segment["description"] for segment in risk_segments)
    if summary_risk_type == "scald":
        groups.extend(
            [
                ["scald", "scalded", "burn", "redness"],
                ["hot soup", "hot water", "hot liquid", "high-temperature liquid", "boiling"],
                ["splash", "spill", "spilled", "splatter", "splashed"],
            ]
        )
    elif summary_risk_type == "electric":
        groups.extend(
            [
                ["electric spark", "spark", "short circuit", "electrical leak", "crackling", "burning smell"],
                ["outlet", "power strip", "power cord", "charger", "power off"],
            ]
        )
    elif summary_risk_type == "fire":
        groups.extend(
            [
                ["fire", "flame", "open flame", "flames", "fire hazard", "combustion"],
                ["oil pan", "pot", "candle", "tissue", "kitchen"],
            ]
        )
    elif summary_risk_type == "smoke":
        groups.extend(
            [
                ["smoke", "smoke plume", "light smoke", "gray smoke"],
                ["overheating", "dry burn", "burnt smell", "vent", "stove", "appliance"],
            ]
        )
    elif summary_risk_type == "breakage":
        groups.extend(
            [
                ["break", "broken", "shards", "crack", "debris", "ceramic fragment"],
                ["glass cup", "glass bowl", "glass vase", "glass", "floor"],
            ]
        )
    elif summary_risk_type == "falling_object":
        groups.extend(
            [
                ["slide off", "fall", "drop", "tumble", "descend", "tip over"],
                ["flower pot", "tv", "shelf items", "stand", "countertop"],
            ]
        )
    elif summary_risk_type == "pinch":
        groups.extend(
            [
                ["pinch", "pinched", "indentation", "swelling", "whitening"],
                ["door gap", "drawer", "cabinet door", "door frame", "finger", "toe"],
            ]
        )
    elif summary_risk_type == "slip":
        groups.extend(
            [
                ["slip", "slipping", "fall down", "lose balance"],
                ["bathroom", "slippery", "water stain", "floor"],
            ]
        )
    elif summary_risk_type == "trip":
        groups.extend(
            [
                ["trip", "fall down", "lose balance"],
                ["hallway", "corridor"],
                ["toy", "wire", "obstacle"],
            ]
        )
    elif summary_risk_type == "collision":
        groups.extend(
            [
                ["kick", "bump", "collide", "stumble", "fall down"],
                ["bedroom", "bedside", "clutter", "bed edge"],
            ]
        )
    elif summary_risk_type == "fall":
        groups.extend(
            [
                ["fall down", "lose balance", "on the ground"],
            ]
        )
    else:
        for risk_type, patterns in RISK_PATTERNS:
            if risk_type == summary_risk_type:
                groups.append(list(patterns))
                break

    for keyword in extract_keywords(combined)[:3]:
        groups.append([keyword])
    return dedupe_groups(groups)


def normalize_solution_section(text: str) -> str:
    text = compact_text(text)
    text = text.strip("::,.; ")
    text = text.replace(".", ";")
    return text.strip(";, ")


def parse_solution_sections(text: str) -> dict[str, str]:
    sections = {key: "" for key, _ in SOLUTION_SECTION_LABELS}
    normalized = compact_text(text)
    if not normalized:
        return sections

    positions: list[tuple[str, int, str]] = []
    for key, labels in GROUND_TRUTH_SOLUTION_SECTION_LABEL_ALIASES:
        matched_label = ""
        position = -1
        for label in labels:
            normalized_label = compact_text(label)
            current_position = normalized.find(normalized_label)
            if current_position == -1:
                continue
            if position == -1 or current_position < position:
                position = current_position
                matched_label = normalized_label
        if position == -1:
            return sections
        positions.append((key, position, matched_label))
    positions.sort(key=lambda item: item[1])

    for index, (key, start_pos, label) in enumerate(positions):
        start = start_pos + len(label)
        end = positions[index + 1][1] if index + 1 < len(positions) else len(normalized)
        sections[key] = normalize_solution_section(normalized[start:end])
    return sections


def parse_prediction_solution_sections(text: str) -> dict[str, str]:
    sections = {
        "person_solution": "",
        "hazard_solution": "",
        "overall_solution": "",
    }
    normalized = compact_text(text)
    if not normalized:
        return sections

    positions: list[tuple[str, int, str]] = []
    for key, label in PREDICTION_SOLUTION_SECTION_LABELS:
        position = normalized.find(label)
        if position == -1:
            continue
        positions.append((key, position, label))
    if not positions:
        return sections
    positions.sort(key=lambda item: item[1])

    for index, (key, start_pos, label) in enumerate(positions):
        start = start_pos + len(label)
        end = positions[index + 1][1] if index + 1 < len(positions) else len(normalized)
        value = normalize_solution_section(normalized[start:end])
        if key == "prevention_solution":
            sections["overall_solution"] = value
        elif key in sections:
            sections[key] = value
    return sections


def build_solution_summary_text(solution_sections: dict[str, str]) -> str:
    ordered_sections = (
        ("person_solution", "Solution for the person: "),
        ("hazard_solution", "Solution for the hazard source: "),
        ("prevention_solution", "Overall solution: "),
    )
    pieces = [
        f"{label}{solution_sections[key]}"
        for key, label in ordered_sections
        if compact_text(solution_sections.get(key, ""))
    ]
    return " ".join(pieces)


def merge_segment_descriptions(segments: list[dict[str, Any]]) -> str:
    return compact_text(
        "; ".join(
            compact_text(segment.get("description", ""))
            for segment in segments
            if compact_text(segment.get("description", ""))
        )
    )

def dedupe_groups(groups: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[list[str]] = []
    for group in groups:
        normalized = tuple(dict.fromkeys(word for word in group if word))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(list(normalized))
    return deduped


def text_keyword_groups(text: str, max_keywords: int = 12) -> list[list[str]]:
    keywords = extract_keywords(compact_text(text))[:max_keywords]
    return [[keyword] for keyword in keywords if keyword]


def normalize_timeline_segment_label(label: str | None) -> str:
    normalized = normalize_text(label)
    if not normalized:
        return ""
    for canonical, aliases in TIMELINE_SEGMENT_LABEL_ALIASES.items():
        if normalized == canonical:
            return canonical
        if any(normalized == normalize_text(alias) for alias in aliases):
            return canonical
    return normalized


def parse_timeline(timeline_text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    bracket_pattern = re.compile(
        r"\[(?P<label>[^\[\]]+)\]\s*"
        r"\[(?P<start>\d+(?:\.\d+)?)\s*[-–—~to]+\s*(?P<end>\d+(?:\.\d+)?)\s*(?:s|sec)\]\s*"
        r"(?P<description>.*?)(?=(?:\s*\[[^\[\]]+\]\s*\[\d+(?:\.\d+)?\s*[-–—~to]+\s*\d+(?:\.\d+)?\s*(?:s|sec)\])|$)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    for match in bracket_pattern.finditer(timeline_text):
        start = parse_float(match.group("start"))
        end = parse_float(match.group("end"))
        label = normalize_timeline_segment_label(match.group("label"))
        description = compact_text(match.group("description")).strip("[];, ")
        segments.append({"start": start, "end": end, "description": description, "label": label})
    if segments:
        return segments

    segment_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–—~to]+\s*(\d+(?:\.\d+)?)s(.*?)(?=(?:\s*\d+(?:\.\d+)?)\s*[-–—~to]+\s*(?:\d+(?:\.\d+)?)s|$)",
        flags=re.DOTALL,
    )
    for match in segment_pattern.finditer(timeline_text):
        start = parse_float(match.group(1))
        end = parse_float(match.group(2))
        description = compact_text(match.group(3))
        segments.append({"start": start, "end": end, "description": description, "label": ""})
    return segments


def split_safe_and_risk_segments(
    segments: list[dict[str, Any]],
    scenario_text: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not segments:
        return [], []

    if any(segment.get("label") for segment in segments):
        safe_segments = [segment for segment in segments if segment.get("label") == "normal"]
        risk_segments = [segment for segment in segments if segment.get("label") in {"sign", "result"}]
        if safe_segments or risk_segments:
            return safe_segments, risk_segments

    risk_index = find_risk_start_index(segments, scenario_text)

    if risk_index is None:
        return segments, []
    return segments[:risk_index], segments[risk_index:]


def parse_ground_truth_line(line: str) -> GroundTruthEntry:
    if "\t" in line and "|" not in line:
        return parse_ground_truth_tsv_line(line)
    video_id, payload = line.split("|", 1)
    payload = payload.strip()
    duration_match = re.search(r"(\d+(?:\.\d+)?)s", payload)
    duration_sec = parse_float(duration_match.group(1)) if duration_match else 0.0
    location_match = re.search(r"Location: ([^.]+).", payload)
    subject_match = re.search(r"Subject: ([^.]+).", payload)
    scenario_match = re.search(r"Scenario: ([^.]+).", payload)
    timeline_match = re.search(r"Timeline: ([^.]+)", payload)

    location = location_match.group(1).strip() if location_match else ""
    subject = subject_match.group(1).strip() if subject_match else ""
    scenario = scenario_match.group(1).strip() if scenario_match else ""
    timeline_text = timeline_match.group(1).strip() if timeline_match else ""
    timeline_segments = parse_timeline(timeline_text)
    safe_segments, risk_segments = split_safe_and_risk_segments(timeline_segments, scenario)
    full_video_description = merge_segment_descriptions(timeline_segments)
    risk_status = infer_risk_status_from_ground_truth(video_id.strip(), "", timeline_text, risk_segments)

    risk_type = infer_risk_type(" ".join(segment["description"] for segment in risk_segments))
    if risk_type == "unknown":
        risk_type = infer_risk_type_from_scenario(scenario)
    display_risk_type = LEGACY_RISK_TYPE_TO_DATASET.get(risk_type, risk_type)
    risk_start = min((segment["start"] for segment in risk_segments), default=0.0)
    risk_end = max((segment["end"] for segment in risk_segments), default=0.0)
    solution_sections = parse_solution_sections(payload)
    solution_summary = build_solution_summary_text(solution_sections)
    risk_description = merge_segment_descriptions(risk_segments)
    time_interval = [risk_start, risk_end] if risk_segments else []
    summary_gt = align_summary_to_risk_status(
        {
            "risk_status": risk_status,
            "video_description": full_video_description,
            "risk_type": display_risk_type,
            "risk_description": risk_description,
            "solution": solution_summary,
            "time_spans": [time_interval] if risk_status != RISK_STATUS_NO_ANOMALY and risk_segments else [],
        }
    )
    summary_gt["solution_sections"] = solution_sections
    if risk_status != RISK_STATUS_NO_ANOMALY and time_interval:
        summary_gt["time_interval"] = time_interval

    safe_groups: list[list[str]] = []
    if location:
        safe_groups.append(location_keywords(location))
    if subject:
        safe_groups.append(subject_keywords(subject))
    if safe_segments:
        safe_groups.append(safe_action_keywords(safe_segments[0]["description"]))

    video_groups: list[list[str]] = []
    if location:
        video_groups.append(location_keywords(location))
    if subject:
        video_groups.append(subject_keywords(subject))
    video_groups.extend(text_keyword_groups(full_video_description))

    return GroundTruthEntry(
        video_id=normalize_video_id_text(video_id),
        duration_sec=duration_sec,
        location=location,
        subject=subject,
        safe_segments=safe_segments,
        risk_segments=risk_segments,
        summary_gt=summary_gt,
        safe_keyword_groups=dedupe_groups(safe_groups),
        risk_keyword_groups=risk_keyword_groups(risk_segments, risk_type),
        video_keyword_groups=dedupe_groups(video_groups),
        solution_sections=solution_sections,
        source_text=payload,
    )


def _parse_comma_list(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _parse_causal_chain(text: str) -> tuple[dict[str, str], ...]:
    chain: list[dict[str, str]] = []
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        segs = [s.strip() for s in part.split(":", 2)]
        if len(segs) == 3:
            chain.append({"source": segs[0], "action": segs[1], "consequence": segs[2]})
        elif len(segs) == 2:
            chain.append({"source": segs[0], "action": segs[1], "consequence": ""})
        elif len(segs) == 1:
            chain.append({"source": segs[0], "action": "", "consequence": ""})
    return tuple(chain)


def parse_ground_truth_tsv_line(line: str) -> GroundTruthEntry:
    parts = [part.strip() for part in line.split("\t")]
    if len(parts) < 3:
        raise ValueError(f"Unsupported TSV ground truth line: {line}")
    explicit_status_hint = ""
    if len(parts) >= 5 and normalize_risk_status_label(parts[2]):
        video_id, path_text, explicit_status_hint, timeline_text, solution_text = parts[:5]
    elif len(parts) >= 4:
        video_id, path_text, timeline_text, solution_text = parts[:4]
    else:
        video_id, path_text, timeline_text = parts[:3]
        solution_text = ""

    risk_sources: tuple[str, ...] = ()
    abnormal_actions: tuple[str, ...] = ()
    consequences: tuple[str, ...] = ()
    causal_chain: tuple[dict[str, str], ...] = ()
    if len(parts) >= 6 and parts[5]:
        risk_sources = _parse_comma_list(parts[5])
    if len(parts) >= 7 and parts[6]:
        abnormal_actions = _parse_comma_list(parts[6])
    if len(parts) >= 8 and parts[7]:
        consequences = _parse_comma_list(parts[7])
    if len(parts) >= 9 and parts[8]:
        causal_chain = _parse_causal_chain(parts[8])

    path_parts = [part.strip() for part in path_text.split("/") if part.strip()]
    location = path_parts[0] if len(path_parts) >= 1 else ""
    subject = path_parts[1] if len(path_parts) >= 2 else ""
    annotated_risk_type = normalize_risk_type_label(path_parts[2] if len(path_parts) >= 3 else "")
    scenario = path_parts[3] if len(path_parts) >= 4 else ""

    timeline_segments = parse_timeline(timeline_text)
    safe_segments, risk_segments = split_safe_and_risk_segments(timeline_segments, scenario)
    full_video_description = merge_segment_descriptions(timeline_segments) or compact_text(timeline_text)
    duration_sec = max((segment["end"] for segment in timeline_segments), default=0.0)
    risk_status = infer_risk_status_from_ground_truth(video_id.strip(), explicit_status_hint, timeline_text, risk_segments)

    inferred_risk_type = infer_risk_type(" ".join(segment["description"] for segment in risk_segments))
    if inferred_risk_type == "unknown":
        inferred_risk_type = infer_risk_type_from_scenario(scenario)
    display_risk_type = annotated_risk_type or LEGACY_RISK_TYPE_TO_DATASET.get(inferred_risk_type, inferred_risk_type)
    risk_start = min((segment["start"] for segment in risk_segments), default=0.0)
    risk_end = max((segment["end"] for segment in risk_segments), default=0.0)
    solution_sections = parse_solution_sections(solution_text)
    solution_summary = build_solution_summary_text(solution_sections)
    risk_description = merge_segment_descriptions(risk_segments)
    time_interval = [risk_start, risk_end] if risk_segments else []
    summary_gt = align_summary_to_risk_status(
        {
            "risk_status": risk_status,
            "video_description": full_video_description,
            "risk_type": display_risk_type,
            "risk_description": risk_description,
            "solution": solution_summary,
            "time_spans": [time_interval] if risk_status != RISK_STATUS_NO_ANOMALY and risk_segments else [],
        }
    )
    summary_gt["solution_sections"] = solution_sections
    if risk_status != RISK_STATUS_NO_ANOMALY and time_interval:
        summary_gt["time_interval"] = time_interval

    safe_groups: list[list[str]] = []
    if location:
        safe_groups.append(location_keywords(location))
    if subject:
        safe_groups.append(subject_keywords(subject))
    if safe_segments:
        safe_groups.append(safe_action_keywords(safe_segments[0]["description"]))

    video_groups: list[list[str]] = []
    if location:
        video_groups.append(location_keywords(location))
    if subject:
        video_groups.append(subject_keywords(subject))
    video_groups.extend(text_keyword_groups(full_video_description))

    affected_objects = tuple()
    if subject:
        affected_objects = (subject,)
    if risk_status != RISK_STATUS_NO_ANOMALY:
        affected_objects = tuple(
            dict.fromkeys(list(affected_objects) + [location] if location else list(affected_objects))
        )

    return GroundTruthEntry(
        video_id=normalize_video_id_text(video_id),
        duration_sec=duration_sec,
        location=location,
        subject=subject,
        safe_segments=safe_segments,
        risk_segments=risk_segments,
        summary_gt=summary_gt,
        safe_keyword_groups=dedupe_groups(safe_groups),
        risk_keyword_groups=risk_keyword_groups(risk_segments, inferred_risk_type),
        video_keyword_groups=dedupe_groups(video_groups),
        solution_sections=solution_sections,
        source_text=line.strip(),
        risk_sources=risk_sources,
        abnormal_actions=abnormal_actions,
        affected_objects=affected_objects,
        consequences=consequences,
        causal_chain=causal_chain,
    )


def load_ground_truth(path: Path) -> dict[str, GroundTruthEntry]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entries: list[dict[str, Any]]
        if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
            raw_entries = [item for item in payload["entries"] if isinstance(item, dict)]
        elif isinstance(payload, list):
            raw_entries = [item for item in payload if isinstance(item, dict)]
        else:
            raise ValueError(f"Unsupported parsed ground-truth JSON format: {path}")
        return {
            entry.video_id: entry
            for entry in (GroundTruthEntry.from_serializable_dict(item) for item in raw_entries)
            if entry.video_id
        }

    entries: dict[str, GroundTruthEntry] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entry = parse_ground_truth_line(line)
        entries[entry.video_id] = entry
    return entries


def extract_section(response: str, marker: str) -> str:
    if marker not in response:
        return ""
    _, remainder = response.split(marker, 1)
    return remainder.strip()


def extract_json_string_field(response: str, field_name: str) -> str:
    normalized = response.replace("\\_", "_")
    patterns = (
        rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"',
        rf"'{field_name}'\s*:\s*'((?:[^'\\]|\\.)*)'",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.DOTALL)
        if match:
            value = match.group(1)
            value = value.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
            return compact_text(value)
    return ""


def extract_risk_description(response: str) -> str:
    json_value = extract_json_string_field(response, "risk_description")
    if json_value:
        return json_value
    patterns = (
        r"abnormal behavior(?: category)?(?: is| was)[\"\']?([^.;\n\"\']+)",
        r"potential abnormal behavior(?: is| was)[\"\']?([^.;\n\"\']+)",
        r"possible abnormal event(?: is| was)[\"\']?([^.;\n\"\']+)",
        r"there is abnormal behavior, namely([^.;\n]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            value = compact_text(match.group(1))
            if value and "if" not in value.lower():
                return value.strip("\"':;, ")
    risk_sentences = [sentence for sentence in split_sentences(response) if any(hint.lower() in sentence.lower() for hint in GENERAL_RISK_HINTS)]
    if risk_sentences:
        return ". ".join(risk_sentences[:2]).strip("\"':;, ")
    return ""


def extract_risk_type(response: str) -> str:
    json_value = extract_json_string_field(response, "risk_type")
    normalized = normalize_risk_type_label(json_value)
    if normalized:
        return normalized
    patterns = (
        r"risk type(?: is| was|:)\s*([^\n,.;]+)",
        r"belongs to ([^\n,.;]+) risk",
    )
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            normalized = normalize_risk_type_label(match.group(1))
            if normalized:
                return normalized
    inferred = infer_risk_type(response)
    return LEGACY_RISK_TYPE_TO_DATASET.get(inferred, "")


def extract_solution_text(response: str) -> str:
    normalized = response.replace("\\n", "\n")
    json_value = extract_json_string_field(normalized, "solution")
    if json_value:
        return json_value
    explicit_positions: list[tuple[str, str, int]] = []
    for key, label in PREDICTION_SOLUTION_SECTION_LABELS:
        start = normalized.find(label)
        if start != -1:
            explicit_positions.append((key, label, start))
    if explicit_positions:
        explicit_positions.sort(key=lambda item: item[2])
        explicit_sections: list[str] = []
        for index, (_, label, start) in enumerate(explicit_positions):
            content_start = start + len(label)
            content_end = len(normalized)
            if index + 1 < len(explicit_positions):
                content_end = explicit_positions[index + 1][2]
            value = compact_text(normalized[content_start:content_end]).strip("\"':;, {}[]")
            if value:
                explicit_sections.append(f"{label}{value}")
        if explicit_sections:
            return " ".join(explicit_sections)

    action_hints = ("turn off", "cover", "extinguish", "clean", "clear", "move away", "seek help", "call for help", "call emergency", "lift", "careful", "attention", "medical", "check", "anti-slip")
    patterns = (
        r"to eliminate the risk,? (?:one )?can([^.;\n]+)",
        r"to resolve the occurred abnormal event,? (?:one )?can([^.;\n]+)",
        r"solution(?: is| was)?([^.;\n]+)",
        r"suggest(?:ed|ion)?([^.;\n]+)",
    )
    candidates: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, response):
            value = compact_text(match.group(1)).strip("\"':;, ")
            if value and "output" not in value.lower() and any(hint in value.lower() for hint in action_hints):
                candidates.append(value)
    if not candidates:
        return ""
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def extract_time_spans(response: str) -> list[list[float]]:
    spans: list[list[float]] = []
    normalized = response.replace("\\_", "_")
    for match in re.finditer(r'"time_spans"\s*:\s*\[\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', normalized):
        start = parse_float(match.group(1))
        end = parse_float(match.group(2))
        if end < start:
            start, end = end, start
        spans.append([start, end])
    patterns = (
        r"(\d+(?:\.\d+)?)\s*(?:-|to|~|—)\s*(\d+(?:\.\d+)?)\s*(?:sec|s)",
        r"(\d+(?:\.\d+)?)s\s*(?:-|to|~|—)\s*(\d+(?:\.\d+)?)s",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, response):
            start = parse_float(match.group(1))
            end = parse_float(match.group(2))
            if end < start:
                start, end = end, start
            spans.append([start, end])
    return merge_spans(spans)


def extract_risk_status(response: str) -> str:
    json_value = normalize_risk_status_label(extract_json_string_field(response, "risk_status"))
    if json_value:
        return json_value

    normalized = compact_text(response)
    compact_normalized = normalize_text(normalized)
    for alias, canonical in sorted(RISK_STATUS_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_normalized = normalize_text(alias)
        if alias_normalized and alias_normalized in compact_normalized:
            return canonical

    has_video_description_field = bool(extract_json_string_field(response, "video_description"))
    has_risk_type_field = bool(extract_json_string_field(response, "risk_type"))
    has_risk_description_field = bool(extract_json_string_field(response, "risk_description"))
    has_solution_field = bool(extract_json_string_field(response, "solution"))
    has_time_spans = bool(extract_time_spans(response))

    explicit_false = re.search(r'"has_?risk"\s*:\s*false', normalized, flags=re.IGNORECASE)
    explicit_true = re.search(r'"has_?risk"\s*:\s*true', normalized, flags=re.IGNORECASE)
    if explicit_false and not explicit_true:
        return RISK_STATUS_NO_ANOMALY
    if explicit_true:
        return RISK_STATUS_OCCURRED

    if has_time_spans:
        return RISK_STATUS_OCCURRED
    if has_video_description_field and not (has_risk_type_field or has_risk_description_field or has_solution_field):
        return RISK_STATUS_NO_ANOMALY
    if (has_risk_type_field or has_risk_description_field or has_solution_field) and not has_video_description_field:
        if any(pattern in normalized for pattern in ("may subsequently", "may develop into anomaly", "not yet occurred", "has not truly occurred", "risk only", "risk_only")):
            return RISK_STATUS_POTENTIAL
        return RISK_STATUS_POTENTIAL

    if any(pattern in normalized for pattern in ("may subsequently", "may develop into anomaly", "not yet occurred", "has not truly occurred")):
        return RISK_STATUS_POTENTIAL
    if any(pattern in normalized for pattern in NEGATIVE_RISK_PATTERNS):
        return RISK_STATUS_NO_ANOMALY
    if any(pattern in normalized.lower() for pattern in GENERAL_RISK_HINTS):
        return RISK_STATUS_OCCURRED
    if has_video_description_field:
        return RISK_STATUS_NO_ANOMALY
    return ""


def extract_has_risk(response: str) -> bool:
    return extract_risk_status(response) != RISK_STATUS_NO_ANOMALY


def extract_safe_context(response: str) -> str:
    json_value = extract_json_string_field(response, "video_description")
    if json_value:
        return json_value
    sentences = split_sentences(response)
    collected: list[str] = []
    for sentence in sentences:
        if infer_risk_type(sentence) != "unknown" or "abnormal behavior" in sentence.lower():
            break
        collected.append(sentence)
    return ". ".join(collected)


def structured_prediction(raw_payload: dict[str, Any]) -> dict[str, Any]:
    response = raw_payload.get("response", "")
    parsed_response = raw_payload.get("parsed_response")
    extracted_video_description = extract_json_string_field(response, "video_description")
    heuristic_summary = align_summary_to_risk_status(
        {
            "risk_status": extract_risk_status(response),
            "has_risk": extract_has_risk(response),
            "risk_type": extract_risk_type(response),
            "video_description": extracted_video_description or compact_text(response),
            "safe_segment_desc": extracted_video_description or extract_safe_context(response),
            "risk_description": extract_risk_description(response),
            "solution": extract_solution_text(response),
            "time_spans": extract_time_spans(response),
        }
    )
    if isinstance(parsed_response, dict):
        aligned_parsed_summary = align_summary_to_risk_status(
            {
                "risk_status": parsed_response.get("risk_status"),
                "has_risk": parsed_response.get("has_risk"),
                "risk_type": parsed_response.get("risk_type"),
                "video_description": parsed_response.get("video_description"),
                "safe_segment_desc": parsed_response.get("safe_segment_desc"),
                "risk_description": parsed_response.get("risk_description"),
                "solution": parsed_response.get("solution"),
                "time_spans": parsed_response.get("time_spans"),
            }
        )
        if aligned_parsed_summary["risk_status"] == RISK_STATUS_NO_ANOMALY and not aligned_parsed_summary["video_description"]:
            aligned_parsed_summary["video_description"] = heuristic_summary["video_description"]
            aligned_parsed_summary["safe_segment_desc"] = heuristic_summary["safe_segment_desc"]
        if aligned_parsed_summary["risk_status"] != RISK_STATUS_NO_ANOMALY:
            aligned_parsed_summary["risk_type"] = aligned_parsed_summary["risk_type"] or heuristic_summary["risk_type"]
            aligned_parsed_summary["risk_description"] = aligned_parsed_summary["risk_description"] or heuristic_summary["risk_description"]
            aligned_parsed_summary["solution"] = aligned_parsed_summary["solution"] or heuristic_summary["solution"]
            if aligned_parsed_summary["risk_status"] != RISK_STATUS_NO_ANOMALY and not aligned_parsed_summary["time_spans"]:
                aligned_parsed_summary["time_spans"] = heuristic_summary["time_spans"]
        return {
            "video_id": match_video_id_from_path(raw_payload.get("video_path")),
            "model_id": raw_payload.get("model_id"),
            "backend": raw_payload.get("backend"),
            "raw_response": response,
            "summary": aligned_parsed_summary,
        }
    return {
        "video_id": match_video_id_from_path(raw_payload.get("video_path")),
        "model_id": raw_payload.get("model_id"),
        "backend": raw_payload.get("backend"),
        "raw_response": response,
        "summary": heuristic_summary,
    }


def merge_spans(spans: list[list[float]]) -> list[list[float]]:
    cleaned = sorted(([float(start), float(end)] for start, end in spans if end > start), key=lambda item: item[0])
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return merged


def span_union_length(spans: list[list[float]]) -> float:
    return sum(end - start for start, end in merge_spans(spans))


def span_intersection_length(a_spans: list[list[float]], b_spans: list[list[float]]) -> float:
    a_merged = merge_spans(a_spans)
    b_merged = merge_spans(b_spans)
    total = 0.0
    i = 0
    j = 0
    while i < len(a_merged) and j < len(b_merged):
        a_start, a_end = a_merged[i]
        b_start, b_end = b_merged[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if end > start:
            total += end - start
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return total


def time_iou(gt_spans: list[list[float]], pred_spans: list[list[float]]) -> float:
    if not gt_spans and not pred_spans:
        return 1.0
    if not gt_spans or not pred_spans:
        return 0.0
    intersection = span_intersection_length(gt_spans, pred_spans)
    union = span_union_length(gt_spans) + span_union_length(pred_spans) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_keyword_groups(text: str, groups: list[list[str]]) -> tuple[int, int, list[list[str]]]:
    normalized = normalize_text(text)
    matched_groups: list[list[str]] = []
    matched = 0
    for group in groups:
        if any(normalize_text(keyword) in normalized for keyword in group if keyword):
            matched += 1
            matched_groups.append(group)
    return matched, len(groups), matched_groups


def heuristic_group_score(text: str, groups: list[list[str]], max_score: float = 5.0) -> tuple[float, dict[str, Any]]:
    matched, total, matched_groups = match_keyword_groups(text, groups)
    if total == 0:
        return 0.0, {"matched_groups": [], "match_ratio": 0.0}
    score = max_score * matched / total
    return round(score, 3), {"matched_groups": matched_groups, "match_ratio": round(matched / total, 4)}


def summary_completeness(summary: dict[str, Any]) -> dict[str, Any]:
    status = normalize_risk_status_label(str(summary.get("risk_status", "")))
    fields = {"risk_status": bool(status)}
    if status == RISK_STATUS_NO_ANOMALY:
        fields.update(
            {
                "video_description": bool(summary.get("video_description")),
                "risk_type": True,
                "risk_description": True,
                "solution": True,
                "time_spans": True,
            }
        )
    elif status == RISK_STATUS_POTENTIAL:
        fields.update(
            {
                "video_description": True,
                "risk_type": bool(summary.get("risk_type")),
                "risk_description": bool(summary.get("risk_description")),
                "solution": bool(summary.get("solution")),
                "time_spans": bool(summary.get("time_spans")),
            }
        )
    elif status == RISK_STATUS_OCCURRED:
        fields.update(
            {
                "video_description": True,
                "risk_type": bool(summary.get("risk_type")),
                "risk_description": bool(summary.get("risk_description")),
                "solution": bool(summary.get("solution")),
                "time_spans": bool(summary.get("time_spans")),
            }
        )
    else:
        fields.update(
            {
                "video_description": bool(summary.get("video_description")),
                "risk_type": bool(summary.get("risk_type")),
                "risk_description": bool(summary.get("risk_description")),
                "solution": bool(summary.get("solution")),
                "time_spans": bool(summary.get("time_spans")),
            }
        )
    ratio = sum(1 for value in fields.values() if value) / len(fields)
    return {"fields": fields, "ratio": round(ratio, 4)}


def weighted_overall_score(
    risk_status: float = 0.0,
    risk_type_score: float = 0.0,
    iou_score: float = 0.0,
    video_desc_score: float = 0.0,
    risk_desc_score: float = 0.0,
    solution_score: float = 0.0,
    risk_source_score: float = 0.0,
    abnormal_action_score: float = 0.0,
    affected_object_score: float = 0.0,
    consequence_score: float = 0.0,
    factual_boundary_score: float = 0.0,
    causal_chain_score: float = 0.0,
    event_summary_score: float = 0.0,
    narrative_coherence_score: float = 0.0,
    unsupported_claim_score: float = 0.0,
) -> dict[str, Any]:
    def norm(v: float, scale: float = 5.0) -> float:
        return min(max(v / scale, 0.0), 1.0)

    perception = (
        0.25 * risk_status
        + 0.25 * norm(risk_type_score)
        + 0.20 * norm(risk_source_score)
        + 0.20 * norm(abnormal_action_score)
        + 0.10 * norm(affected_object_score)
    )
    cognition = (
        0.25 * norm(risk_desc_score)
        + 0.25 * norm(consequence_score)
        + 0.25 * norm(factual_boundary_score)
        + 0.25 * norm(causal_chain_score)
    )
    summarization = (
        0.30 * norm(video_desc_score)
        + 0.30 * norm(event_summary_score)
        + 0.20 * norm(narrative_coherence_score)
        + 0.20 * norm(unsupported_claim_score)
    )
    temporal = 0.0 if iou_score is None else float(iou_score)
    intervention = norm(solution_score)

    overall = (
        0.25 * perception
        + 0.20 * cognition
        + 0.15 * summarization
        + 0.15 * temporal
        + 0.25 * intervention
    )
    return {
        "overall_score": round(overall * 100.0, 3),
        "category_scores": {
            "perception": round(perception * 100.0, 3),
            "cognition": round(cognition * 100.0, 3),
            "summarization": round(summarization * 100.0, 3),
            "temporal_grounding": round(temporal * 100.0, 3),
            "intervention_planning": round(intervention * 100.0, 3),
        },
    }


def risk_type_heuristic(gt: GroundTruthEntry, structured_pred: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    gt_status = normalize_risk_status_label(gt.summary_gt.get("risk_status", ""))
    gt_label = normalize_risk_type_label(gt.summary_gt.get("risk_type", ""))
    pred_label = normalize_risk_type_label(structured_pred["summary"].get("risk_type", ""))
    if gt_status == RISK_STATUS_NO_ANOMALY:
        score = 5.0 if not pred_label else 0.0
        return score, {"gt_risk_type": gt_label, "pred_risk_type": pred_label, "no_risk_case": True}
    gt_groups = risk_type_keyword_groups(gt_label)
    if not gt_groups:
        return 0.0, {"gt_risk_type": gt_label, "pred_risk_type": pred_label}
    score, meta = heuristic_group_score(pred_label, gt_groups)
    meta["gt_risk_type"] = gt_label
    meta["pred_risk_type"] = pred_label
    return score, meta


def safe_description_heuristic(gt: GroundTruthEntry, structured_pred: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    response = structured_pred["raw_response"]
    safe_text = (
        structured_pred["summary"].get("safe_segment_desc")
        or structured_pred["summary"].get("video_description")
        or response
    )
    return heuristic_group_score(safe_text, gt.safe_keyword_groups)


def video_description_heuristic(gt: GroundTruthEntry, structured_pred: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    gt_status = normalize_risk_status_label(gt.summary_gt.get("risk_status", ""))
    pred_video_text = compact_text(str(structured_pred["summary"].get("video_description", "")))
    if gt_status != RISK_STATUS_NO_ANOMALY:
        score = 5.0 if not pred_video_text else 0.0
        return score, {"not_applicable": True, "prediction_present": bool(pred_video_text)}
    video_text = structured_pred["summary"].get("video_description") or structured_pred["raw_response"]
    return heuristic_group_score(video_text, gt.video_keyword_groups)


def risk_description_heuristic(gt: GroundTruthEntry, structured_pred: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    gt_status = normalize_risk_status_label(gt.summary_gt.get("risk_status", ""))
    pred_risk_text = compact_text(str(structured_pred["summary"].get("risk_description", "")))
    if gt_status == RISK_STATUS_NO_ANOMALY:
        score = 5.0 if not pred_risk_text else 0.0
        return score, {"not_applicable": True, "prediction_present": bool(pred_risk_text)}
    risk_text = structured_pred["summary"].get("risk_description") or structured_pred["raw_response"]
    return heuristic_group_score(risk_text, gt.risk_keyword_groups)


def solution_heuristic(gt: GroundTruthEntry, structured_pred: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    gt_status = normalize_risk_status_label(gt.summary_gt.get("risk_status", ""))
    solution_text = structured_pred["summary"].get("solution") or structured_pred["raw_response"]
    if gt_status == RISK_STATUS_NO_ANOMALY:
        final_score = 5.0 if not compact_text(str(structured_pred["summary"].get("solution", ""))) else 0.0
        return final_score, {
            "prediction_sections": parse_prediction_solution_sections(solution_text),
            "section_scores": {
                "person_solution": final_score,
                "hazard_solution": final_score,
                "overall_solution": final_score,
            },
            "section_meta": {
                "person_solution": {"not_applicable": True},
                "hazard_solution": {"not_applicable": True},
                "overall_solution": {"not_applicable": True},
            },
        }
    prediction_sections = parse_prediction_solution_sections(solution_text)
    gt_section_map = {
        "person_solution": gt.solution_sections.get("person_solution", ""),
        "hazard_solution": gt.solution_sections.get("hazard_solution", ""),
        "overall_solution": gt.solution_sections.get("prevention_solution", ""),
    }

    section_scores: dict[str, float] = {}
    section_meta: dict[str, Any] = {}
    score_values: list[float] = []
    for key, gt_text in gt_section_map.items():
        pred_text = prediction_sections.get(key, "")
        if gt_text:
            score, meta = heuristic_group_score(pred_text, text_keyword_groups(gt_text, max_keywords=8))
        else:
            score, meta = (0.0, {"matched_groups": [], "match_ratio": 0.0})
        section_scores[key] = score
        section_meta[key] = meta
        score_values.append(score)

    final_score = round(sum(score_values) / len(score_values), 3) if score_values else 0.0
    return final_score, {
        "prediction_sections": prediction_sections,
        "section_scores": section_scores,
        "section_meta": section_meta,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text!r}")
    obj = json.loads(match.group(0))
    # Tolerate string-encoded scores / missing score key.
    if "score" not in obj:
        for cand in ("分数", "评分", "value", "rating"):
            if cand in obj:
                obj["score"] = obj[cand]
                break
    if "score" in obj:
        try:
            obj["score"] = float(obj["score"])
        except (TypeError, ValueError):
            raise ValueError(f"Judge returned non-numeric score: {obj.get('score')!r}")
    return obj


# ── Temporal Grounding (derived from time_spans) ──


def start_time_error(gt_spans: list[list[float]], pred_spans: list[list[float]]) -> float:
    if not gt_spans or not pred_spans:
        return 0.0
    gt_start = gt_spans[0][0]
    pred_start = pred_spans[0][0]
    return round(abs(gt_start - pred_start), 3)


def end_time_error(gt_spans: list[list[float]], pred_spans: list[list[float]]) -> float:
    if not gt_spans or not pred_spans:
        return 0.0
    gt_end = gt_spans[-1][-1]
    pred_end = pred_spans[-1][-1]
    return round(abs(gt_end - pred_end), 3)


def duration_error(gt_spans: list[list[float]], pred_spans: list[list[float]]) -> float:
    if not gt_spans or not pred_spans:
        return 0.0
    gt_dur = span_union_length(gt_spans)
    pred_dur = span_union_length(pred_spans)
    return round(abs(gt_dur - pred_dur), 3)


def start_position_analysis(
    gt_spans: list[list[float]], pred_spans: list[list[float]], total_duration: float = 0.0,
) -> str:
    if not gt_spans or not pred_spans or total_duration <= 0:
        return "unknown"
    gt_start = gt_spans[0][0]
    ratio = gt_start / total_duration
    if ratio < 0.33:
        return "early"
    if ratio < 0.66:
        return "middle"
    return "late"


def compute_temporal_metrics(
    gt_spans: list[list[float]], pred_spans: list[list[float]], total_duration: float = 0.0,
) -> dict[str, Any]:
    return {
        "start_time_error": start_time_error(gt_spans, pred_spans),
        "end_time_error": end_time_error(gt_spans, pred_spans),
        "duration_error": duration_error(gt_spans, pred_spans),
        "start_position": start_position_analysis(gt_spans, pred_spans, total_duration),
    }


# ── Perception scoring ──


def risk_source_recognition_heuristic(
    gt: GroundTruthEntry, structured_pred: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    gt_sources = gt.risk_sources
    if not gt_sources:
        return 0.0, {"not_annotated": True, "message": "risk_sources not annotated in GT"}
    pred_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["raw_response"]
    )
    matched: list[str] = []
    for source in gt_sources:
        if normalize_text(source) in normalize_text(pred_text):
            matched.append(source)
    ratio = len(matched) / len(gt_sources) if gt_sources else 0.0
    score = round(5.0 * ratio, 3)
    return score, {"matched": matched, "total": len(gt_sources), "match_ratio": ratio}


def abnormal_action_recognition_heuristic(
    gt: GroundTruthEntry, structured_pred: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    gt_actions = gt.abnormal_actions
    if not gt_actions:
        return 0.0, {"not_annotated": True, "message": "abnormal_actions not annotated in GT"}
    pred_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["raw_response"]
    )
    matched: list[str] = []
    for action in gt_actions:
        if normalize_text(action) in normalize_text(pred_text):
            matched.append(action)
    ratio = len(matched) / len(gt_actions) if gt_actions else 0.0
    score = round(5.0 * ratio, 3)
    return score, {"matched": matched, "total": len(gt_actions), "match_ratio": ratio}


def affected_object_recognition_heuristic(
    gt: GroundTruthEntry, structured_pred: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    gt_objects = gt.affected_objects
    if not gt_objects:
        return 0.0, {"not_annotated": True, "message": "affected_objects not annotated in GT"}
    pred_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["summary"].get("video_description", "")
        or structured_pred["raw_response"]
    )
    matched: list[str] = []
    for obj in gt_objects:
        if normalize_text(obj) in normalize_text(pred_text):
            matched.append(obj)
    ratio = len(matched) / len(gt_objects) if gt_objects else 0.0
    score = round(5.0 * ratio, 3)
    return score, {"matched": matched, "total": len(gt_objects), "match_ratio": ratio}


# ── Cognition scoring (heuristic + LLM wrappers) ──


def consequence_understanding_heuristic(
    gt: GroundTruthEntry, structured_pred: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    gt_consequences = gt.consequences
    if not gt_consequences:
        return 0.0, {"not_annotated": True, "message": "consequences not annotated in GT"}
    pred_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["summary"].get("solution", "")
        or structured_pred["raw_response"]
    )
    matched: list[str] = []
    for c in gt_consequences:
        if normalize_text(c) in normalize_text(pred_text):
            matched.append(c)
    ratio = len(matched) / len(gt_consequences) if gt_consequences else 0.0
    score = round(5.0 * ratio, 3)
    return score, {"matched": matched, "total": len(gt_consequences), "match_ratio": ratio}


def causal_chain_understanding_heuristic(
    gt: GroundTruthEntry, structured_pred: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    chain = gt.causal_chain
    if not chain:
        return 0.0, {"not_annotated": True, "message": "causal_chain not annotated in GT"}
    pred_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["raw_response"]
    )
    matched = 0
    for link in chain:
        source = link.get("source", "")
        action = link.get("action", "")
        consequence = link.get("consequence", "")
        norm = normalize_text(pred_text)
        source_ok = not source or normalize_text(source) in norm
        action_ok = not action or normalize_text(action) in norm
        consequence_ok = not consequence or normalize_text(consequence) in norm
        if source_ok and action_ok and consequence_ok:
            matched += 1
    ratio = matched / len(chain) if chain else 0.0
    score = round(5.0 * ratio, 3)
    return score, {"matched_links": matched, "total_links": len(chain), "match_ratio": ratio}


# ── Summarization scoring (LLM-based wrappers) ──


def event_summary_completeness_llm(
    gt: GroundTruthEntry, structured_pred: dict[str, Any], judge: Any,
) -> dict[str, Any]:
    summary = structured_pred["summary"]
    pred_text = summary.get("risk_description") or summary.get("video_description") or ""
    gt_text = gt.summary_gt.get("risk_description") or gt.summary_gt.get("video_description") or ""
    return judge_or_heuristic(
        "event_summary_completeness",
        {
            "risk_status": gt.summary_gt.get("risk_status", ""),
            "gt_summary": gt_text,
            "risk_type": gt.summary_gt.get("risk_type", ""),
            "solution_sections": gt.solution_sections,
        },
        {"pred_summary": pred_text, "raw_response": structured_pred["raw_response"]},
        0.0,
        {"fallback": True},
        judge,
    )


def event_narrative_coherence_llm(
    gt: GroundTruthEntry, structured_pred: dict[str, Any], judge: Any,
) -> dict[str, Any]:
    summary = structured_pred["summary"]
    pred_text = summary.get("risk_description") or summary.get("video_description") or ""
    return judge_or_heuristic(
        "event_narrative_coherence",
        {
            "risk_status": gt.summary_gt.get("risk_status", ""),
            "gt_risk_type": gt.summary_gt.get("risk_type", ""),
        },
        {"pred_narrative": pred_text, "raw_response": structured_pred["raw_response"]},
        0.0,
        {"fallback": True},
        judge,
    )


def unsupported_factual_claim_llm(
    gt: GroundTruthEntry, structured_pred: dict[str, Any], judge: Any,
) -> dict[str, Any]:
    summary = structured_pred["summary"]
    pred_text = summary.get("risk_description") or summary.get("video_description") or ""
    gt_text = gt.summary_gt.get("risk_description") or gt.summary_gt.get("video_description") or ""
    return judge_or_heuristic(
        "unsupported_factual_claim",
        {
            "risk_status": gt.summary_gt.get("risk_status", ""),
            "gt_summary": gt_text,
            "gt_risk_type": gt.summary_gt.get("risk_type", ""),
        },
        {"pred_summary": pred_text, "raw_response": structured_pred["raw_response"]},
        0.0,
        {"fallback": True, "method": "lower_is_better"},
        judge,
    )


# ── Cognition LLM wrappers (use judge for semantic scoring) ──


def consequence_understanding_llm(
    gt: GroundTruthEntry, structured_pred: dict[str, Any], judge: Any,
) -> dict[str, Any]:
    risk_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["summary"].get("solution", "")
        or ""
    )
    gt_risk_text = gt.summary_gt.get("risk_description", "")
    return judge_or_heuristic(
        "consequence_understanding",
        {
            "risk_status": gt.summary_gt.get("risk_status", ""),
            "gt_risk_description": gt_risk_text,
            "gt_consequences": list(gt.consequences),
            "gt_solution_sections": gt.solution_sections,
        },
        {"pred_risk_description": risk_text, "raw_response": structured_pred["raw_response"]},
        consequence_understanding_heuristic(gt, structured_pred)[0],
        {"fallback": True},
        judge,
    )


def factual_boundary_error_llm(
    gt: GroundTruthEntry, structured_pred: dict[str, Any], judge: Any,
) -> dict[str, Any]:
    risk_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["summary"].get("video_description", "")
        or ""
    )
    return judge_or_heuristic(
        "factual_boundary_error",
        {
            "gt_risk_status": gt.summary_gt.get("risk_status", ""),
            "gt_summary": gt.summary_gt.get("risk_description", "")
            or gt.summary_gt.get("video_description", ""),
        },
        {"pred_text": risk_text, "raw_response": structured_pred["raw_response"]},
        0.0,
        {"fallback": True, "method": "lower_is_better"},
        judge,
    )


def causal_chain_understanding_llm(
    gt: GroundTruthEntry, structured_pred: dict[str, Any], judge: Any,
) -> dict[str, Any]:
    risk_text = (
        structured_pred["summary"].get("risk_description", "")
        or structured_pred["raw_response"]
    )
    gt_risk_text = gt.summary_gt.get("risk_description", "")
    return judge_or_heuristic(
        "causal_chain_understanding",
        {
            "gt_risk_description": gt_risk_text,
            "gt_causal_chain": list(gt.causal_chain),
            "risk_status": gt.summary_gt.get("risk_status", ""),
        },
        {"pred_risk_description": risk_text, "raw_response": structured_pred["raw_response"]},
        causal_chain_understanding_heuristic(gt, structured_pred)[0],
        {"fallback": True},
        judge,
    )


class OpenAICompatibleJudge:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required for judge mode 'openai'.")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = float(os.environ.get("EVAL_JUDGE_TIMEOUT_SECONDS", "60"))
        self.max_retries = int(os.environ.get("EVAL_JUDGE_MAX_RETRIES", "0"))
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            self.request_url = normalized
            self.base_url = normalized[: -len("/chat/completions")]
        else:
            self.base_url = normalized
            self.request_url = normalized + "/chat/completions"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

    def score(self, dimension: str, ground_truth: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "你是一个视频安全评测专家。根据给定的 Ground Truth 和模型输出，"
            "仅对指定维度进行打分。"
            "必须且只返回一个 JSON 对象，格式为 {\"score\": 0到5之间的数字, \"reason\": \"简短的中文解释\"}。"
            "不要输出任何额外内容、不要使用 markdown 代码块。"
        )
        user_prompt = {
            "dimension": dimension,
            "rubric": {
                "risk_type": "评估风险类型标签是否抓住了主要风险类别。允许同义词或上位词，但不应是完全不同的风险。",
                "video_description": "评估整体视频描述是否准确，关注是否覆盖了人物、场景和主要动作/变化。不因遗漏不可见的细节而扣分。",
                "risk_description": "评估风险段落的描述是否准确，重点关注风险类型、动作链条和场景一致性。",
                "risk_description_normal": "评估模型是否正确报告了正常活动（无异常）。模型不应声称存在风险。risk_description 为空是正确且可接受的。如果没有声称存在风险则打 5 分，如果错误地声称存在风险则打 0 分。",
                "solution_person": "评估对人的解决方案是否将人员安全放在首位、是否可操作、是否符合风险后果。",
                "solution_hazard": "评估对危险源的解决方案是否能切断或消除危险、是否具体且可执行。",
                "solution_overall": "评估整体/预防方案是否包含复盘和预防措施、是否符合家庭场景的常识。",

                "consequence_understanding": "评估模型是否正确理解风险/异常事件的后果。对比 GT 的后果列表与模型预测的后果列表。允许同义描述。5 分表示所有后果都被正确识别，0 分表示完全未识别。",
                "factual_boundary_error": "评估模型是否混淆了潜在风险征兆与实际已发生的事件。如果模型正确地停留在事实边界内（不将风险描述为已发生的事实）则打 5 分。如果模型将潜在风险描述为已发生的事件则打 0 分。",
                "causal_chain_understanding": "评估模型是否理解因果链：风险源 → 异常行为 → 后果。5 分表示完整因果链描述正确，0 分表示没有捕捉到因果关系。",

                "event_summary_completeness": "评估模型的摘要是否覆盖所有关键要素：场景、主体、风险源、异常行为和后果。5 分表示所有要素都存在，0 分表示毫无覆盖。",
                "event_narrative_coherence": "评估模型是否按逻辑顺序描述事件（时间顺序，因前果后）。5 分表示完全连贯，0 分表示完全混乱。",
                "unsupported_factual_claim": "评估模型是否包含 Ground Truth 中不存在的虚假事实（幻觉）。5 分表示无虚假声明，0 分表示存在严重幻觉。错误越少分数越高。",

                "risk_source_recognition": "评估模型是否正确识别了风险源。对比 GT 的 risk_sources 列表与模型预测的 risk_sources 列表。允许同义词或改写（如\"灶台\"与\"厨房电器\"）。5 分表示所有风险源都被正确识别，0 分表示完全未识别。",
                "abnormal_action_recognition": "评估模型是否正确识别了异常行为。对比 GT 的 abnormal_actions 列表与模型预测的 abnormal_actions 列表。允许同义词或改写（如\"摔倒\"与\"失去平衡后倒下\"）。5 分表示所有异常行为都被正确识别，0 分表示完全未识别。",
                "affected_object_recognition": "评估模型是否正确识别了受影响的对象/主体。从以下选项中选择：儿童、年轻人、老年人、中年人、所有人。对比 GT 的 affected_objects 列表与模型预测的 affected_objects 列表。允许同义词（如\"小孩\"→\"儿童\"，\"青年\"→\"年轻人\"，\"老人\"→\"老年人\"）。5 分表示所有受影响对象都被正确识别，0 分表示完全未识别。",
            }.get(dimension, dimension),
            "ground_truth": ground_truth,
            "prediction": prediction,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_json(user_prompt)},
        ]
        last_exc: Exception | None = None
        content = ""
        # Attempt 1: ask for strict JSON via response_format (supported by Ollama /
        # OpenAI-compatible servers). Attempt 2 (fallback): drop response_format and
        # rely on the prompt + robust regex parsing, in case the server rejects the param.
        for attempt in range(2):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "temperature": 0,
                    "messages": messages,
                    "timeout": self.timeout_seconds,
                }
                if attempt == 0:
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("Judge returned empty content.")
                result = parse_json_object(content)
                if "score" not in result:
                    raise ValueError(f"Judge response missing 'score': {content!r}")
                result["score"] = float(result["score"])
                result["judge_mode"] = "openai"
                return result
            except Exception as exc:  # noqa: BLE001 - we re-raise after logging
                last_exc = exc
                logger.warning(
                    "Judge call failed (model=%s, dimension=%s, attempt=%d): %s | raw=%r",
                    self.model, dimension, attempt, exc, content[:200],
                )
        raise last_exc  # type: ignore[misc]


class MultiEndpointJudge:
    """Round-robin dispatch across multiple OpenAI-compatible endpoints.

    Each thread gets a stable starting index (based on thread id) and walks
    the endpoint list with thread-local monotonic counter. This spreads load
    evenly across endpoints when many worker threads are active, and works
    for both:
      * a single Ollama process serving multiple models / parallel slots, and
      * N independent Ollama processes (one per GPU) bound to distinct ports.
    """

    def __init__(self, model: str, api_key: str, base_urls: list[str]) -> None:
        if not base_urls:
            raise ValueError("MultiEndpointJudge requires at least one base_url.")
        self.model = model
        self.api_key = api_key
        self.base_url = ",".join(base_urls)
        self.base_urls = list(base_urls)
        self._judges: list[OpenAICompatibleJudge] = [
            OpenAICompatibleJudge(model=model, api_key=api_key, base_url=u) for u in base_urls
        ]
        self._thread_counters: dict[int, int] = {}
        self._lock = threading.Lock()

    def _next_index(self) -> int:
        tid = threading.get_ident()
        with self._lock:
            idx = self._thread_counters.get(tid, 0)
            self._thread_counters[tid] = idx + 1
        return idx % len(self._judges)

    def score(self, dimension: str, ground_truth: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        n = len(self._judges)
        for attempt in range(n):
            judge = self._judges[self._next_index()]
            try:
                return judge.score(dimension, ground_truth, prediction)
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("MultiEndpointJudge has no endpoints configured.")


def build_judge(
    mode: str,
    model: str | None,
    api_key: str | None,
    base_url: str | list[str] | tuple[str, ...] | None,
) -> "OpenAICompatibleJudge | MultiEndpointJudge | None":
    if mode == "heuristic":
        return None
    llm_default = None
    try:
        from llm import get_default_llm_config

        llm_default = get_default_llm_config()
    except Exception:
        llm_default = None

    api_key = (
        api_key
        or os.environ.get("EVAL_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or (llm_default.api_key if llm_default is not None else None)
    )
    model = model or os.environ.get("EVAL_JUDGE_MODEL") or (llm_default.model if llm_default is not None else None)
    base_url = (
        base_url
        or os.environ.get("EVAL_JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or (llm_default.base_url if llm_default is not None else None)
    )
    if mode == "openai" and (not api_key or not model or not base_url):
        raise ValueError("Judge mode 'openai' requires API key, model, and base URL.")
    if not api_key or not model or not base_url:
        return None
    if isinstance(base_url, (list, tuple)):
        urls = [u for u in base_url if u]
    else:
        urls = [base_url]
    if len(urls) == 1:
        return OpenAICompatibleJudge(model=model, api_key=api_key, base_url=urls[0])
    return MultiEndpointJudge(model=model, api_key=api_key, base_urls=urls)


def judge_or_heuristic(
    dimension: str,
    gt_payload: dict[str, Any],
    pred_payload: dict[str, Any],
    heuristic_score: float,
    heuristic_meta: dict[str, Any],
    judge: OpenAICompatibleJudge | None,
    record: dict[str, Any] | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    if record is not None and key is not None:
        record.setdefault(key, {})
    if judge is None:
        if record is not None and key is not None:
            record[key] = {"judge_mode": "heuristic", "reason": "heuristic keyword match"}
        return {
            "score": heuristic_score,
            "reason": "heuristic keyword match",
            "judge_mode": "heuristic",
            "heuristic_meta": heuristic_meta,
        }
    try:
        result = judge.score(dimension, gt_payload, pred_payload)
        result["heuristic_meta"] = heuristic_meta
        if record is not None and key is not None:
            record[key] = {"judge_mode": result.get("judge_mode", "openai"), "reason": result.get("reason", "")}
        return result
    except Exception as exc:
        logger.error(
            "Judge FALLBACK for dimension=%s -> using default score=%s. Error: %s",
            dimension, heuristic_score, exc,
        )
        if record is not None and key is not None:
            record[key] = {"judge_mode": "heuristic_fallback", "reason": f"judge fallback to heuristic: {exc}"}
        return {
            "score": heuristic_score,
            "reason": f"judge fallback to heuristic: {exc}",
            "judge_mode": "heuristic_fallback",
            "heuristic_meta": heuristic_meta,
        }

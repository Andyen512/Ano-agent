#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from app import APP_DIR, PROJECT_ROOT, init_db


PUBLIC_DATA_ROOT = PROJECT_ROOT / "data" / "public_data"


def norm(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def most_common(values: list[str]) -> str:
    values = [norm(v) for v in values if norm(v)]
    return Counter(values).most_common(1)[0][0] if values else ""


def first_nonempty(values: list[str]) -> str:
    for value in values:
        value = norm(value)
        if value:
            return value
    return ""


def video_key_from_path(video_path: str) -> str:
    path = Path(video_path).resolve()
    try:
        return str(path.relative_to(PUBLIC_DATA_ROOT.resolve()))
    except ValueError:
        return path.name


def site_and_id(video_key: str) -> tuple[str, str]:
    stem = Path(video_key).stem
    if "__" in stem:
        site, video_id = stem.split("__", 1)
        return site, video_id
    return "public_data", stem


def source_output_root_from_summary(summary_path: Path) -> str:
    for parent in summary_path.parents:
        if parent.name == "reviews":
            return str(parent.parent)
    return str(summary_path.parent)


def prediction_from_strict_row(row: dict[str, Any], strict_path: Path) -> dict[str, Any] | None:
    video_key = norm(row.get("video_relpath"))
    video_path = norm(row.get("video_path"))
    summary_json = norm(row.get("summary_json"))
    if not video_key or not video_path:
        return None

    dataset = video_key.split("/", 1)[0] if "/" in video_key else "web"
    site, video_id = site_and_id(video_key)
    majority = row.get("source_majority_vote") or {}
    source_summary = Path(summary_json) if summary_json else strict_path
    meta = {
        "source": "qwen_strict_schema",
        "strict_schema_jsonl": str(strict_path),
        "formatted_text": row.get("formatted_text"),
        "extraction_method": row.get("extraction_method"),
        "validation_errors": row.get("validation_errors") or [],
    }

    return {
        "video_key": video_key,
        "site": site,
        "video_id": video_id,
        "dataset": dataset,
        "file_path": video_path,
        "source_output_root": source_output_root_from_summary(source_summary),
        "summary_json": summary_json or str(strict_path),
        "risk": norm(row.get("Risk")),
        "level1_scene": norm(row.get("Level 1 scene")),
        "level2_subject": norm(row.get("Level 2 subject")),
        "level3_risk_type": norm(row.get("Level 3 risk type")),
        "description": norm(row.get("Normal-video description or risk description")),
        "risk_localization": norm(row.get("Risk localization")),
        "solution_for_person": norm(row.get("Solutions-For person")),
        "solution_for_hazard_source": norm(row.get("Solutions-For hazard source")),
        "solution_prevent_recurrence": norm(row.get("Solutions-Prevent recurrence")),
        "yes_count": majority.get("yes_count"),
        "no_count": majority.get("no_count"),
        "votes_cast": majority.get("votes_cast"),
        "complete": 1 if majority.get("complete") is True else 0,
        "majority_json": json.dumps(majority, ensure_ascii=False),
        "agents_json": json.dumps(meta, ensure_ascii=False),
    }


def prediction_from_summary(summary_path: Path) -> dict[str, Any] | None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest") or {}
    majority = payload.get("majority_vote") or {}
    agents = payload.get("agent_results") or []
    if majority.get("complete") is not True:
        return None

    video_path = str(manifest.get("video_path") or "")
    if not video_path:
        return None
    video_key = video_key_from_path(video_path)
    dataset = video_key.split("/", 1)[0] if "/" in video_key else "web"
    site, video_id = site_and_id(video_key)
    risk = "Yes" if majority.get("risk_presence") == "Yes" or majority.get("keep") is True else "No"

    parsed = [item.get("parsed_decision") or {} for item in agents if item.get("returncode") == 0]
    yes_items = [item for item in parsed if item.get("risk_presence") == "Yes" or item.get("keep") is True]
    no_items = [item for item in parsed if item.get("risk_presence") == "No" or item.get("keep") is False]
    selected = yes_items if risk == "Yes" else no_items
    fallback = parsed

    level1 = majority.get("level1_scene_top") or majority.get("normal_level1_scene_top") or most_common(
        [item.get("level1_scene") for item in selected or fallback]
    )
    level2 = majority.get("level2_subject_top") or majority.get("normal_level2_subject_top") or most_common(
        [item.get("level2_subject") for item in selected or fallback]
    )
    level3 = majority.get("level3_risk_type_top") or most_common(
        [item.get("level3_risk_type") for item in selected or fallback]
    )
    if risk == "No":
        level3 = "None"

    description = first_nonempty(
        [item.get("risk_description") for item in yes_items]
        if risk == "Yes"
        else [*(majority.get("normal_video_descriptions") or []), *[item.get("normal_video_description") for item in no_items]]
    )
    risk_localization = first_nonempty([item.get("risk_time_interval") for item in yes_items]) if risk == "Yes" else "None"
    solution_for_person = first_nonempty([item.get("solution_for_person") for item in yes_items]) if risk == "Yes" else "None"
    solution_for_hazard_source = first_nonempty([item.get("solution_for_hazard_source") for item in yes_items]) if risk == "Yes" else "None"
    solution_prevent_recurrence = first_nonempty([item.get("solution_to_prevent_recurrence") for item in yes_items]) if risk == "Yes" else "None"

    return {
        "video_key": video_key,
        "site": site,
        "video_id": video_id,
        "dataset": dataset,
        "file_path": video_path,
        "source_output_root": source_output_root_from_summary(summary_path),
        "summary_json": str(summary_path),
        "risk": risk,
        "level1_scene": level1,
        "level2_subject": level2,
        "level3_risk_type": level3,
        "description": description,
        "risk_localization": risk_localization,
        "solution_for_person": solution_for_person,
        "solution_for_hazard_source": solution_for_hazard_source,
        "solution_prevent_recurrence": solution_prevent_recurrence,
        "yes_count": majority.get("yes_count"),
        "no_count": majority.get("no_count"),
        "votes_cast": majority.get("votes_cast"),
        "complete": 1,
        "majority_json": json.dumps(majority, ensure_ascii=False),
        "agents_json": json.dumps({"source": "review_summary", "agent_count": len(agents)}, ensure_ascii=False),
    }


def upsert_prediction(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO videos (video_key, site, video_id, dataset, title, file_path)
            VALUES (:video_key, :site, :video_id, :dataset, :video_key, :file_path)
            ON CONFLICT(video_key) DO UPDATE SET
                site=excluded.site,
                video_id=excluded.video_id,
                dataset=excluded.dataset,
                file_path=excluded.file_path
            """,
            row,
        )
        conn.execute(
            """
            INSERT INTO model_predictions (
                video_key, source_output_root, summary_json, risk, level1_scene, level2_subject,
                level3_risk_type, description, risk_localization, solution_for_person,
                solution_for_hazard_source, solution_prevent_recurrence, yes_count, no_count,
                votes_cast, complete, majority_json, agents_json, updated_at
            ) VALUES (
                :video_key, :source_output_root, :summary_json, :risk, :level1_scene, :level2_subject,
                :level3_risk_type, :description, :risk_localization, :solution_for_person,
                :solution_for_hazard_source, :solution_prevent_recurrence, :yes_count, :no_count,
                :votes_cast, :complete, :majority_json, :agents_json, datetime('now')
            )
            ON CONFLICT(video_key) DO UPDATE SET
                source_output_root=excluded.source_output_root,
                summary_json=excluded.summary_json,
                risk=excluded.risk,
                level1_scene=excluded.level1_scene,
                level2_subject=excluded.level2_subject,
                level3_risk_type=excluded.level3_risk_type,
                description=excluded.description,
                risk_localization=excluded.risk_localization,
                solution_for_person=excluded.solution_for_person,
                solution_for_hazard_source=excluded.solution_for_hazard_source,
                solution_prevent_recurrence=excluded.solution_prevent_recurrence,
                yes_count=excluded.yes_count,
                no_count=excluded.no_count,
                votes_cast=excluded.votes_cast,
                complete=excluded.complete,
                majority_json=excluded.majority_json,
                agents_json=excluded.agents_json,
                updated_at=datetime('now')
            """,
            row,
        )


def import_strict_schema(conn: sqlite3.Connection, strict_path: Path) -> tuple[int, int]:
    imported = 0
    skipped = 0
    with strict_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = prediction_from_strict_row(json.loads(line), strict_path)
            if row is None:
                skipped += 1
                continue
            upsert_prediction(conn, row)
            imported += 1
    return imported, skipped


def import_predictions(batch_outputs: Path, db_path: Path) -> dict[str, int]:
    init_db()
    conn = sqlite3.connect(db_path)

    strict_files = sorted(batch_outputs.glob("*/strict_english_schema.jsonl"))
    strict_imported = 0
    strict_skipped = 0
    for strict_path in strict_files:
        imported, skipped = import_strict_schema(conn, strict_path)
        strict_imported += imported
        strict_skipped += skipped

    imported_keys = {
        row[0]
        for row in conn.execute(
            "SELECT video_key FROM model_predictions WHERE agents_json LIKE '%qwen_strict_schema%'"
        ).fetchall()
    }

    repaired = {p.parent: p for p in sorted(batch_outputs.glob("*/reviews/**/review_summary.repaired.json"))}
    raw = {p.parent: p for p in sorted(batch_outputs.glob("*/reviews/**/review_summary.json"))}
    summaries = [repaired.get(parent) or raw[parent] for parent in sorted(set(repaired) | set(raw))]

    summary_imported = 0
    summary_skipped = 0
    for summary_path in summaries:
        row = prediction_from_summary(summary_path)
        if row is None:
            summary_skipped += 1
            continue
        if row["video_key"] in imported_keys:
            continue
        upsert_prediction(conn, row)
        summary_imported += 1

    conn.commit()
    conn.close()
    return {
        "strict_files": len(strict_files),
        "strict_imported": strict_imported,
        "strict_skipped": strict_skipped,
        "summary_found": len(summaries),
        "summary_imported_fallback": summary_imported,
        "summary_skipped_incomplete": summary_skipped,
        "total_imported_or_updated": strict_imported + summary_imported,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-outputs", type=Path, default=PROJECT_ROOT / "public_data_filter" / "batch_outputs")
    parser.add_argument("--db", type=Path, default=APP_DIR / "data" / "app.db")
    args = parser.parse_args()
    stats = import_predictions(args.batch_outputs.resolve(), args.db.resolve())
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

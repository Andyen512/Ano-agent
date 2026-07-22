from __future__ import annotations

import json
from pathlib import Path


def snapshot_index_files(model_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in model_dir.glob("*.index.json")
        if path.is_file()
    )


def referenced_snapshot_files(index_path: Path) -> set[str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    return {str(filename) for filename in weight_map.values() if filename}


def missing_snapshot_files(model_dir: Path) -> list[str]:
    if not model_dir.exists():
        return []

    missing: set[str] = set()
    for index_path in snapshot_index_files(model_dir):
        for relative_name in referenced_snapshot_files(index_path):
            if not (model_dir / relative_name).exists():
                missing.add(relative_name)
    return sorted(missing)


def invalid_snapshot_files(model_dir: Path) -> list[str]:
    try:
        import torch  # noqa: F401
        from safetensors import safe_open
    except Exception:
        return []

    invalid: set[str] = set()
    referenced: set[str] = set()
    for index_path in snapshot_index_files(model_dir):
        referenced.update(name for name in referenced_snapshot_files(index_path) if name.endswith(".safetensors"))

    for relative_name in sorted(referenced):
        path = model_dir / relative_name
        if not path.exists():
            continue
        try:
            with safe_open(path, framework="pt", device="cpu"):
                pass
        except Exception:
            invalid.add(relative_name)
    return sorted(invalid)


def has_incomplete_downloads(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    for path in model_dir.rglob("*.incomplete"):
        # Hugging Face local-dir downloads may leave stale cache fragments under
        # `.cache/` even after the final shard has been materialized in the
        # snapshot root. Those should not block inference once the root snapshot
        # is complete.
        if ".cache" in path.parts:
            continue
        return True
    return False


def is_model_snapshot_complete(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if has_incomplete_downloads(model_dir):
        return False

    index_files = snapshot_index_files(model_dir)
    if index_files:
        return not missing_snapshot_files(model_dir) and not invalid_snapshot_files(model_dir)

    return (model_dir / ".download_complete").exists()

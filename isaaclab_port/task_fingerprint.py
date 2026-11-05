"""Portable, strict compatibility record for a simulator-task checkpoint.

This module deliberately imports no simulator packages. A checkpoint is only
resumable when its adjacent ``task_fingerprint.json`` matches the current task,
physics packages and implementation. Absolute installation paths are excluded.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA_VERSION = 1
_PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_FILES = {
    "task": _PACKAGE_ROOT / "env.py",
    "evaluator": _PACKAGE_ROOT / "runtime.py",
    "algorithm": _PACKAGE_ROOT / "core.py",
    "robot_usd": _PACKAGE_ROOT / "assets" / "planar_biped.usda",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def build_task_fingerprint(
    *, episode_steps: int, solver: str, substeps: int,
    simulation_versions: Mapping[str, Any],
    task_config: Mapping[str, Any] | None = None,
    source_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Hash exact task semantics, source/assets, and simulation dependencies.

    ``task_config`` should contain effective configuration, including device
    family when cross-device score comparability is required. Episode length and
    Newton's solver/substeps are always included. ``source_files`` is primarily
    for isolated tests or a deliberately different robot asset; normal callers
    use the shipped native USD plus task, evaluator and algorithm source.
    """
    if isinstance(episode_steps, bool) or not isinstance(episode_steps, int) or episode_steps < 1:
        raise ValueError("episode_steps must be a positive integer")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if not isinstance(solver, str) or not solver.strip():
        raise ValueError("solver must be a nonempty name")
    sources = _SOURCE_FILES if source_files is None else source_files
    if not sources:
        raise ValueError("At least one source or asset is required")
    payload = {
        "task_name": "poet_planar_biped_isaaclab3",
        "episode_steps": episode_steps,
        "solver": solver,
        "substeps": substeps,
        "simulation_versions": dict(simulation_versions),
        "task_config": dict(task_config or {}),
        "source_sha256": {
            str(label): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for label, path in sources.items()
        },
    }
    # JSON normalization also rejects unserializable/nonfinite configuration.
    payload = json.loads(_canonical(payload))
    return {"schema_version": SCHEMA_VERSION, "sha256": _digest(payload), "payload": payload}


def _check_record(record: Any, description: str) -> None:
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported or missing task fingerprint schema: {description}")
    payload = record.get("payload")
    if not isinstance(payload, dict) or record.get("sha256") != _digest(payload):
        raise ValueError(f"Invalid task fingerprint checksum: {description}")


def write_task_fingerprint(path: str | Path, fingerprint: Mapping[str, Any]) -> None:
    """Atomically write the compatibility sidecar next to a checkpoint."""
    record = dict(fingerprint)
    _check_record(record, "current task")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".task_fingerprint_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _differences(saved: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        changes = []
        for key in sorted(saved.keys() | current.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in saved or key not in current:
                changes.append(name)
            else:
                changes.extend(_differences(saved[key], current[key], name))
        return changes
    return [] if saved == current else [prefix]


def validate_task_fingerprint(path: str | Path, fingerprint: Mapping[str, Any]) -> None:
    """Reject missing, corrupted, or incompatible checkpoint task metadata."""
    source = Path(path)
    _check_record(dict(fingerprint), "current task")
    try:
        saved = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Checkpoint task metadata is missing: {source}; compatibility cannot be verified") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint task metadata: {source}") from exc
    _check_record(saved, str(source))
    if saved["sha256"] != fingerprint["sha256"]:
        changes = _differences(saved["payload"], fingerprint["payload"])
        raise ValueError("Checkpoint task is incompatible; changed: " + ", ".join(changes))

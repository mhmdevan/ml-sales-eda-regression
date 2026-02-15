from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


def _model_dir(registry_dir: Path, model_name: str) -> Path:
    return registry_dir / model_name


def _versions_dir(registry_dir: Path, model_name: str) -> Path:
    return _model_dir(registry_dir, model_name) / "versions"


def _aliases_path(registry_dir: Path, model_name: str) -> Path:
    return _model_dir(registry_dir, model_name) / "aliases.json"


def _read_aliases(registry_dir: Path, model_name: str) -> dict[str, str | None]:
    path = _aliases_path(registry_dir, model_name)
    if not path.exists():
        return {"staging": None, "prod": None, "previous_prod": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "staging": payload.get("staging"),
        "prod": payload.get("prod"),
        "previous_prod": payload.get("previous_prod"),
    }


def _write_aliases(registry_dir: Path, model_name: str, aliases: dict[str, str | None]) -> None:
    model_root = _model_dir(registry_dir, model_name)
    model_root.mkdir(parents=True, exist_ok=True)
    _aliases_path(registry_dir, model_name).write_text(
        json.dumps(aliases, indent=2),
        encoding="utf-8",
    )


def _next_version(registry_dir: Path, model_name: str) -> str:
    versions_root = _versions_dir(registry_dir, model_name)
    versions_root.mkdir(parents=True, exist_ok=True)

    version_numbers: list[int] = []
    for child in versions_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("v"):
            continue
        try:
            version_numbers.append(int(name[1:]))
        except ValueError:
            continue

    next_number = 1 if not version_numbers else max(version_numbers) + 1
    return f"v{next_number}"


def register_model_version(
    *,
    source_dir: Path,
    registry_dir: Path,
    model_name: str,
    artifact_filenames: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source artifact directory does not exist: {source_dir}")

    version = _next_version(registry_dir, model_name)
    destination = _versions_dir(registry_dir, model_name) / version
    destination.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for filename in artifact_filenames:
        source_path = source_dir / filename
        if not source_path.exists():
            continue
        shutil.copy2(source_path, destination / filename)
        copied_files.append(filename)

    if not copied_files:
        raise FileNotFoundError("No registry artifacts were copied; verify artifact filenames.")

    manifest = {
        "model_name": model_name,
        "version": version,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "files": copied_files,
        "metadata": metadata or {},
    }

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    aliases = _read_aliases(registry_dir, model_name)
    if aliases["staging"] is None:
        aliases["staging"] = version
        _write_aliases(registry_dir, model_name, aliases)

    return {
        "model_name": model_name,
        "version": version,
        "path": str(destination),
        "files": copied_files,
        "aliases": aliases,
    }


def set_alias(
    *,
    registry_dir: Path,
    model_name: str,
    alias: str,
    version: str,
) -> dict[str, str | None]:
    if alias not in {"staging", "prod"}:
        raise ValueError("alias must be one of: staging, prod")

    version_path = _versions_dir(registry_dir, model_name) / version
    if not version_path.exists():
        raise FileNotFoundError(f"Version does not exist: {version_path}")

    aliases = _read_aliases(registry_dir, model_name)
    aliases[alias] = version
    _write_aliases(registry_dir, model_name, aliases)
    return aliases


def promote_to_prod(
    *,
    registry_dir: Path,
    model_name: str,
    version: str,
) -> dict[str, str | None]:
    aliases = _read_aliases(registry_dir, model_name)
    current_prod = aliases.get("prod")

    version_path = _versions_dir(registry_dir, model_name) / version
    if not version_path.exists():
        raise FileNotFoundError(f"Version does not exist: {version_path}")

    aliases["previous_prod"] = current_prod
    aliases["prod"] = version
    _write_aliases(registry_dir, model_name, aliases)
    return aliases


def rollback_prod(*, registry_dir: Path, model_name: str) -> dict[str, str | None]:
    aliases = _read_aliases(registry_dir, model_name)
    target = aliases.get("previous_prod")
    if target is None:
        raise RuntimeError("No previous production version exists for rollback.")

    current_prod = aliases.get("prod")
    aliases["prod"] = target
    aliases["previous_prod"] = current_prod
    _write_aliases(registry_dir, model_name, aliases)
    return aliases


def resolve_alias_artifact_paths(
    *,
    registry_dir: Path,
    model_name: str,
    model_filename: str,
    metadata_filename: str,
    primary_alias: str = "prod",
    shadow_alias: str | None = "staging",
) -> dict[str, Path | None]:
    aliases = _read_aliases(registry_dir, model_name)

    primary_version = aliases.get(primary_alias)
    shadow_version = aliases.get(shadow_alias) if shadow_alias else None

    primary_model_path: Path | None = None
    primary_metadata_path: Path | None = None
    shadow_model_path: Path | None = None
    shadow_metadata_path: Path | None = None

    if primary_version:
        primary_root = _versions_dir(registry_dir, model_name) / primary_version
        primary_model_path = primary_root / model_filename
        primary_metadata_path = primary_root / metadata_filename

    if shadow_version and shadow_version != primary_version:
        shadow_root = _versions_dir(registry_dir, model_name) / shadow_version
        shadow_model_path = shadow_root / model_filename
        shadow_metadata_path = shadow_root / metadata_filename

    return {
        "primary_model_path": primary_model_path,
        "primary_metadata_path": primary_metadata_path,
        "shadow_model_path": shadow_model_path,
        "shadow_metadata_path": shadow_metadata_path,
    }


def read_aliases(*, registry_dir: Path, model_name: str) -> dict[str, str | None]:
    return _read_aliases(registry_dir, model_name)

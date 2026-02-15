from pathlib import Path

from ml_core.registry.versioning import (
    promote_to_prod,
    read_aliases,
    register_model_version,
    resolve_alias_artifact_paths,
    rollback_prod,
    set_alias,
)


def _create_artifacts(directory: Path, model_name: str, metadata_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / model_name).write_text("model-bytes", encoding="utf-8")
    (directory / metadata_name).write_text('{"best_model_name": "m"}', encoding="utf-8")


def test_registry_versioning_and_rollback(tmp_path: Path) -> None:
    source_dir = tmp_path / "artifacts"
    registry_dir = tmp_path / "registry"

    _create_artifacts(source_dir, "model.joblib", "metadata.json")

    v1 = register_model_version(
        source_dir=source_dir,
        registry_dir=registry_dir,
        model_name="sales",
        artifact_filenames=["model.joblib", "metadata.json"],
    )

    aliases = read_aliases(registry_dir=registry_dir, model_name="sales")
    assert aliases["staging"] == v1["version"]

    set_alias(registry_dir=registry_dir, model_name="sales", alias="prod", version=v1["version"])

    _create_artifacts(source_dir, "model.joblib", "metadata.json")
    v2 = register_model_version(
        source_dir=source_dir,
        registry_dir=registry_dir,
        model_name="sales",
        artifact_filenames=["model.joblib", "metadata.json"],
    )

    promote_to_prod(registry_dir=registry_dir, model_name="sales", version=v2["version"])
    aliases_after_promote = read_aliases(registry_dir=registry_dir, model_name="sales")
    assert aliases_after_promote["prod"] == v2["version"]
    assert aliases_after_promote["previous_prod"] == v1["version"]

    aliases_after_rollback = rollback_prod(registry_dir=registry_dir, model_name="sales")
    assert aliases_after_rollback["prod"] == v1["version"]

    paths = resolve_alias_artifact_paths(
        registry_dir=registry_dir,
        model_name="sales",
        model_filename="model.joblib",
        metadata_filename="metadata.json",
    )
    assert paths["primary_model_path"] is not None
    assert paths["primary_model_path"].exists()

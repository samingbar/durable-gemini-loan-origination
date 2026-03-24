# ruff: noqa: SLF001
"""Tests for the insurance fixed-flow reset cache CLI helper."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from src.workflows.insurance_claims_fixed_flow.utils import reset_cache


def test_resolve_upload_root_is_repo_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve relative upload roots from the repository root."""
    monkeypatch.setattr(reset_cache, "_repo_root", lambda: tmp_path)

    resolved = reset_cache._resolve_upload_root("datasets/uploads/insurance_claims")

    assert resolved == tmp_path / "datasets/uploads/insurance_claims"


def test_reset_cache_removes_manifest_without_purging_uploads(tmp_path: Path) -> None:
    """Clear only the manifest unless upload purging is requested."""
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    manifest_path = upload_root / reset_cache.MANIFEST_NAME
    manifest_path.write_text("[]")
    case_dir = upload_root / "CLM-20260324-ABC12345"
    case_dir.mkdir()

    removed = reset_cache._reset_cache(upload_root, purge_uploads=False)

    assert removed == [manifest_path]
    assert not manifest_path.exists()
    assert case_dir.exists()


def test_reset_cache_purges_uploaded_case_content(tmp_path: Path) -> None:
    """Delete uploaded case folders and extra files when purge is enabled."""
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    manifest_path = upload_root / reset_cache.MANIFEST_NAME
    manifest_path.write_text("[]")
    case_dir = upload_root / "CLM-20260324-ABC12345"
    case_dir.mkdir()
    extra_file = upload_root / "orphan.txt"
    extra_file.write_text("data")

    removed = reset_cache._reset_cache(upload_root, purge_uploads=True)

    assert manifest_path in removed
    assert case_dir in removed
    assert extra_file in removed
    assert not manifest_path.exists()
    assert not case_dir.exists()
    assert not extra_file.exists()


def test_main_prefers_insurance_upload_root_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use INSURANCE_UPLOAD_ROOT instead of the generic UPLOAD_ROOT env var."""
    monkeypatch.setattr(reset_cache, "_repo_root", lambda: tmp_path)

    wrong_root = tmp_path / "wrong"
    wrong_root.mkdir()
    wrong_manifest = wrong_root / reset_cache.MANIFEST_NAME
    wrong_manifest.write_text("[]")

    insurance_root = tmp_path / "insurance-cache"
    insurance_root.mkdir()
    insurance_manifest = insurance_root / reset_cache.MANIFEST_NAME
    insurance_manifest.write_text("[]")

    monkeypatch.setenv("UPLOAD_ROOT", str(wrong_root))
    monkeypatch.setenv(reset_cache.UPLOAD_ROOT_ENV_VAR, "insurance-cache")
    monkeypatch.setattr(sys, "argv", ["reset_cache.py"])

    reset_cache.main()
    output = capsys.readouterr().out

    assert "Cleared:" in output
    assert str(insurance_manifest) in output
    assert not insurance_manifest.exists()
    assert wrong_manifest.exists()

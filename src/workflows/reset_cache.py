"""CLI helper to reset the human review cache history."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

DEFAULT_UPLOAD_ROOT = "datasets/uploads"
MANIFEST_NAME = "cases.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_upload_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def _reset_cache(upload_root: Path, purge_uploads: bool) -> list[Path]:
    removed: list[Path] = []

    manifest_path = upload_root / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()
        removed.append(manifest_path)

    if purge_uploads and upload_root.exists():
        for child in upload_root.iterdir():
            if child.name == MANIFEST_NAME:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed.append(child)

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset the human review cache history (cases.json).",
    )
    parser.add_argument(
        "--upload-root",
        default=os.environ.get("UPLOAD_ROOT", DEFAULT_UPLOAD_ROOT),
        help="Upload directory used by the review UI (default: datasets/uploads).",
    )
    parser.add_argument(
        "--purge-uploads",
        action="store_true",
        help="Also delete uploaded case folders inside the upload root.",
    )
    args = parser.parse_args()

    upload_root = _resolve_upload_root(args.upload_root)
    if not upload_root.exists():
        print(f"Upload root does not exist: {upload_root}")
        return

    removed = _reset_cache(upload_root, args.purge_uploads)
    if not removed:
        print("Nothing to clear.")
        return

    print("Cleared:")
    for path in removed:
        print(f"- {path}")


if __name__ == "__main__":
    main()

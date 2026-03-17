# ruff: noqa: T201
"""Run the insurance claims workflow against OCR-extracted images."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from .insurance_models import ClaimAdjudicationInput
from .insurance_workflow import InsuranceClaimAdjudicationWorkflow

TASK_QUEUE = os.environ.get("INSURANCE_TASK_QUEUE", "insurance-claims")
TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
DEFAULT_IMAGE_DIR = "datasets/insurance_claims/images"


def _load_dotenv() -> None:
    """Load .env values into the process environment without overriding exports."""
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'").strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _find_case_ids(image_dir: Path) -> list[str]:
    """Find case ids from case-scoped image filenames."""
    if not image_dir.exists():
        return []

    case_ids: set[str] = set()
    pattern = re.compile(r"^(?P<case_id>.+)_p\d+\.(png|jpg|jpeg)$", re.IGNORECASE)
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            case_ids.add(match.group("case_id"))
    return sorted(case_ids)


def _has_any_images(image_dir: Path) -> bool:
    """Return whether the directory contains at least one image."""
    return any(any(image_dir.glob(pattern)) for pattern in ("*.png", "*.jpg", "*.jpeg"))


async def _run_case(client: Client, case_id: str, image_dir: str) -> None:
    """Execute the workflow for a single case and print a summary."""
    input_data = ClaimAdjudicationInput(case_id=case_id, image_dir=image_dir)
    result = await client.execute_workflow(
        InsuranceClaimAdjudicationWorkflow.run,
        input_data,
        id=f"insurance-{case_id}",
        task_queue=TASK_QUEUE,
    )

    print("=" * 80)
    print(f"Case {case_id}")
    print(f"Workflow ID: insurance-{case_id}")
    print(f"Final decision: {result.final_decision}")
    print(f"Risk score: {result.risk_score}")
    print(f"Human review required: {result.human_review_required}")
    if result.human_review_required:
        print("Review needed. Start the review UI and submit a decision.")
    print("Risk flags:")
    for flag in result.risk_flags:
        print(f"- {flag}")
    print("Bias flags:")
    for flag in result.bias_flags:
        print(f"- {flag}")
    print("Decision memo preview:")
    print(result.decision_memo[:600])
    print()


async def main() -> None:
    """Run all discovered insurance claim cases from an image directory."""
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Run OCR-first insurance claims demo.")
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing OCR input images (default: {DEFAULT_IMAGE_DIR}).",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    case_ids = _find_case_ids(image_dir)
    if not case_ids:
        if not _has_any_images(image_dir):
            print(f"No case images found in {image_dir}.")
            return
        case_ids = ["DEMO-UNSCOPED"]
        print("No case-scoped filenames found. Processing all images as DEMO-UNSCOPED.")

    client = await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )
    for case_id in case_ids:
        await _run_case(client, case_id, str(image_dir))


if __name__ == "__main__":
    asyncio.run(main())

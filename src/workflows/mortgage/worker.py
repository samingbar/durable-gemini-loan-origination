"""Worker for the mortgage underwriting workflow."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from .mortgage_activities import (
    extract_application_from_images,
    retrieve_policy_context,
    run_agent_analysis,
    run_supervisor,
    run_critic_review,
    run_decision_memo,
)
from .mortgage_workflow import MortgageUnderwritingWorkflow

TASK_QUEUE = os.environ.get("MORTGAGE_TASK_QUEUE", "mortgage-underwriting")


def _load_dotenv() -> None:
    """Load .env values into the process environment (non-destructive)."""

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


async def main() -> None:
    """Run the Temporal worker."""

    _load_dotenv()
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MortgageUnderwritingWorkflow],
        activities=[
            extract_application_from_images,
            retrieve_policy_context,
            run_agent_analysis,
            run_supervisor,
            run_critic_review,
            run_decision_memo,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

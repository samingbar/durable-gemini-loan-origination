"""Run the mortgage underwriting workflow against the sample test cases."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from .mortgage_activities import (
    retrieve_policy_context,
    run_agent_analysis,
    run_supervisor,
    run_critic_review,
    run_decision_memo,
)
from .mortgage_models import MortgageApplication, UnderwritingInput
from .mortgage_workflow import MortgageUnderwritingWorkflow

TASK_QUEUE = "mortgage-underwriting"
TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


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


def _load_test_cases() -> list[dict]:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "notebook-inspiration" / "mortgage_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"]


async def _run_case(client, case: dict) -> None:
    applicant = MortgageApplication(**case)
    input_data = UnderwritingInput(case_id=case["case_id"], applicant=applicant)

    result = await client.execute_workflow(
        MortgageUnderwritingWorkflow.run,
        input_data,
        id=f"mortgage-{case['case_id']}",
        task_queue=TASK_QUEUE,
    )

    print("=" * 80)
    print(f"Case {case['case_id']} - Expected {case['expected_decision']}")
    print(f"Workflow ID: mortgage-{case['case_id']}")
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
    _load_dotenv()
    client = await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )
    for case in _load_test_cases():
        await _run_case(client, case)


if __name__ == "__main__":
    asyncio.run(main())

"""Workflow-level tests with mocked activities for insurance claims."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.worker import Worker

from src.platform.lakebase.repository import LakebaseRepository
from src.workflows.insurance_claims_fixed_flow.insurance_models import (
    AgentResult,
    AgentTask,
    ClaimAdjudicationInput,
    ClaimOcrTask,
    CriticResult,
    CriticTask,
    DecisionRecommendation,
    DecisionResult,
    DecisionTask,
    HumanReviewInput,
    InsuranceClaim,
)
from src.workflows.insurance_claims_fixed_flow.insurance_operational_activities import (
    publish_claim_update,
    record_analysis_result,
    record_case_state,
    upsert_review_task,
)
from src.workflows.insurance_claims_fixed_flow.insurance_workflow import (
    InsuranceClaimAdjudicationWorkflow,
)

if TYPE_CHECKING:
    from temporalio.client import Client

TASK_QUEUE = "test-insurance-claims-queue"


def _load_case(index: int) -> dict:
    repo_root = Path(__file__).resolve().parents[4]
    data_path = repo_root / "resources" / "insurance_claim_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"][index]


@activity.defn(name="retrieve_policy_context")
async def fake_retrieve_policy_context(_query: str) -> str:
    """Return a fixed policy context for workflow tests."""
    return "policy context"


@activity.defn(name="extract_claim_from_images")
async def fake_extract_claim_from_images(task: ClaimOcrTask) -> InsuranceClaim:
    """Return a deterministic insurance claim fixture."""
    case = _load_case(1)
    case["case_id"] = task.case_id
    return InsuranceClaim(**case)


@activity.defn(name="run_agent_analysis")
async def fake_run_agent_analysis(task: AgentTask) -> AgentResult:
    """Return a simple specialist analysis string."""
    return AgentResult(analysis=f"{task.agent_name} analysis complete")


@activity.defn(name="run_critic_review")
async def fake_run_critic_review(_task: CriticTask) -> CriticResult:
    """Return a fixed critic review."""
    return CriticResult(review="critic ok")


@activity.defn(name="run_decision_memo")
async def fake_run_decision_memo(_task: DecisionTask) -> DecisionResult:
    """Force the workflow into its human review path."""
    recommendation = DecisionRecommendation(
        decision="CONDITIONAL",
        risk_score=57,
        memo="needs human review",
        conditions=[],
        human_review_reason="test review",
    )
    return DecisionResult(recommendation=recommendation, raw_response="{}")


@pytest.mark.asyncio
async def test_workflow_human_review_signal(
    client: Client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait for the review packet, signal a reviewer decision, and complete."""
    db_path = tmp_path / "lakebase.sqlite3"
    outbox_path = tmp_path / "claim_updates.jsonl"
    monkeypatch.setenv("LAKEBASE_DB_PATH", str(db_path))
    monkeypatch.setenv("CLAIMS_SYSTEM_OUTBOX_PATH", str(outbox_path))

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[InsuranceClaimAdjudicationWorkflow],
        activities=[
            fake_extract_claim_from_images,
            fake_retrieve_policy_context,
            fake_run_agent_analysis,
            fake_run_critic_review,
            fake_run_decision_memo,
            record_case_state,
            record_analysis_result,
            upsert_review_task,
            publish_claim_update,
        ],
    ):
        case = _load_case(1)
        image_dir = tmp_path / "fake-images"
        image_dir.mkdir()
        workflow_input = ClaimAdjudicationInput(
            case_id=case["case_id"],
            image_dir=str(image_dir),
        )

        handle = await client.start_workflow(
            InsuranceClaimAdjudicationWorkflow.run,
            workflow_input,
            id=f"test-{case['case_id']}-{uuid.uuid4()}",
            task_queue=TASK_QUEUE,
        )

        packet = None
        for _ in range(10):
            packet = await handle.query(InsuranceClaimAdjudicationWorkflow.get_review_packet)
            if packet is not None:
                break
            await asyncio.sleep(0.1)

        assert packet is not None
        assert packet.decision_recommendation.decision == "CONDITIONAL"
        assert packet.display_name

        await handle.signal(
            InsuranceClaimAdjudicationWorkflow.submit_human_review,
            HumanReviewInput(
                reviewer="QA Reviewer",
                decision="APPROVED",
                notes="Approved after senior examiner review.",
            ),
        )

        result = await handle.result()
        assert result.final_decision == "APPROVED"
        assert result.human_review is not None
        assert result.downstream_action is not None
        assert result.downstream_action.status == "PUBLISHED"
        assert result.analyses.coverage
        assert result.analyses.liability
        assert result.analyses.damages
        assert result.analyses.fraud
        assert "[CLAIMANT_NAME]" not in result.decision_memo

    repository = LakebaseRepository(db_path)
    case_record = repository.fetch_case(case["case_id"])
    assert case_record is not None
    assert case_record.status == "COMPLETED"
    assert case_record.current_decision == "APPROVED"

    workflow_run = repository.fetch_workflow_run(handle.id)
    assert workflow_run is not None
    assert workflow_run.stage == "COMPLETED"

    review_task = repository.fetch_review_task(f"{case['case_id']}:human-review")
    assert review_task is not None
    assert review_task.status == "COMPLETED"
    assert review_task.decision == "APPROVED"

    downstream_action = repository.fetch_downstream_action(f"{handle.id}:claim-decision")
    assert downstream_action is not None
    assert downstream_action.status == "PUBLISHED"
    assert downstream_action.target_system == "claims-system-demo"
    assert outbox_path.exists()

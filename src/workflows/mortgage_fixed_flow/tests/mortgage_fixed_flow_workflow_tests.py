"""Workflow-level tests for the mortgage fixed flow with mocked activities."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from temporalio import activity
from temporalio.worker import Worker

from src.workflows.mortgage_fixed_flow.mortgage_models import (
    AgentResult,
    AgentTask,
    ApplicationOcrTask,
    CriticResult,
    CriticTask,
    DecisionRecommendation,
    DecisionResult,
    DecisionTask,
    HumanReviewInput,
    MortgageApplication,
    UnderwritingInput,
)
from src.workflows.mortgage_fixed_flow.mortgage_workflow import MortgageUnderwritingWorkflow

if TYPE_CHECKING:
    from temporalio.client import Client

TASK_QUEUE = "test-mortgage-fixed-flow"
REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = REPO_ROOT / "resources" / "mortgage_test_cases.json"


def _load_case(index: int) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text())
    return payload["test_cases"][index]


@activity.defn(name="retrieve_policy_context")
async def fake_retrieve_policy_context(query: str) -> str:
    """Return deterministic policy text for the workflow test."""
    return f"policy context for {query}"


@activity.defn(name="extract_application_from_images")
async def fake_extract_application_from_images(task: ApplicationOcrTask) -> MortgageApplication:
    """Return a standard application fixture for OCR extraction."""
    case = _load_case(1)
    case["case_id"] = task.case_id
    return MortgageApplication(**case)


@activity.defn(name="run_agent_analysis")
async def fake_run_agent_analysis(task: AgentTask) -> AgentResult:
    """Return a deterministic analysis for each specialist agent."""
    return AgentResult(analysis=f"{task.agent_name} analysis complete")


@activity.defn(name="run_critic_review")
async def fake_run_critic_review(task: CriticTask) -> CriticResult:
    """Return a deterministic critic response."""
    return CriticResult(review=f"critic ok for {task.applicant.case_id}")


@activity.defn(name="run_decision_memo")
async def fake_run_decision_memo(_task: DecisionTask) -> DecisionResult:
    """Force the workflow into the human-review path."""
    recommendation = DecisionRecommendation(
        decision="CONDITIONAL",
        risk_score=55,
        memo="needs human review for [APPLICANT_NAME]",
        conditions=[],
        human_review_reason="test review",
    )
    return DecisionResult(recommendation=recommendation, raw_response="{}")


@activity.defn(name="extract_application_from_images")
async def fake_extract_weak_application(task: ApplicationOcrTask) -> MortgageApplication:
    """Return a weak fixture so policy violations are guaranteed."""
    case = _load_case(2)
    case["case_id"] = task.case_id
    return MortgageApplication(**case)


@activity.defn(name="run_decision_memo")
async def fake_run_approved_memo(_task: DecisionTask) -> DecisionResult:
    """Return an unsafe approval so the workflow must override it."""
    recommendation = DecisionRecommendation(
        decision="APPROVED",
        risk_score=72,
        memo="approve [APPLICANT_NAME]",
        conditions=[],
    )
    return DecisionResult(recommendation=recommendation, raw_response="{}")


@pytest.mark.asyncio
async def test_workflow_human_review_signal(client: Client) -> None:
    """Pause on conditional decisions and continue after a human signal."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MortgageUnderwritingWorkflow],
        activities=[
            fake_extract_application_from_images,
            fake_retrieve_policy_context,
            fake_run_agent_analysis,
            fake_run_critic_review,
            fake_run_decision_memo,
        ],
    ):
        case = _load_case(1)
        workflow_input = UnderwritingInput(
            case_id=case["case_id"],
            image_dir="fixtures/fake-images",
        )

        handle = await client.start_workflow(
            MortgageUnderwritingWorkflow.run,
            workflow_input,
            id=f"test-{case['case_id']}",
            task_queue=TASK_QUEUE,
        )

        packet = None
        for _ in range(10):
            packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
            if packet is not None:
                break
            await asyncio.sleep(0.1)

        assert packet is not None
        assert packet.decision_recommendation.decision == "CONDITIONAL"
        assert packet.display_name

        await handle.signal(
            MortgageUnderwritingWorkflow.submit_human_review,
            HumanReviewInput(
                reviewer="QA Reviewer",
                decision="APPROVED",
                notes="Approved after review.",
            ),
        )

        result = await handle.result()
        assert result.final_decision == "APPROVED"
        assert result.human_review is not None
        assert result.analyses.credit == "Credit analysis complete"
        assert result.analyses.income == "Income analysis complete"
        assert result.analyses.assets == "Assets analysis complete"
        assert result.analyses.collateral == "Collateral analysis complete"
        assert "[APPLICANT_NAME]" not in result.decision_memo


@pytest.mark.asyncio
async def test_workflow_policy_violations_force_human_review(client: Client) -> None:
    """Convert unsafe LLM approvals into conditional human-review cases."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MortgageUnderwritingWorkflow],
        activities=[
            fake_extract_weak_application,
            fake_retrieve_policy_context,
            fake_run_agent_analysis,
            fake_run_critic_review,
            fake_run_approved_memo,
        ],
    ):
        case = _load_case(2)
        workflow_input = UnderwritingInput(
            case_id=case["case_id"],
            image_dir="fixtures/fake-images",
        )

        handle = await client.start_workflow(
            MortgageUnderwritingWorkflow.run,
            workflow_input,
            id=f"hard-stop-{case['case_id']}",
            task_queue=TASK_QUEUE,
        )

        packet = None
        for _ in range(10):
            packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
            if packet is not None:
                break
            await asyncio.sleep(0.1)

        assert packet is not None
        assert packet.decision_recommendation.decision == "CONDITIONAL"
        assert packet.decision_recommendation.human_review_reason == (
            "Policy hard-stop violations require review."
        )
        assert packet.policy_violations

        await handle.signal(
            MortgageUnderwritingWorkflow.submit_human_review,
            HumanReviewInput(
                reviewer="QA Reviewer",
                decision="REJECTED",
                notes="Rejected because hard-stop violations were confirmed.",
            ),
        )

        result = await handle.result()
        assert result.final_decision == "REJECTED"
        assert result.human_review_required is True
        assert result.policy_violations

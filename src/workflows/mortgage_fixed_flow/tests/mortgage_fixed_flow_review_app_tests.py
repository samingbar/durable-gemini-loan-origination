# ruff: noqa: D102, D107, PLR2004
"""FastAPI tests for the mortgage fixed-flow review app."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from src.workflows.mortgage_fixed_flow import review_app
from src.workflows.mortgage_fixed_flow.mortgage_models import (
    DecisionRecommendation,
    HumanReviewInput,
    HumanReviewPacket,
    MortgageApplication,
    UnderwritingAnalyses,
    UnderwritingOutput,
)
from src.workflows.mortgage_fixed_flow.mortgage_utils import compute_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = REPO_ROOT / "resources" / "mortgage_test_cases.json"


class FakeExecutionStatus:
    """Match the subset of Temporal status shape used by the UI."""

    def __init__(self, name: str) -> None:
        self.name = f"WORKFLOW_EXECUTION_STATUS_{name}"


class FakeDescription:
    """Small workflow description shim for test handles."""

    def __init__(self, state: FakeWorkflowState) -> None:
        self.status = FakeExecutionStatus(state.status)
        self.start_time = state.start_time


class FakeWorkflowInfo:
    """Small workflow listing record for test clients."""

    def __init__(self, workflow_id: str, start_time: datetime) -> None:
        self.id = workflow_id
        self.start_time = start_time


@dataclass
class FakeWorkflowState:
    """In-memory workflow state for the fake Temporal client."""

    status: str = "RUNNING"
    packet: HumanReviewPacket | None = None
    result: UnderwritingOutput | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    signals: list[HumanReviewInput] = field(default_factory=list)


class FakeHandle:
    """Fake workflow handle backed by a FakeWorkflowState."""

    def __init__(self, state: FakeWorkflowState) -> None:
        self.state = state

    async def describe(self) -> FakeDescription:
        return FakeDescription(self.state)

    async def query(self, _query: object) -> HumanReviewPacket | None:
        return self.state.packet

    async def result(self) -> UnderwritingOutput:
        assert self.state.result is not None
        return self.state.result

    async def signal(self, _signal: object, payload: HumanReviewInput) -> None:
        self.state.signals.append(payload)


class FakeClient:
    """Small fake Temporal client for the review-app routes."""

    def __init__(self, states: dict[str, FakeWorkflowState] | None = None) -> None:
        self.states = states or {}
        self.start_calls: list[dict[str, object]] = []

    def close(self) -> None:
        """Match the real client's close method."""

    def get_workflow_handle(self, workflow_id: str, result_type: type | None = None) -> FakeHandle:
        del result_type
        state = self.states.setdefault(workflow_id, FakeWorkflowState())
        return FakeHandle(state)

    async def start_workflow(self, workflow: object, arg: object, **kwargs: object) -> FakeHandle:
        del workflow, arg
        self.start_calls.append(kwargs)
        workflow_id = str(kwargs["id"])
        state = self.states.setdefault(workflow_id, FakeWorkflowState())
        state.status = "RUNNING"
        return FakeHandle(state)

    async def list_workflows(self, query: str, limit: int) -> AsyncIterator[FakeWorkflowInfo]:
        del query, limit
        for workflow_id, state in self.states.items():
            if state.status == "RUNNING":
                yield FakeWorkflowInfo(workflow_id, state.start_time)


def _load_case(index: int = 0) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text())
    return payload["test_cases"][index]


def _sample_applicant(index: int = 0) -> MortgageApplication:
    return MortgageApplication(**_load_case(index))


def _sample_packet(case_id: str, decision: str = "CONDITIONAL") -> HumanReviewPacket:
    applicant = _sample_applicant(1)
    applicant.case_id = case_id
    metrics = compute_metrics(applicant)
    recommendation = DecisionRecommendation(
        decision=decision,
        risk_score=67,
        memo="Needs review.",
        conditions=[],
        human_review_reason="Needs review." if decision == "CONDITIONAL" else None,
    )
    return HumanReviewPacket(
        case_id=case_id,
        display_name="S. J.",
        sanitized_applicant=applicant,
        metrics=metrics,
        analyses=UnderwritingAnalyses(
            credit="credit",
            income="income",
            assets="assets",
            collateral="collateral",
        ),
        critic_review="critic",
        decision_recommendation=recommendation,
        risk_flags=["flag"],
        policy_violations=[],
        risk_score=recommendation.risk_score,
    )


def _sample_result(case_id: str) -> UnderwritingOutput:
    applicant = _sample_applicant(0)
    applicant.case_id = case_id
    metrics = compute_metrics(applicant)
    return UnderwritingOutput(
        case_id=case_id,
        sanitized_applicant=applicant,
        metrics=metrics,
        analyses=UnderwritingAnalyses(
            credit="credit",
            income="income",
            assets="assets",
            collateral="collateral",
        ),
        critic_review="critic",
        decision_memo="memo",
        final_decision="APPROVED",
        risk_score=88,
        risk_flags=["flag"],
        bias_flags=[],
        policy_violations=[],
        human_review_required=False,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _workflow_id(case_id: str) -> str:
    return f"mortgage-{case_id}"


@contextmanager
def configured_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_client: FakeClient,
) -> Iterator[tuple[TestClient, Path]]:
    """Configure the review app against a temporary upload root and fake client."""
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(review_app, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(review_app, "MANIFEST_PATH", upload_root / "cases.json")

    async def fake_connect(*_args: object, **_kwargs: object) -> FakeClient:
        return fake_client

    monkeypatch.setattr(review_app.Client, "connect", fake_connect)

    with TestClient(review_app.app) as client:
        yield client, upload_root


def test_upload_starts_workflow_and_persists_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Upload files, start a workflow, and persist the case entry."""
    fake_client = FakeClient()
    case_id = "MTG-20260324-UPLOAD01"
    monkeypatch.setattr(review_app, "_generate_case_id", lambda: case_id)

    with configured_client(monkeypatch, tmp_path, fake_client) as (client, upload_root):
        response = client.post(
            "/upload",
            files=[
                ("files", ("loan.png", b"png-bytes", "image/png")),
                ("files", ("loan.jpg", b"jpg-bytes", "image/jpeg")),
            ],
            follow_redirects=False,
        )

    manifest = json.loads((upload_root / "cases.json").read_text())

    assert response.status_code == 303
    assert response.headers["location"] == f"/case/{case_id}"
    assert fake_client.start_calls[-1]["id"] == _workflow_id(case_id)
    assert manifest == [
        {
            "case_id": case_id,
            "created_at": manifest[0]["created_at"],
            "image_dir": str(upload_root / case_id),
        }
    ]
    assert (upload_root / case_id / f"{case_id}_p1.png").read_bytes() == b"png-bytes"
    assert (upload_root / case_id / f"{case_id}_p2.jpg").read_bytes() == b"jpg-bytes"


def test_case_view_renders_review_form_for_conditional_packet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Show the submit-review form when a workflow is awaiting review."""
    case_id = "MTG-20260324-CASEVIEW"
    workflow_id = _workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="RUNNING",
                packet=_sample_packet(case_id),
            )
        }
    )

    with configured_client(monkeypatch, tmp_path, fake_client) as (client, _upload_root):
        response = client.get(f"/case/{case_id}")

    assert response.status_code == 200
    assert "Submit Review" in response.text
    assert "CONDITIONAL" in response.text
    assert "Needs review." in response.text


def test_case_view_renders_final_outcome_for_completed_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Render the final outcome block for completed workflows."""
    case_id = "MTG-20260324-COMPLETE"
    workflow_id = _workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="COMPLETED",
                result=_sample_result(case_id),
            )
        }
    )

    with configured_client(monkeypatch, tmp_path, fake_client) as (client, _upload_root):
        response = client.get(f"/case/{case_id}")

    assert response.status_code == 200
    assert "Final Outcome" in response.text
    assert "APPROVED" in response.text
    assert "Risk Score" in response.text


def test_submit_review_signals_running_conditional_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Signal the workflow when a valid human review is submitted."""
    case_id = "MTG-20260324-REVIEW01"
    workflow_id = _workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="RUNNING",
                packet=_sample_packet(case_id),
            )
        }
    )

    with configured_client(monkeypatch, tmp_path, fake_client) as (client, _upload_root):
        response = client.post(
            "/submit",
            data={
                "case_id": case_id,
                "reviewer": "QA Reviewer",
                "decision": "APPROVED",
                "notes": "Reviewed and approved.",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/case/{case_id}?message=Submitted%20review%20for%20{case_id}."
    )
    assert fake_client.states[workflow_id].signals == [
        HumanReviewInput(
            reviewer="QA Reviewer",
            decision="APPROVED",
            notes="Reviewed and approved.",
        )
    ]


def test_submit_review_rejects_non_conditional_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject review submissions unless the workflow is awaiting review."""
    case_id = "MTG-20260324-REVIEW02"
    workflow_id = _workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="RUNNING",
                packet=_sample_packet(case_id, decision="APPROVED"),
            )
        }
    )

    with configured_client(monkeypatch, tmp_path, fake_client) as (client, _upload_root):
        response = client.post(
            "/submit",
            data={
                "case_id": case_id,
                "reviewer": "QA Reviewer",
                "decision": "APPROVED",
                "notes": "Reviewed.",
            },
            follow_redirects=True,
        )

    assert response.status_code == 200
    assert "workflow is not awaiting human review" in response.text
    assert fake_client.states[workflow_id].signals == []

# ruff: noqa: D102, D107, EM101, PLR2004, SLF001, TC002, TRY003
"""FastAPI and unit tests for the insurance claims review app."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import TemporalError

from src.workflows.insurance_claims_fixed_flow import review_app
from src.workflows.insurance_claims_fixed_flow.insurance_models import (
    ClaimAdjudicationOutput,
    ClaimAnalyses,
    DecisionRecommendation,
    HumanReviewInput,
    HumanReviewPacket,
    InsuranceClaim,
)
from src.workflows.insurance_claims_fixed_flow.insurance_utils import compute_metrics

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = REPO_ROOT / "resources" / "insurance_claim_test_cases.json"


class FakeTemporalError(TemporalError):
    """Concrete TemporalError for review-app tests."""


class FakeExecutionStatus:
    """Small status shim that matches Temporal's describe response shape."""

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
    result: ClaimAdjudicationOutput | None = None
    result_error: Exception | None = None
    describe_error: Exception | None = None
    query_error: Exception | None = None
    signal_error: Exception | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    signals: list[HumanReviewInput] = field(default_factory=list)


class FakeHandle:
    """Fake workflow handle backed by a FakeWorkflowState."""

    def __init__(self, state: FakeWorkflowState) -> None:
        self.state = state

    async def describe(self) -> FakeDescription:
        if self.state.describe_error is not None:
            raise self.state.describe_error
        return FakeDescription(self.state)

    async def query(self, _query: object) -> HumanReviewPacket | None:
        if self.state.query_error is not None:
            raise self.state.query_error
        return self.state.packet

    async def result(self) -> ClaimAdjudicationOutput:
        if self.state.result_error is not None:
            raise self.state.result_error
        if self.state.result is None:
            raise FakeTemporalError("No result available.")
        return self.state.result

    async def signal(self, _signal: object, payload: HumanReviewInput) -> None:
        if self.state.signal_error is not None:
            raise self.state.signal_error
        self.state.signals.append(payload)


class FakeClient:
    """Small fake Temporal client for the review-app routes."""

    def __init__(self, states: dict[str, FakeWorkflowState] | None = None) -> None:
        self.states = states or {}
        self.start_calls: list[dict] = []
        self.start_error: Exception | None = None
        self.list_error: Exception | None = None

    def close(self) -> None:
        """Match the real client's close method."""

    def get_workflow_handle(self, workflow_id: str, result_type: type | None = None) -> FakeHandle:
        del result_type
        state = self.states.setdefault(workflow_id, FakeWorkflowState())
        return FakeHandle(state)

    async def start_workflow(self, workflow: object, arg: object, **kwargs: object) -> FakeHandle:
        del workflow, arg
        self.start_calls.append(kwargs)
        if self.start_error is not None:
            raise self.start_error

        workflow_id = str(kwargs["id"])
        state = self.states.setdefault(workflow_id, FakeWorkflowState())
        state.status = "RUNNING"
        state.packet = None
        state.result = None
        state.result_error = None
        return FakeHandle(state)

    async def list_workflows(self, query: str, limit: int) -> Iterator[FakeWorkflowInfo]:
        del query, limit
        if self.list_error is not None:
            raise self.list_error

        for workflow_id, state in self.states.items():
            if state.status == "RUNNING":
                yield FakeWorkflowInfo(workflow_id, state.start_time)


def _load_case(index: int = 0) -> dict:
    payload = json.loads(FIXTURE_PATH.read_text())
    return payload["test_cases"][index]


def _sample_claim() -> InsuranceClaim:
    return InsuranceClaim(**_load_case(0))


def _sample_packet(decision: str = "CONDITIONAL") -> HumanReviewPacket:
    claim = _sample_claim()
    metrics = compute_metrics(claim)
    recommendation = DecisionRecommendation(
        decision=decision,
        risk_score=67,
        memo="Needs review.",
        conditions=[],
        human_review_reason="Needs review." if decision == "CONDITIONAL" else None,
    )
    return HumanReviewPacket(
        case_id=claim.case_id,
        display_name="E. R.",
        sanitized_claim=claim,
        metrics=metrics,
        analyses=ClaimAnalyses(
            coverage="coverage",
            liability="liability",
            damages="damages",
            fraud="fraud",
        ),
        critic_review="critic",
        decision_recommendation=recommendation,
        risk_flags=["flag"],
        policy_violations=[],
        risk_score=recommendation.risk_score,
    )


def _sample_result() -> ClaimAdjudicationOutput:
    claim = _sample_claim()
    metrics = compute_metrics(claim)
    return ClaimAdjudicationOutput(
        case_id=claim.case_id,
        sanitized_claim=claim,
        metrics=metrics,
        analyses=ClaimAnalyses(
            coverage="coverage",
            liability="liability",
            damages="damages",
            fraud="fraud",
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


def _case_id(suffix: str) -> str:
    return f"{datetime.now(UTC).strftime('CLM-%Y%m%d')}-{suffix}"


@contextmanager
def configured_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fake_client: FakeClient | None = None,
    connect_error: Exception | None = None,
) -> Iterator[tuple[TestClient, Path]]:
    """Configure the review app against a temporary upload root and fake client."""
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(review_app, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(review_app, "MANIFEST_PATH", upload_root / "cases.json")

    async def fake_connect() -> FakeClient:
        if connect_error is not None:
            raise connect_error
        assert fake_client is not None
        return fake_client

    monkeypatch.setattr(review_app, "_connect_temporal_client", fake_connect)

    with TestClient(review_app.app) as client:
        yield client, upload_root


def test_generate_case_id_skips_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generate a unique case id even when the first suffix collides."""
    seen_suffixes = iter(["DUPL0001", "UNIQ0002"])
    monkeypatch.setattr(review_app, "_random_case_suffix", lambda: next(seen_suffixes))

    duplicate = _case_id("DUPL0001")
    assert review_app._generate_case_id({duplicate}) == _case_id("UNIQ0002")


def test_manifest_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Write and read manifest records atomically."""
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(review_app, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(review_app, "MANIFEST_PATH", upload_root / "cases.json")

    record = review_app.CaseRecord(
        case_id=_case_id("ROUNDTRIP"),
        workflow_id=review_app._workflow_id(_case_id("ROUNDTRIP")),
        status=review_app.CaseStatus.RUNNING,
        created_at=datetime.now(UTC).isoformat(),
        updated_at=datetime.now(UTC).isoformat(),
        image_dir=str(upload_root / "images"),
        image_count=2,
    )

    review_app._write_manifest_records([record])
    records, warning = review_app._read_manifest_records()

    assert warning is None
    assert records == [record]


def test_corrupted_manifest_falls_back_to_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fall back to scanning directories when the manifest is corrupted."""
    upload_root = tmp_path / "uploads"
    case_id = _case_id("SCAN0001")
    case_dir = upload_root / case_id
    case_dir.mkdir(parents=True)
    (case_dir / f"{case_id}_p1.png").write_bytes(b"png")

    monkeypatch.setattr(review_app, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(review_app, "MANIFEST_PATH", upload_root / "cases.json")
    review_app.MANIFEST_PATH.write_text("{invalid")

    records, warning = review_app._read_manifest_records()

    assert warning is not None
    assert records[0].case_id == case_id
    assert records[0].status == review_app.CaseStatus.QUEUED


def test_derive_case_status_identifies_awaiting_review() -> None:
    """Map running workflows with a conditional review packet to awaiting review."""
    conditional_packet = _sample_packet(decision="CONDITIONAL")
    approved_packet = _sample_packet(decision="APPROVED")

    assert (
        review_app._derive_case_status("RUNNING", conditional_packet)
        == review_app.CaseStatus.AWAITING_REVIEW
    )
    assert (
        review_app._derive_case_status("RUNNING", approved_packet)
        == review_app.CaseStatus.RUNNING
    )


def test_healthz_and_readyz_show_degraded_temporal_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the app healthy while failing readiness when Temporal is unavailable."""
    with configured_client(
        monkeypatch,
        tmp_path,
        connect_error=OSError("offline"),
    ) as (client, _upload_root):
        health = client.get("/healthz")
        ready = client.get("/readyz")
        home = client.get("/")

    assert health.status_code == 200
    assert ready.status_code == 503
    assert "offline" in ready.text
    assert "Temporal is unavailable" in home.text


def test_upload_success_persists_running_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Upload files, start a workflow, and persist a RUNNING case record."""
    fake_client = FakeClient()
    case_id = _case_id("UPLOAD01")
    monkeypatch.setattr(review_app, "_random_case_suffix", lambda: "UPLOAD01")

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, upload_root):
        response = client.post(
            "/upload",
            files=[("files", ("claim.png", b"image-bytes", "image/png"))],
            follow_redirects=False,
        )
        records, warning = review_app._read_manifest_records()

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/case/{case_id}")
    assert warning is None
    assert records[0].case_id == case_id
    assert records[0].status == review_app.CaseStatus.RUNNING
    assert records[0].image_count == 1
    assert fake_client.start_calls[-1]["id_reuse_policy"] == WorkflowIDReusePolicy.REJECT_DUPLICATE
    assert (upload_root / case_id / f"{case_id}_p1.png").read_bytes() == b"image-bytes"


def test_upload_rejects_invalid_file_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject uploads whose file extension is not supported."""
    fake_client = FakeClient()

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, _upload_root):
        response = client.post(
            "/upload",
            files=[("files", ("claim.txt", b"not-an-image", "text/plain"))],
            follow_redirects=True,
        )

    assert response.status_code == 200
    assert "Unsupported file type" in response.text
    assert fake_client.start_calls == []


def test_upload_rejects_empty_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject uploads whose file payload is empty."""
    fake_client = FakeClient()

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, _upload_root):
        response = client.post(
            "/upload",
            files=[("files", ("claim.png", b"", "image/png"))],
            follow_redirects=True,
        )

    assert response.status_code == 200
    assert "is empty" in response.text
    assert fake_client.start_calls == []


def test_upload_start_failure_persists_start_failed_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist a START_FAILED case when Temporal start_workflow fails."""
    fake_client = FakeClient()
    fake_client.start_error = FakeTemporalError("workflow start failed")
    case_id = _case_id("START001")
    monkeypatch.setattr(review_app, "_random_case_suffix", lambda: "START001")

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, _upload_root):
        response = client.post(
            "/upload",
            files=[("files", ("claim.png", b"image-bytes", "image/png"))],
            follow_redirects=True,
        )
        records, _ = review_app._read_manifest_records()

    assert response.status_code == 200
    assert case_id in response.text
    assert "START_FAILED" in response.text
    assert records[0].status == review_app.CaseStatus.START_FAILED
    assert "workflow start failed" in (records[0].last_error or "")


def test_failed_case_view_surfaces_error_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Render a failed workflow error and allow retrying it from the case page."""
    case_id = _case_id("FAILED01")
    workflow_id = review_app._workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="FAILED",
                result_error=FakeTemporalError("LLM crashed"),
            )
        }
    )

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, upload_root):
        case_dir = upload_root / case_id
        case_dir.mkdir(parents=True)
        (case_dir / f"{case_id}_p1.png").write_bytes(b"png")
        review_app._write_manifest_records(
            [
                review_app.CaseRecord(
                    case_id=case_id,
                    workflow_id=workflow_id,
                    status=review_app.CaseStatus.RUNNING,
                    created_at=datetime.now(UTC).isoformat(),
                    updated_at=datetime.now(UTC).isoformat(),
                    image_dir=str(case_dir),
                    image_count=1,
                )
            ]
        )

        page = client.get(f"/case/{case_id}")
        retry = client.post(f"/case/{case_id}/retry", follow_redirects=False)
        records, _ = review_app._read_manifest_records()

    assert page.status_code == 200
    assert "FAILED" in page.text
    assert "LLM crashed" in page.text
    assert retry.status_code == 303
    assert fake_client.start_calls[-1]["id_reuse_policy"] == (
        WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY
    )
    assert records[0].status == review_app.CaseStatus.RUNNING
    assert records[0].retry_count == 1


def test_retry_rejects_non_failed_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject retry attempts when a case is not in a retryable state."""
    case_id = _case_id("RUNNING01")
    workflow_id = review_app._workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="RUNNING",
                packet=_sample_packet(decision="APPROVED"),
            )
        }
    )

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, upload_root):
        case_dir = upload_root / case_id
        case_dir.mkdir(parents=True)
        review_app._write_manifest_records(
            [
                review_app.CaseRecord(
                    case_id=case_id,
                    workflow_id=workflow_id,
                    status=review_app.CaseStatus.RUNNING,
                    created_at=datetime.now(UTC).isoformat(),
                    updated_at=datetime.now(UTC).isoformat(),
                    image_dir=str(case_dir),
                    image_count=1,
                )
            ]
        )

        response = client.post(f"/case/{case_id}/retry", follow_redirects=True)

    assert response.status_code == 200
    assert "Cannot retry a case in RUNNING state." in response.text
    assert fake_client.start_calls == []


def test_submit_review_requires_awaiting_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject review submissions unless the workflow is awaiting review."""
    case_id = _case_id("REVIEW01")
    workflow_id = review_app._workflow_id(case_id)
    fake_client = FakeClient(
        states={
            workflow_id: FakeWorkflowState(
                status="RUNNING",
                packet=_sample_packet(decision="APPROVED"),
            )
        }
    )

    with configured_client(
        monkeypatch,
        tmp_path,
        fake_client=fake_client,
    ) as (client, _upload_root):
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

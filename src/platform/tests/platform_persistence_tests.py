"""Tests for the demo Lakebase repository and downstream claims adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.platform.downstream.claims_adapter import ClaimsDownstreamAdapter
from src.platform.lakebase.models import (
    AuditEventRecord,
    DownstreamActionRecord,
    OperationalCaseRecord,
    ReviewTaskRecord,
    WorkflowRunRecord,
)
from src.platform.lakebase.repository import LakebaseRepository

if TYPE_CHECKING:
    from pathlib import Path


def test_lakebase_repository_round_trip(tmp_path: Path) -> None:
    """Persist and read the core operational records."""
    repository = LakebaseRepository(tmp_path / "lakebase.sqlite3")

    repository.upsert_case(
        OperationalCaseRecord(
            case_id="CLM-123",
            domain="insurance_claim",
            status="RUNNING",
            display_name="E. R.",
            current_decision=None,
            payload={"stage": "INTAKE"},
            created_at="2026-03-31T00:00:00+00:00",
            updated_at="2026-03-31T00:00:00+00:00",
        )
    )
    repository.upsert_workflow_run(
        WorkflowRunRecord(
            workflow_id="insurance-CLM-123",
            run_id="run-1",
            case_id="CLM-123",
            orchestration_mode="fixed",
            stage="AWAITING_REVIEW",
            status="AWAITING_REVIEW",
            payload={"risk_score": 72},
            started_at="2026-03-31T00:00:00+00:00",
            updated_at="2026-03-31T00:05:00+00:00",
        )
    )
    repository.upsert_review_task(
        ReviewTaskRecord(
            review_task_id="CLM-123:human-review",
            case_id="CLM-123",
            workflow_id="insurance-CLM-123",
            status="OPEN",
            reason="High loss amount",
            reviewer=None,
            decision=None,
            notes=None,
            payload={"risk_flags": ["high_claim_amount"]},
            created_at="2026-03-31T00:05:00+00:00",
            updated_at="2026-03-31T00:05:00+00:00",
        )
    )
    repository.append_audit_event(
        AuditEventRecord(
            case_id="CLM-123",
            workflow_id="insurance-CLM-123",
            event_type="review_task_updated",
            payload={"status": "OPEN"},
            created_at="2026-03-31T00:05:00+00:00",
        )
    )
    repository.upsert_downstream_action(
        DownstreamActionRecord(
            idempotency_key="insurance-CLM-123:claim-decision",
            case_id="CLM-123",
            workflow_id="insurance-CLM-123",
            target_system="claims-system-demo",
            action_name="claim_decision_sync",
            status="PUBLISHED",
            external_record_id="CLAIM-SYNC-CLM-123",
            payload={"final_decision": "APPROVED"},
            created_at="2026-03-31T00:06:00+00:00",
            updated_at="2026-03-31T00:06:00+00:00",
        )
    )

    assert repository.fetch_case("CLM-123") is not None
    assert repository.fetch_workflow_run("insurance-CLM-123") is not None
    assert repository.fetch_review_task("CLM-123:human-review") is not None
    assert repository.fetch_downstream_action("insurance-CLM-123:claim-decision") is not None
    assert repository.list_audit_events("insurance-CLM-123")[0].event_type == "review_task_updated"


def test_claims_adapter_is_idempotent(tmp_path: Path) -> None:
    """Return the same external record when publishing the same action twice."""
    adapter = ClaimsDownstreamAdapter(tmp_path / "claims_outbox.jsonl")

    first = adapter.publish_claim_decision(
        case_id="CLM-456",
        workflow_id="insurance-CLM-456",
        idempotency_key="insurance-CLM-456:claim-decision",
        final_decision="APPROVED",
        risk_score=81,
        decision_memo="Approved.",
        reviewer="A. Reviewer",
    )
    second = adapter.publish_claim_decision(
        case_id="CLM-456",
        workflow_id="insurance-CLM-456",
        idempotency_key="insurance-CLM-456:claim-decision",
        final_decision="APPROVED",
        risk_score=81,
        decision_memo="Approved.",
        reviewer="A. Reviewer",
    )

    assert first["external_record_id"] == second["external_record_id"]
    assert first["action_status"] == "PUBLISHED"
    assert second["action_status"] == "ALREADY_PUBLISHED"

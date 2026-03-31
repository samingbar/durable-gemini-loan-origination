"""Operational-store and downstream activities for insurance claims."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from temporalio import activity

from src.platform.downstream.claims_adapter import ClaimsDownstreamAdapter
from src.platform.lakebase.models import (
    AnalysisResultRecord,
    AuditEventRecord,
    DownstreamActionRecord,
    ReviewTaskRecord,
    WorkflowRunRecord,
)
from src.platform.lakebase.repository import LakebaseRepository

from .insurance_models import (
    ClaimSystemUpdateTask,
    DownstreamActionResult,
    OperationalAnalysisTask,
    OperationalClaimStateTask,
    OperationalReviewTask,
)


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _case_payload(task: OperationalClaimStateTask) -> dict[str, object]:
    """Build a stable payload for the external case record."""
    payload = dict(task.metadata)
    payload["stage"] = task.stage
    if task.image_dir:
        payload["image_dir"] = task.image_dir
    if task.risk_score is not None:
        payload["risk_score"] = task.risk_score
    if task.human_review_required is not None:
        payload["human_review_required"] = task.human_review_required
    return payload


def _record_case_state_sync(task: OperationalClaimStateTask) -> None:
    """Persist the latest case and workflow state."""
    repository = LakebaseRepository()
    now = _utc_now_iso()
    repository.touch_case(
        case_id=task.case_id,
        domain="insurance_claim",
        status=task.status,
        display_name=task.display_name,
        current_decision=task.current_decision,
        payload=_case_payload(task),
    )
    existing_run = repository.fetch_workflow_run(task.workflow_id)
    repository.upsert_workflow_run(
        WorkflowRunRecord(
            workflow_id=task.workflow_id,
            run_id=task.run_id,
            case_id=task.case_id,
            orchestration_mode=task.orchestration_mode,
            stage=task.stage,
            status=task.status,
            payload=_case_payload(task),
            started_at=existing_run.started_at if existing_run is not None else now,
            updated_at=now,
        )
    )
    repository.append_audit_event(
        AuditEventRecord(
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            event_type="case_state_recorded",
            payload={
                "stage": task.stage,
                "status": task.status,
                "current_decision": task.current_decision,
            },
            created_at=now,
        )
    )


def _record_analysis_sync(task: OperationalAnalysisTask) -> None:
    """Persist an analysis artifact and append an audit event."""
    repository = LakebaseRepository()
    now = _utc_now_iso()
    repository.upsert_analysis_result(
        AnalysisResultRecord(
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            analysis_name=task.analysis_name,
            summary=task.summary,
            payload=task.payload,
            created_at=now,
        )
    )
    repository.append_audit_event(
        AuditEventRecord(
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            event_type="analysis_recorded",
            payload={"analysis_name": task.analysis_name},
            created_at=now,
        )
    )


def _record_review_task_sync(task: OperationalReviewTask) -> None:
    """Persist or update a review task in the operational store."""
    repository = LakebaseRepository()
    now = _utc_now_iso()
    existing = repository.fetch_review_task(task.review_task_id)
    repository.upsert_review_task(
        ReviewTaskRecord(
            review_task_id=task.review_task_id,
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            status=task.status,
            reason=task.reason,
            reviewer=task.reviewer,
            decision=task.decision,
            notes=task.notes,
            payload=task.payload,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
    )
    repository.append_audit_event(
        AuditEventRecord(
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            event_type="review_task_updated",
            payload={
                "review_task_id": task.review_task_id,
                "status": task.status,
                "decision": task.decision,
            },
            created_at=now,
        )
    )


def _publish_claim_update_sync(task: ClaimSystemUpdateTask) -> DownstreamActionResult:
    """Publish the final decision to the downstream claims outbox."""
    repository = LakebaseRepository()
    adapter = ClaimsDownstreamAdapter()
    idempotency_key = f"{task.workflow_id}:claim-decision"
    payload = adapter.publish_claim_decision(
        case_id=task.case_id,
        workflow_id=task.workflow_id,
        idempotency_key=idempotency_key,
        final_decision=task.final_decision,
        risk_score=task.risk_score,
        decision_memo=task.decision_memo,
        reviewer=task.reviewer,
    )
    now = _utc_now_iso()
    action_status = str(payload["action_status"])
    repository.upsert_downstream_action(
        DownstreamActionRecord(
            idempotency_key=idempotency_key,
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            target_system=adapter.target_system,
            action_name="claim_decision_sync",
            status=action_status,
            external_record_id=str(payload["external_record_id"]),
            payload={
                "final_decision": task.final_decision,
                "risk_score": task.risk_score,
                "reviewer": task.reviewer,
                "metadata": task.metadata,
                "outbox_path": str(adapter.outbox_path),
            },
            created_at=now,
            updated_at=now,
        )
    )
    repository.append_audit_event(
        AuditEventRecord(
            case_id=task.case_id,
            workflow_id=task.workflow_id,
            event_type="downstream_action_recorded",
            payload={
                "idempotency_key": idempotency_key,
                "status": action_status,
                "target_system": adapter.target_system,
            },
            created_at=now,
        )
    )
    return DownstreamActionResult(
        target_system=adapter.target_system,
        action_name="claim_decision_sync",
        status=action_status,
        external_record_id=str(payload["external_record_id"]),
        idempotency_key=idempotency_key,
        outbox_location=str(adapter.outbox_path),
    )


@activity.defn
async def record_case_state(task: OperationalClaimStateTask) -> None:
    """Persist externally visible case and workflow state."""
    activity.logger.info("Recording insurance case state %s for %s", task.stage, task.case_id)
    await asyncio.to_thread(_record_case_state_sync, task)


@activity.defn
async def record_analysis_result(task: OperationalAnalysisTask) -> None:
    """Persist a specialist or decision analysis artifact."""
    activity.logger.info("Recording %s analysis for %s", task.analysis_name, task.case_id)
    await asyncio.to_thread(_record_analysis_sync, task)


@activity.defn
async def upsert_review_task(task: OperationalReviewTask) -> None:
    """Persist the current review work item state."""
    activity.logger.info("Upserting review task %s for %s", task.review_task_id, task.case_id)
    await asyncio.to_thread(_record_review_task_sync, task)


@activity.defn
async def publish_claim_update(task: ClaimSystemUpdateTask) -> DownstreamActionResult:
    """Publish the claim decision to the demo downstream adapter."""
    activity.logger.info("Publishing downstream claim decision for %s", task.case_id)
    return await asyncio.to_thread(_publish_claim_update_sync, task)

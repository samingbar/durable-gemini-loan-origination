"""Pydantic models for the demo operational store."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OperationalCaseRecord(BaseModel):
    """Business-facing case state mirrored from workflow progress."""

    case_id: str
    domain: str
    status: str
    display_name: str | None = None
    current_decision: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class WorkflowRunRecord(BaseModel):
    """Workflow execution metadata mirrored to the operational store."""

    workflow_id: str
    run_id: str
    case_id: str
    orchestration_mode: str
    stage: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    updated_at: str


class AnalysisResultRecord(BaseModel):
    """Structured or unstructured specialist output captured for audit."""

    case_id: str
    workflow_id: str
    analysis_name: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ReviewTaskRecord(BaseModel):
    """Human review work item mirrored to the operational store."""

    review_task_id: str
    case_id: str
    workflow_id: str
    status: str
    reason: str | None = None
    reviewer: str | None = None
    decision: str | None = None
    notes: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class AuditEventRecord(BaseModel):
    """Append-only audit event persisted outside workflow history."""

    case_id: str
    workflow_id: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class DownstreamActionRecord(BaseModel):
    """Idempotent external action persisted for audit and replay safety."""

    idempotency_key: str
    case_id: str
    workflow_id: str
    target_system: str
    action_name: str
    status: str
    external_record_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

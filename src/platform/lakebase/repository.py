"""SQLite-backed demo repository that mirrors the intended Lakebase contracts."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from .models import (
    AnalysisResultRecord,
    AuditEventRecord,
    DownstreamActionRecord,
    OperationalCaseRecord,
    ReviewTaskRecord,
    WorkflowRunRecord,
)


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _default_db_path() -> Path:
    """Return the default local database path for the demo operational store."""
    configured = os.environ.get("LAKEBASE_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return _repo_root() / "datasets" / "operations" / "lakebase_demo.sqlite3"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


class LakebaseRepository:
    """Persist workflow-visible business state in a local SQLite database.

    The real thesis target is Lakebase/Postgres. SQLite keeps the same write
    points available for local demos and tests without introducing new runtime
    dependencies.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the repository and ensure the schema exists."""
        self._db_path = db_path or _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        """Return the backing SQLite file path."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Open a connection configured for predictable row handling."""
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _ensure_schema(self) -> None:
        """Create the operational-store tables if they do not exist."""
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    status TEXT NOT NULL,
                    display_name TEXT,
                    current_decision TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    workflow_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    orchestration_mode TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_results (
                    case_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    analysis_name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (workflow_id, analysis_name)
                );
                CREATE TABLE IF NOT EXISTS review_tasks (
                    review_task_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    reviewer TEXT,
                    decision TEXT,
                    notes TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS downstream_actions (
                    idempotency_key TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    target_system TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    external_record_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _json(payload: dict[str, object]) -> str:
        """Encode a payload as stable JSON."""
        return json.dumps(payload, sort_keys=True)

    def upsert_case(self, record: OperationalCaseRecord) -> None:
        """Insert or update a case record."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO cases (
                    case_id, domain, status, display_name, current_decision,
                    payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    domain=excluded.domain,
                    status=excluded.status,
                    display_name=excluded.display_name,
                    current_decision=excluded.current_decision,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.case_id,
                    record.domain,
                    record.status,
                    record.display_name,
                    record.current_decision,
                    self._json(record.payload),
                    record.created_at,
                    record.updated_at,
                ),
            )

    def fetch_case(self, case_id: str) -> OperationalCaseRecord | None:
        """Return a case record if present."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT case_id, domain, status, display_name, current_decision,
                       payload_json, created_at, updated_at
                FROM cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return OperationalCaseRecord(
            case_id=row["case_id"],
            domain=row["domain"],
            status=row["status"],
            display_name=row["display_name"],
            current_decision=row["current_decision"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert_workflow_run(self, record: WorkflowRunRecord) -> None:
        """Insert or update a workflow run record."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    workflow_id, run_id, case_id, orchestration_mode, stage,
                    status, payload_json, started_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    case_id=excluded.case_id,
                    orchestration_mode=excluded.orchestration_mode,
                    stage=excluded.stage,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.workflow_id,
                    record.run_id,
                    record.case_id,
                    record.orchestration_mode,
                    record.stage,
                    record.status,
                    self._json(record.payload),
                    record.started_at,
                    record.updated_at,
                ),
            )

    def fetch_workflow_run(self, workflow_id: str) -> WorkflowRunRecord | None:
        """Return a workflow run record if present."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT workflow_id, run_id, case_id, orchestration_mode, stage,
                       status, payload_json, started_at, updated_at
                FROM workflow_runs
                WHERE workflow_id = ?
                """,
                (workflow_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkflowRunRecord(
            workflow_id=row["workflow_id"],
            run_id=row["run_id"],
            case_id=row["case_id"],
            orchestration_mode=row["orchestration_mode"],
            stage=row["stage"],
            status=row["status"],
            payload=json.loads(row["payload_json"]),
            started_at=row["started_at"],
            updated_at=row["updated_at"],
        )

    def upsert_analysis_result(self, record: AnalysisResultRecord) -> None:
        """Insert or update a named analysis result for a workflow."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO analysis_results (
                    case_id, workflow_id, analysis_name, summary, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id, analysis_name) DO UPDATE SET
                    case_id=excluded.case_id,
                    summary=excluded.summary,
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at
                """,
                (
                    record.case_id,
                    record.workflow_id,
                    record.analysis_name,
                    record.summary,
                    self._json(record.payload),
                    record.created_at,
                ),
            )

    def list_analysis_results(self, workflow_id: str) -> list[AnalysisResultRecord]:
        """Return all analysis results for a workflow."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT case_id, workflow_id, analysis_name, summary, payload_json, created_at
                FROM analysis_results
                WHERE workflow_id = ?
                ORDER BY analysis_name
                """,
                (workflow_id,),
            ).fetchall()
        return [
            AnalysisResultRecord(
                case_id=row["case_id"],
                workflow_id=row["workflow_id"],
                analysis_name=row["analysis_name"],
                summary=row["summary"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def upsert_review_task(self, record: ReviewTaskRecord) -> None:
        """Insert or update a review task."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO review_tasks (
                    review_task_id, case_id, workflow_id, status, reason, reviewer,
                    decision, notes, payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_task_id) DO UPDATE SET
                    case_id=excluded.case_id,
                    workflow_id=excluded.workflow_id,
                    status=excluded.status,
                    reason=excluded.reason,
                    reviewer=excluded.reviewer,
                    decision=excluded.decision,
                    notes=excluded.notes,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.review_task_id,
                    record.case_id,
                    record.workflow_id,
                    record.status,
                    record.reason,
                    record.reviewer,
                    record.decision,
                    record.notes,
                    self._json(record.payload),
                    record.created_at,
                    record.updated_at,
                ),
            )

    def fetch_review_task(self, review_task_id: str) -> ReviewTaskRecord | None:
        """Return a review task if present."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT review_task_id, case_id, workflow_id, status, reason, reviewer,
                       decision, notes, payload_json, created_at, updated_at
                FROM review_tasks
                WHERE review_task_id = ?
                """,
                (review_task_id,),
            ).fetchone()
        if row is None:
            return None
        return ReviewTaskRecord(
            review_task_id=row["review_task_id"],
            case_id=row["case_id"],
            workflow_id=row["workflow_id"],
            status=row["status"],
            reason=row["reason"],
            reviewer=row["reviewer"],
            decision=row["decision"],
            notes=row["notes"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def append_audit_event(self, record: AuditEventRecord) -> None:
        """Append an audit event."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    case_id, workflow_id, event_type, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.case_id,
                    record.workflow_id,
                    record.event_type,
                    self._json(record.payload),
                    record.created_at,
                ),
            )

    def list_audit_events(self, workflow_id: str) -> list[AuditEventRecord]:
        """Return audit events for a workflow."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT case_id, workflow_id, event_type, payload_json, created_at
                FROM audit_events
                WHERE workflow_id = ?
                ORDER BY event_id
                """,
                (workflow_id,),
            ).fetchall()
        return [
            AuditEventRecord(
                case_id=row["case_id"],
                workflow_id=row["workflow_id"],
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def upsert_downstream_action(self, record: DownstreamActionRecord) -> None:
        """Insert or update an idempotent downstream action record."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT INTO downstream_actions (
                    idempotency_key, case_id, workflow_id, target_system, action_name,
                    status, external_record_id, payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    case_id=excluded.case_id,
                    workflow_id=excluded.workflow_id,
                    target_system=excluded.target_system,
                    action_name=excluded.action_name,
                    status=excluded.status,
                    external_record_id=excluded.external_record_id,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.idempotency_key,
                    record.case_id,
                    record.workflow_id,
                    record.target_system,
                    record.action_name,
                    record.status,
                    record.external_record_id,
                    self._json(record.payload),
                    record.created_at,
                    record.updated_at,
                ),
            )

    def fetch_downstream_action(self, idempotency_key: str) -> DownstreamActionRecord | None:
        """Return a downstream action if present."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT idempotency_key, case_id, workflow_id, target_system, action_name,
                       status, external_record_id, payload_json, created_at, updated_at
                FROM downstream_actions
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return DownstreamActionRecord(
            idempotency_key=row["idempotency_key"],
            case_id=row["case_id"],
            workflow_id=row["workflow_id"],
            target_system=row["target_system"],
            action_name=row["action_name"],
            status=row["status"],
            external_record_id=row["external_record_id"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def touch_case(  # noqa: PLR0913
        self,
        *,
        case_id: str,
        domain: str,
        status: str,
        display_name: str | None = None,
        current_decision: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> OperationalCaseRecord:
        """Convenience helper for callers that need generated timestamps."""
        existing = self.fetch_case(case_id)
        created_at = existing.created_at if existing is not None else _utc_now_iso()
        record = OperationalCaseRecord(
            case_id=case_id,
            domain=domain,
            status=status,
            display_name=display_name,
            current_decision=current_decision,
            payload=payload or {},
            created_at=created_at,
            updated_at=_utc_now_iso(),
        )
        self.upsert_case(record)
        return record

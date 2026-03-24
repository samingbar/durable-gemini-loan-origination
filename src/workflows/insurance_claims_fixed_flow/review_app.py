"""FastAPI-based human review UI for insurance claim adjudication."""

from __future__ import annotations

import asyncio
import html
import json
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from pydantic import BaseModel, ValidationError
from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import TemporalError

from .insurance_models import (
    ClaimAdjudicationInput,
    ClaimAdjudicationOutput,
    HumanReviewInput,
    HumanReviewPacket,
)
from .insurance_workflow import InsuranceClaimAdjudicationWorkflow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TASK_QUEUE = os.environ.get("INSURANCE_TASK_QUEUE", "insurance-claims")
UPLOAD_ROOT = Path(os.environ.get("INSURANCE_UPLOAD_ROOT", "datasets/uploads/insurance_claims"))
MANIFEST_PATH = UPLOAD_ROOT / "cases.json"
RUNNING_WORKFLOW_QUERY = (
    'WorkflowType = "InsuranceClaimAdjudicationWorkflow" and ExecutionStatus = "Running"'
)
ALLOWED_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_STATUS_CONCURRENCY = 10

CaseIdForm = Annotated[str, Form(...)]
ReviewerForm = Annotated[str, Form(...)]
DecisionForm = Annotated[str, Form(...)]
NotesForm = Annotated[str, Form(...)]
UploadFiles = Annotated[list[UploadFile], File(...)]


class CaseStatus(StrEnum):
    """Persisted workflow and UI status for an uploaded case."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    START_FAILED = "START_FAILED"


ACTIVE_CASE_STATUSES = {
    CaseStatus.QUEUED,
    CaseStatus.RUNNING,
    CaseStatus.AWAITING_REVIEW,
}
RETRYABLE_CASE_STATUSES = {CaseStatus.START_FAILED, CaseStatus.FAILED}
FAILED_TEMPORAL_STATUSES = {"FAILED", "TIMED_OUT", "TERMINATED", "CANCELED"}


class CaseRecord(BaseModel):
    """Persisted metadata for a UI-visible claim case."""

    case_id: str
    workflow_id: str
    status: CaseStatus
    created_at: str
    updated_at: str
    image_dir: str
    image_count: int
    last_error: str | None = None
    retry_count: int = 0


@dataclass(slots=True)
class CaseSnapshot:
    """Case data combined with live Temporal state for rendering."""

    record: CaseRecord
    packet: HumanReviewPacket | None = None
    result: ClaimAdjudicationOutput | None = None
    failure_error: str | None = None
    lookup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up shared app state for the review UI."""
    app.state.manifest_lock = asyncio.Lock()
    app.state.temporal_lock = asyncio.Lock()
    app.state.client = None
    app.state.temporal_error = None
    app.state.upload_root_error = _ensure_upload_root_exists()

    if app.state.upload_root_error is None:
        client = await _ensure_client()
        if client is None and app.state.temporal_error is None:
            app.state.temporal_error = "Temporal is unavailable."

    try:
        yield
    finally:
        client = getattr(app.state, "client", None)
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


app = FastAPI(title="Insurance Claims Human Review", lifespan=lifespan)


def _page(title: str, body: str, refresh_seconds: int | None = None) -> HTMLResponse:
    """Render a small standalone HTML page."""
    refresh_meta = ""
    if refresh_seconds:
        refresh_meta = f'<meta http-equiv="refresh" content="{refresh_seconds}">'
    html_text = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  {refresh_meta}
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1080px;
      margin: 32px auto;
      padding: 0 20px 40px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    label {{ display: block; margin-top: 12px; font-weight: 600; }}
    input, select, textarea {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      font-size: 14px;
    }}
    textarea {{ min-height: 120px; }}
    button {{
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      border: none;
      background: var(--accent);
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }}
    button.secondary {{
      background: #111827;
    }}
    .card {{
      border: 1px solid var(--border);
      padding: 16px;
      border-radius: 12px;
      background: var(--card);
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      margin-top: 16px;
    }}
    .muted {{ color: var(--muted); }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .inline-form {{
      margin: 0;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      background: #f3f4f6;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
      line-height: 1.5;
    }}
    ul {{ padding-left: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .status-queued {{ background: #e5e7eb; color: #374151; }}
    .status-running {{ background: #dbeafe; color: #1d4ed8; }}
    .status-awaiting-review {{ background: #fef3c7; color: #92400e; }}
    .status-completed {{ background: #dcfce7; color: #166534; }}
    .status-failed, .status-start-failed {{ background: #fee2e2; color: #991b1b; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .alert {{
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 16px;
      border: 1px solid transparent;
      font-weight: 600;
    }}
    .alert-success {{
      background: #ecfdf3;
      color: #065f46;
      border-color: #d1fae5;
    }}
    .alert-error {{
      background: #fef2f2;
      color: #991b1b;
      border-color: #fee2e2;
    }}
    .alert-warning {{
      background: #fffbeb;
      color: #92400e;
      border-color: #fde68a;
    }}
  </style>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""
    return HTMLResponse(html_text)


def _workflow_id(case_id: str) -> str:
    """Build the canonical workflow id for a case."""
    return f"insurance-{case_id}"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _max_files() -> int:
    """Return the maximum upload count for a single case."""
    return int(os.environ.get("INSURANCE_MAX_FILES", "10"))


def _max_file_bytes() -> int:
    """Return the maximum bytes allowed for one uploaded image."""
    return int(os.environ.get("INSURANCE_MAX_FILE_BYTES", "10485760"))


def _max_total_bytes() -> int:
    """Return the maximum bytes allowed for one upload batch."""
    return int(os.environ.get("INSURANCE_MAX_TOTAL_BYTES", "26214400"))


def _escape(value: object) -> str:
    """Escape values for HTML output."""
    return html.escape("" if value is None else str(value))


def _format_percent(value: float) -> str:
    """Format a float as a percentage."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_money(value: float) -> str:
    """Format a float as a currency string."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _render_list(items: list[str]) -> str:
    """Render a list of items or a muted empty state."""
    if not items:
        return '<p class="muted">None</p>'
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _render_status_badge(status: CaseStatus) -> str:
    """Render a colored status badge."""
    css_class = status.value.lower().replace("_", "-")
    return f'<span class="badge status-{css_class}">{_escape(status.value)}</span>'


def _render_alert(level: str, message: str) -> str:
    """Render a single alert box."""
    return f'<div class="alert alert-{level}">{_escape(message)}</div>'


def _ensure_upload_root_exists() -> str | None:
    """Create the upload root if needed and return an error message on failure."""
    try:
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"Upload root is unavailable: {exc}"
    return None


def _check_upload_root_writable() -> str | None:
    """Verify that the upload root can be written to."""
    mkdir_error = _ensure_upload_root_exists()
    if mkdir_error is not None:
        return mkdir_error

    try:
        with tempfile.NamedTemporaryFile(dir=UPLOAD_ROOT, delete=True):
            return None
    except OSError as exc:
        return f"Upload root is not writable: {exc}"


async def _connect_temporal_client() -> Client:
    """Create a Temporal client using the shared data converter."""
    return await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )


async def _ensure_client() -> Client | None:
    """Return a connected Temporal client or keep the app in degraded mode."""
    client = getattr(app.state, "client", None)
    if client is not None:
        return client

    lock = getattr(app.state, "temporal_lock", None)
    if lock is None:
        return None

    async with lock:
        client = getattr(app.state, "client", None)
        if client is not None:
            return client

        try:
            client = await _connect_temporal_client()
        except (OSError, RuntimeError, TemporalError) as exc:
            app.state.temporal_error = str(exc)
            return None

        app.state.client = client
        app.state.temporal_error = None
        return client


def _current_temporal_warning() -> str | None:
    """Return the current degraded-mode Temporal warning."""
    return getattr(app.state, "temporal_error", None)


def _current_upload_warning() -> str | None:
    """Return any upload-root warning detected by the app."""
    return getattr(app.state, "upload_root_error", None)


def _scan_cases() -> list[CaseRecord]:
    """Scan the upload directory for cases not present in the manifest."""
    if not UPLOAD_ROOT.exists():
        return []

    records: list[CaseRecord] = []
    for path in sorted(UPLOAD_ROOT.iterdir()):
        if path.name == MANIFEST_PATH.name or not path.is_dir():
            continue

        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        image_count = sum(
            1
            for file_path in path.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS
        )
        records.append(
            CaseRecord(
                case_id=path.name,
                workflow_id=_workflow_id(path.name),
                status=CaseStatus.QUEUED,
                created_at=created_at,
                updated_at=created_at,
                image_dir=str(path),
                image_count=image_count,
            )
        )

    return records


def _read_manifest_records() -> tuple[list[CaseRecord], str | None]:
    """Load uploaded-case metadata from disk or fall back to scanning directories."""
    if not MANIFEST_PATH.exists():
        return [], None

    try:
        payload = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError as exc:
        message = "Case manifest is invalid; showing scanned case folders instead."
        return _scan_cases(), f"{message} {exc}"

    if not isinstance(payload, list):
        return _scan_cases(), "Case manifest is invalid; showing scanned case folders instead."

    try:
        return [CaseRecord.model_validate(item) for item in payload], None
    except ValidationError as exc:
        message = "Case manifest is invalid; showing scanned case folders instead."
        return _scan_cases(), f"{message} {exc}"


def _write_manifest_records(records: list[CaseRecord]) -> None:
    """Persist uploaded-case metadata to disk using an atomic replace."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.model_dump(mode="json") for record in _sort_case_records(records)]
    temp_path = MANIFEST_PATH.with_name(f"{MANIFEST_PATH.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2))
    temp_path.replace(MANIFEST_PATH)


async def _load_case_records() -> tuple[list[CaseRecord], str | None]:
    """Load case records under the app manifest lock."""
    async with app.state.manifest_lock:
        return _read_manifest_records()


async def _save_case_records(records: list[CaseRecord]) -> None:
    """Persist case records under the app manifest lock."""
    async with app.state.manifest_lock:
        _write_manifest_records(records)


def _sort_case_records(records: list[CaseRecord]) -> list[CaseRecord]:
    """Sort case records by creation time descending."""
    return sorted(records, key=lambda record: record.created_at, reverse=True)


def _record_map(records: list[CaseRecord]) -> dict[str, CaseRecord]:
    """Create a case-id keyed mapping of records."""
    return {record.case_id: record for record in records}


def _replace_case_record(records: list[CaseRecord], record: CaseRecord) -> list[CaseRecord]:
    """Replace or insert a case record in a record list."""
    mapping = _record_map(records)
    mapping[record.case_id] = record
    return list(mapping.values())


def _records_payload(records: list[CaseRecord]) -> list[dict]:
    """Serialize case records into plain dictionaries for comparisons."""
    return [record.model_dump(mode="json") for record in _sort_case_records(records)]


def _record_with_updates(record: CaseRecord, **changes: object) -> CaseRecord:
    """Return a case record with only the changed values updated."""
    changed = {
        key: value
        for key, value in changes.items()
        if getattr(record, key) != value
    }
    if not changed:
        return record
    changed.setdefault("updated_at", _utc_now_iso())
    return record.model_copy(update=changed)


def _random_case_suffix() -> str:
    """Return an uppercase random suffix for a case id."""
    return secrets.token_hex(4).upper()


def _generate_case_id(existing_case_ids: set[str]) -> str:
    """Generate a unique case id for new uploaded insurance claims."""
    prefix = datetime.now(UTC).strftime("CLM-%Y%m%d")
    for _ in range(16):
        candidate = f"{prefix}-{_random_case_suffix()}"
        if candidate not in existing_case_ids:
            return candidate
    message = "Unable to allocate a unique case id."
    raise RuntimeError(message)


def _validated_extension(filename: str) -> str:
    """Return a safe extension or raise if the file type is unsupported."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        message = f"Unsupported file type for {filename or 'unnamed upload'}."
        raise ValueError(message)
    return suffix


async def _prepare_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Validate uploaded files and return their content."""
    if not files:
        message = "No files uploaded."
        raise ValueError(message)
    if len(files) > _max_files():
        message = f"Upload at most {_max_files()} files per case."
        raise ValueError(message)

    prepared: list[tuple[str, bytes]] = []
    total_bytes = 0
    for upload in files:
        filename = (upload.filename or "").strip()
        extension = _validated_extension(filename)
        content = await upload.read()
        await upload.close()
        if not content:
            message = f"Uploaded file {filename or 'unnamed upload'} is empty."
            raise ValueError(message)
        if len(content) > _max_file_bytes():
            message = (
                f"Uploaded file {filename or 'unnamed upload'} exceeds "
                f"{_max_file_bytes()} bytes."
            )
            raise ValueError(message)
        total_bytes += len(content)
        if total_bytes > _max_total_bytes():
            message = f"Upload exceeds the total limit of {_max_total_bytes()} bytes."
            raise ValueError(message)
        prepared.append((extension, content))

    return prepared


def _redirect_home(
    message: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Redirect back to the home page with a success or error message."""
    query: list[str] = []
    if message:
        query.append(f"message={quote(message)}")
    if error:
        query.append(f"error={quote(error)}")
    suffix = f"?{'&'.join(query)}" if query else ""
    return RedirectResponse(url=f"/{suffix}", status_code=303)


def _redirect_case(
    case_id: str,
    message: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Redirect back to a case page with a success or error message."""
    query: list[str] = []
    if message:
        query.append(f"message={quote(message)}")
    if error:
        query.append(f"error={quote(error)}")
    suffix = f"?{'&'.join(query)}" if query else ""
    return RedirectResponse(url=f"/case/{case_id}{suffix}", status_code=303)


def _derive_case_status(
    temporal_status: str | None,
    packet: HumanReviewPacket | None,
) -> CaseStatus:
    """Map a Temporal execution status plus review packet into a UI case status."""
    if temporal_status == "COMPLETED":
        return CaseStatus.COMPLETED
    if temporal_status in FAILED_TEMPORAL_STATUSES:
        return CaseStatus.FAILED
    if temporal_status == "RUNNING":
        if packet and packet.decision_recommendation.decision == "CONDITIONAL":
            return CaseStatus.AWAITING_REVIEW
        return CaseStatus.RUNNING
    return CaseStatus.QUEUED


def _maybe_failure_text(exc: Exception) -> str:
    """Return a display-friendly exception string."""
    return str(exc) or exc.__class__.__name__


def _placeholder_record(case_id: str, created_at: str, image_dir: str = "") -> CaseRecord:
    """Build a record for a discovered workflow that was not in the manifest."""
    return CaseRecord(
        case_id=case_id,
        workflow_id=_workflow_id(case_id),
        status=CaseStatus.RUNNING,
        created_at=created_at,
        updated_at=created_at,
        image_dir=image_dir,
        image_count=0,
    )


async def _fetch_case_snapshot(record: CaseRecord, client: Client) -> CaseSnapshot:
    """Fetch the live Temporal state for a case record."""
    if record.status == CaseStatus.START_FAILED:
        return CaseSnapshot(record=record)

    handle = client.get_workflow_handle(
        record.workflow_id,
        result_type=ClaimAdjudicationOutput,
    )
    try:
        description = await handle.describe()
    except TemporalError as exc:
        if record.status == CaseStatus.QUEUED:
            updated_record = _record_with_updates(
                record,
                status=CaseStatus.START_FAILED,
                last_error=_maybe_failure_text(exc),
            )
            return CaseSnapshot(record=updated_record, lookup_error=_maybe_failure_text(exc))
        return CaseSnapshot(record=record, lookup_error=_maybe_failure_text(exc))

    temporal_status = (
        description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
        if description.status
        else None
    )

    packet = None
    if temporal_status == "RUNNING":
        try:
            packet = await handle.query(InsuranceClaimAdjudicationWorkflow.get_review_packet)
        except TemporalError:
            packet = None

    derived_status = _derive_case_status(temporal_status, packet)
    result = None
    failure_error = None

    if derived_status == CaseStatus.COMPLETED:
        try:
            result = await handle.result()
        except TemporalError as exc:
            failure_error = _maybe_failure_text(exc)
    elif derived_status == CaseStatus.FAILED:
        try:
            await handle.result()
        except TemporalError as exc:
            failure_error = _maybe_failure_text(exc)

    next_error = failure_error if derived_status == CaseStatus.FAILED else None
    updated_record = _record_with_updates(
        record,
        status=derived_status,
        last_error=next_error,
    )
    return CaseSnapshot(
        record=updated_record,
        packet=packet,
        result=result,
        failure_error=failure_error,
    )


async def _refresh_case_snapshots(
    records: list[CaseRecord],
    client: Client,
) -> list[CaseSnapshot]:
    """Refresh case records from Temporal using bounded concurrency."""
    semaphore = asyncio.Semaphore(MAX_STATUS_CONCURRENCY)

    async def refresh(record: CaseRecord) -> CaseSnapshot:
        async with semaphore:
            return await _fetch_case_snapshot(record, client)

    return await asyncio.gather(*(refresh(record) for record in records))


async def _case_record_from_sources(
    case_id: str,
    client: Client | None,
) -> tuple[CaseRecord | None, str | None]:
    """Load a case record from the manifest or create one from a live workflow."""
    records, manifest_warning = await _load_case_records()
    record = _record_map(records).get(case_id)
    if record is not None or client is None:
        return record, manifest_warning

    handle = client.get_workflow_handle(
        _workflow_id(case_id),
        result_type=ClaimAdjudicationOutput,
    )
    try:
        description = await handle.describe()
    except TemporalError:
        return None, manifest_warning

    created_at = (
        description.start_time.isoformat()
        if getattr(description, "start_time", None) is not None
        else _utc_now_iso()
    )
    record = _placeholder_record(case_id=case_id, created_at=created_at)
    updated_records = _replace_case_record(records, record)
    await _save_case_records(updated_records)
    return record, manifest_warning


async def _start_case_workflow(
    client: Client,
    record: CaseRecord,
    reuse_policy: WorkflowIDReusePolicy,
) -> None:
    """Start a workflow for a case record."""
    await client.start_workflow(
        InsuranceClaimAdjudicationWorkflow.run,
        ClaimAdjudicationInput(case_id=record.case_id, image_dir=record.image_dir),
        id=record.workflow_id,
        task_queue=TASK_QUEUE,
        id_reuse_policy=reuse_policy,
    )


@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    """Return process health regardless of Temporal connectivity."""
    return PlainTextResponse("ok")


@app.get("/readyz")
async def readyz() -> PlainTextResponse:
    """Return readiness for upload storage and Temporal connectivity."""
    readiness_errors: list[str] = []

    upload_error = _check_upload_root_writable()
    if upload_error is not None:
        readiness_errors.append(upload_error)

    client = await _ensure_client()
    if client is None:
        readiness_errors.append(_current_temporal_warning() or "Temporal is unavailable.")

    if readiness_errors:
        return PlainTextResponse("\n".join(readiness_errors), status_code=503)
    return PlainTextResponse("ok")


@app.get("/", response_class=HTMLResponse)
async def index(  # noqa: C901, PLR0912
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Render the insurance claims console home page."""
    records, manifest_warning = await _load_case_records()
    records_by_case = _record_map(records)
    live_lookup_warning: str | None = None

    client = await _ensure_client()
    if client is not None:
        try:
            async for workflow_info in client.list_workflows(
                query=RUNNING_WORKFLOW_QUERY,
                limit=50,
            ):
                workflow_id = workflow_info.id
                if not workflow_id.startswith("insurance-"):
                    continue
                case_id = workflow_id.replace("insurance-", "", 1)
                if case_id not in records_by_case:
                    created_at = (
                        workflow_info.start_time.isoformat()
                        if getattr(workflow_info, "start_time", None) is not None
                        else _utc_now_iso()
                    )
                    records_by_case[case_id] = _placeholder_record(case_id, created_at)
        except TemporalError as exc:
            live_lookup_warning = (
                "Unable to refresh the live workflow list: "
                f"{_maybe_failure_text(exc)}"
            )

        current_records = list(records_by_case.values())
        snapshots = await _refresh_case_snapshots(current_records, client)
        refreshed_records = [snapshot.record for snapshot in snapshots]
        if _records_payload(current_records) != _records_payload(refreshed_records):
            await _save_case_records(refreshed_records)
        records = refreshed_records
    else:
        records = list(records_by_case.values())

    rows = [
        (
            f"<tr>"
            f'<td><a href="/case/{_escape(record.case_id)}">{_escape(record.case_id)}</a></td>'
            f"<td>{_escape(record.created_at)}</td>"
            f"<td>{_escape(record.updated_at)}</td>"
            f"<td>{_render_status_badge(record.status)}</td>"
            f"<td>{_escape(record.image_count)}</td>"
            f"<td>{_escape(record.retry_count)}</td>"
            f"</tr>"
        )
        for record in _sort_case_records(records)
    ]

    alerts: list[str] = []
    if message:
        alerts.append(_render_alert("success", message))
    if error:
        alerts.append(_render_alert("error", error))
    if manifest_warning:
        alerts.append(_render_alert("warning", manifest_warning))
    if _current_upload_warning():
        alerts.append(_render_alert("warning", _current_upload_warning()))
    if _current_temporal_warning():
        alerts.append(
            _render_alert(
                "warning",
                (
                    "Temporal is unavailable. Live status updates, uploads, retries, "
                    f"and review submissions are disabled. {_current_temporal_warning()}"
                ),
            )
        )
    if live_lookup_warning:
        alerts.append(_render_alert("warning", live_lookup_warning))

    body = f"""
<h1>Insurance Claims Console</h1>
<p class="muted">Upload a new claim package or review the status of existing workflows.</p>
{"".join(alerts)}

<div class="card">
  <h2>Upload Claim Images</h2>
  <p class="muted">
    Limits: {_escape(_max_files())} files, {_escape(_max_file_bytes())} bytes per file,
    {_escape(_max_total_bytes())} bytes total.
  </p>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <label>Claim Document Images (PNG/JPG/JPEG)</label>
    <input type="file" name="files" multiple required />
    <button type="submit">Upload & Start Workflow</button>
  </form>
</div>

<div class="card">
  <h2>Recent Cases</h2>
  <table>
    <thead>
      <tr>
        <th>Case ID</th>
        <th>Created</th>
        <th>Updated</th>
        <th>Status</th>
        <th>Images</th>
        <th>Retries</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows) if rows else '<tr><td colspan="6" class="muted">No cases yet.</td></tr>'}
    </tbody>
  </table>
</div>
"""
    refresh_seconds = (
        10 if any(record.status in ACTIVE_CASE_STATUSES for record in records) else None
    )
    return _page("Insurance Claims Console", body, refresh_seconds=refresh_seconds)


@app.get("/case/", response_class=HTMLResponse)
async def case_redirect(case_id: str) -> HTMLResponse:
    """Support query-string case routing."""
    return await case_view(case_id)


@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_view(  # noqa: C901, PLR0912, PLR0915
    case_id: str,
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Render a specific case page with packet details and review actions."""
    client = await _ensure_client()
    record, manifest_warning = await _case_record_from_sources(case_id, client)
    if record is None:
        not_found_alerts = []
        if manifest_warning:
            not_found_alerts.append(_render_alert("warning", manifest_warning))
        if _current_temporal_warning():
            not_found_alerts.append(_render_alert("warning", _current_temporal_warning()))
        return _page(
            "Case Not Found",
            f"<h1>Case {_escape(case_id)}</h1>"
            f"{''.join(not_found_alerts)}"
            "<p>Unable to locate that case.</p>",
        )

    snapshot = CaseSnapshot(record=record)
    if client is not None:
        snapshot = await _fetch_case_snapshot(record, client)
        if snapshot.record.model_dump(mode="json") != record.model_dump(mode="json"):
            records, _ = await _load_case_records()
            await _save_case_records(_replace_case_record(records, snapshot.record))

    alert_blocks: list[str] = []
    if message:
        alert_blocks.append(_render_alert("success", message))
    if error:
        alert_blocks.append(_render_alert("error", error))
    if manifest_warning:
        alert_blocks.append(_render_alert("warning", manifest_warning))
    if _current_upload_warning():
        alert_blocks.append(_render_alert("warning", _current_upload_warning()))
    if _current_temporal_warning():
        alert_blocks.append(_render_alert("warning", _current_temporal_warning()))
    if snapshot.lookup_error:
        alert_blocks.append(_render_alert("warning", snapshot.lookup_error))

    final_decision_block = ""
    if snapshot.record.status == CaseStatus.COMPLETED:
        if snapshot.result is not None:
            final_decision_block = f"""
<div class="card">
  <h2>Final Outcome</h2>
  <p>{_render_status_badge(CaseStatus.COMPLETED)}</p>
  <p><strong>Final Decision:</strong> {_escape(snapshot.result.final_decision)}</p>
  <p><strong>Risk Score:</strong> {_escape(snapshot.result.risk_score)}</p>
  <p><strong>Human Review Required:</strong> {_escape(snapshot.result.human_review_required)}</p>
  <p><strong>Decision Memo (Preview):</strong></p>
  <pre>{_escape(snapshot.result.decision_memo[:800])}</pre>
</div>
"""
        else:
            final_decision_block = """
<div class="card">
  <h2>Final Outcome</h2>
  <p class="muted">The workflow completed, but the final result could not be loaded.</p>
</div>
"""

    failure_block = ""
    if snapshot.record.status in {CaseStatus.FAILED, CaseStatus.START_FAILED}:
        failure_text = snapshot.failure_error or snapshot.record.last_error or "Unknown failure."
        failure_block = f"""
<div class="card">
  <h2>Failure Details</h2>
  <pre>{_escape(failure_text)}</pre>
</div>
"""

    recommendation_block = """
<div class="card">
  <h2>Review Packet</h2>
  <p class="muted">No review packet available yet.</p>
</div>
"""
    details_block = ""
    allow_submit = (
        snapshot.record.status == CaseStatus.AWAITING_REVIEW and snapshot.packet is not None
    )

    if snapshot.packet is not None:
        claim = snapshot.packet.sanitized_claim
        metrics = snapshot.packet.metrics
        analyses = snapshot.packet.analyses
        recommendation = snapshot.packet.decision_recommendation
        recommendation_block = f"""
<div class="card">
  <h2>Recommendation</h2>
  <p>{_render_status_badge(snapshot.record.status)}</p>
  <p><strong>Risk Score:</strong> {_escape(recommendation.risk_score)}</p>
  <p><strong>Reason:</strong> {_escape(recommendation.human_review_reason or "N/A")}</p>
  <p><strong>Conditions:</strong></p>
  {_render_list(recommendation.conditions)}
  <p><strong>Memo:</strong></p>
  <pre>{_escape(recommendation.memo)}</pre>
</div>
"""

        details_block = f"""
<div class="card">
  <h2>Claim Summary (Sanitized)</h2>
  <div class="grid">
    <div>
      <p><strong>Name:</strong> {_escape(claim.name or "N/A")}</p>
      <p><strong>Policy Number:</strong> {_escape(claim.policy_number or "N/A")}</p>
      <p><strong>Email:</strong> {_escape(claim.email or "N/A")}</p>
      <p><strong>Phone:</strong> {_escape(claim.phone or "N/A")}</p>
      <p><strong>Address:</strong> {_escape(claim.address or "N/A")}</p>
    </div>
    <div>
      <p><strong>Policy Type:</strong> {_escape(claim.policy.policy_type)}</p>
      <p><strong>Coverage Confirmed:</strong> {_escape(claim.policy.coverage_confirmed)}</p>
      <p><strong>Policy Status:</strong> {_escape(claim.policy.policy_status)}</p>
      <p><strong>Premium Status:</strong> {_escape(claim.policy.premium_status)}</p>
      <p><strong>Prior Claims (3y):</strong> {_escape(claim.policy.prior_claims_3y)}</p>
    </div>
    <div>
      <p><strong>Claim Type:</strong> {_escape(claim.incident.claim_type)}</p>
      <p><strong>Date of Loss:</strong> {_escape(claim.incident.date_of_loss)}</p>
      <p><strong>Reported Date:</strong> {_escape(claim.incident.reported_date)}</p>
      <p><strong>Police Report Filed:</strong> {_escape(claim.incident.police_report_filed)}</p>
      <p><strong>Third Party Involved:</strong> {_escape(claim.parties.third_party_involved)}</p>
    </div>
    <div>
      <p><strong>Coverage Limit:</strong> {_format_money(claim.policy.coverage_limit)}</p>
      <p><strong>Deductible:</strong> {_format_money(claim.policy.deductible)}</p>
      <p><strong>Claimed Amount:</strong> {_format_money(claim.loss.claimed_amount)}</p>
      <p><strong>Estimated Damage:</strong> {_format_money(claim.loss.estimated_damage)}</p>
      <p><strong>Repair Status:</strong> {_escape(claim.loss.repair_status)}</p>
    </div>
  </div>
</div>

<div class="card">
  <h2>Metrics</h2>
  <div class="grid">
    <p><strong>Claimed/Coverage Ratio:</strong>
    {_format_percent(metrics.claimed_to_coverage_ratio)}</p>
    <p><strong>Estimate Gap Ratio:</strong> {_format_percent(metrics.estimate_gap_ratio)}</p>
    <p><strong>Reporting Lag:</strong> {_escape(metrics.reporting_lag_days)} days</p>
    <p><strong>Documentation Completeness:</strong>
    {_format_percent(metrics.documentation_completeness)}</p>
    <p><strong>Net Claim Exposure:</strong> {_format_money(metrics.net_claim_exposure)}</p>
  </div>
</div>

<div class="card">
  <h2>Analyst Findings</h2>
  <h3>Coverage</h3>
  <pre>{_escape(analyses.coverage)}</pre>
  <h3>Liability</h3>
  <pre>{_escape(analyses.liability)}</pre>
  <h3>Damages</h3>
  <pre>{_escape(analyses.damages)}</pre>
  <h3>Fraud</h3>
  <pre>{_escape(analyses.fraud)}</pre>
</div>

<div class="card">
  <h2>Critic Review</h2>
  <pre>{_escape(snapshot.packet.critic_review)}</pre>
</div>

<div class="card">
  <h2>Risk Flags</h2>
  {_render_list(snapshot.packet.risk_flags)}
</div>

<div class="card">
  <h2>Policy Violations</h2>
  {_render_list(snapshot.packet.policy_violations)}
</div>
"""

    submit_block = ""
    if allow_submit:
        submit_block = f"""
<div class="card">
  <h2>Submit Review</h2>
  <form method="post" action="/submit">
    <input type="hidden" name="case_id" value="{_escape(case_id)}" />
    <label>Reviewer</label>
    <input name="reviewer" placeholder="Senior Claims Examiner" required />

    <label>Decision</label>
    <select name="decision">
      <option value="APPROVED">APPROVED</option>
      <option value="REJECTED">REJECTED</option>
    </select>

    <label>Notes</label>
    <textarea name="notes" placeholder="Write your review notes..." required></textarea>

    <button type="submit">Submit Review</button>
  </form>
</div>
"""

    retry_block = ""
    if snapshot.record.status in RETRYABLE_CASE_STATUSES:
        if client is None:
            retry_block = """
<div class="card">
  <h2>Retry Workflow</h2>
  <p class="muted">Retry is unavailable while Temporal is disconnected.</p>
</div>
"""
        else:
            retry_block = f"""
<div class="card">
  <h2>Retry Workflow</h2>
  <form method="post" action="/case/{_escape(case_id)}/retry" class="inline-form">
    <button type="submit" class="secondary">Retry Workflow</button>
  </form>
</div>
"""

    display_name = f" - {_escape(snapshot.packet.display_name)}" if snapshot.packet else ""
    body = f"""
<h1>Case {_escape(case_id)}{display_name}</h1>
<p class="muted">Workflow ID: {_escape(snapshot.record.workflow_id)}</p>
{''.join(alert_blocks)}

<div class="card">
  <h2>Status</h2>
  <div class="actions">
    {_render_status_badge(snapshot.record.status)}
    <span class="muted">Created {_escape(snapshot.record.created_at)}</span>
    <span class="muted">Updated {_escape(snapshot.record.updated_at)}</span>
  </div>
  <div class="grid">
    <p><strong>Image Files:</strong> {_escape(snapshot.record.image_count)}</p>
    <p><strong>Retries:</strong> {_escape(snapshot.record.retry_count)}</p>
    <p><strong>Image Directory:</strong> {_escape(snapshot.record.image_dir or "N/A")}</p>
    <p><strong>Last Error:</strong> {_escape(snapshot.record.last_error or "None")}</p>
  </div>
</div>

{final_decision_block}
{failure_block}
{recommendation_block}
{details_block}
{submit_block}
{retry_block}
"""
    refresh_seconds = 10 if snapshot.record.status in ACTIVE_CASE_STATUSES else None
    return _page(f"Case {case_id}", body, refresh_seconds=refresh_seconds)


@app.post("/submit")
async def submit_review(  # noqa: PLR0911
    case_id: CaseIdForm,
    reviewer: ReviewerForm,
    decision: DecisionForm,
    notes: NotesForm,
) -> Response:
    """Signal the workflow with a human review decision."""
    client = await _ensure_client()
    if client is None:
        return _redirect_case(
            case_id.strip(),
            error="Temporal is unavailable; cannot submit a review.",
        )

    case_id = case_id.strip()
    reviewer = reviewer.strip()
    decision = decision.strip().upper()
    notes = notes.strip()

    if not reviewer:
        return _redirect_case(case_id, error="Reviewer is required.")
    if not notes:
        return _redirect_case(case_id, error="Review notes are required.")
    if decision not in {"APPROVED", "REJECTED"}:
        return _redirect_case(case_id, error="Invalid decision. Use APPROVED or REJECTED.")

    handle = client.get_workflow_handle(_workflow_id(case_id))
    try:
        description = await handle.describe()
        temporal_status = (
            description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
            if description.status
            else None
        )
        packet = await handle.query(InsuranceClaimAdjudicationWorkflow.get_review_packet)
    except TemporalError as exc:
        return _redirect_case(case_id, error=f"Unable to submit review: {_maybe_failure_text(exc)}")

    if _derive_case_status(temporal_status, packet) != CaseStatus.AWAITING_REVIEW:
        return _redirect_case(
            case_id,
            error="Cannot submit review: workflow is not awaiting human review.",
        )

    try:
        await handle.signal(
            InsuranceClaimAdjudicationWorkflow.submit_human_review,
            HumanReviewInput(reviewer=reviewer, decision=decision, notes=notes),
        )
    except TemporalError as exc:
        return _redirect_case(case_id, error=f"Unable to submit review: {_maybe_failure_text(exc)}")

    return _redirect_case(case_id, message=f"Submitted review for {case_id}.")


@app.post("/case/{case_id}/retry")
async def retry_case(case_id: str) -> Response:
    """Retry a case whose workflow failed before or during execution."""
    client = await _ensure_client()
    if client is None:
        return _redirect_case(case_id, error="Temporal is unavailable; retry is disabled.")

    records, _ = await _load_case_records()
    record = _record_map(records).get(case_id)
    if record is None:
        return _redirect_case(case_id, error="Case not found.")

    current_record = record
    if record.status != CaseStatus.START_FAILED:
        current_snapshot = await _fetch_case_snapshot(record, client)
        current_record = current_snapshot.record
        if current_record.model_dump(mode="json") != record.model_dump(mode="json"):
            records = _replace_case_record(records, current_record)
            await _save_case_records(records)

    if current_record.status not in RETRYABLE_CASE_STATUSES:
        return _redirect_case(
            case_id,
            error=f"Cannot retry a case in {current_record.status} state.",
        )

    if not Path(current_record.image_dir).exists():
        return _redirect_case(case_id, error="Cannot retry: case image directory is missing.")

    reuse_policy = (
        WorkflowIDReusePolicy.REJECT_DUPLICATE
        if current_record.status == CaseStatus.START_FAILED
        else WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY
    )

    queued_record = _record_with_updates(
        current_record,
        status=CaseStatus.QUEUED,
        retry_count=current_record.retry_count + 1,
        last_error=None,
    )
    records = _replace_case_record(records, queued_record)
    await _save_case_records(records)

    try:
        await _start_case_workflow(client, queued_record, reuse_policy)
    except (OSError, RuntimeError, TemporalError) as exc:
        failed_record = _record_with_updates(
            queued_record,
            status=CaseStatus.START_FAILED,
            last_error=_maybe_failure_text(exc),
        )
        records = _replace_case_record(records, failed_record)
        await _save_case_records(records)
        return _redirect_case(
            case_id,
            error="Retry failed before the workflow could start.",
        )

    running_record = _record_with_updates(
        queued_record,
        status=CaseStatus.RUNNING,
        last_error=None,
    )
    records = _replace_case_record(records, running_record)
    await _save_case_records(records)
    return _redirect_case(case_id, message=f"Retry started for {case_id}.")


@app.post("/upload")
async def upload_case(files: UploadFiles) -> Response:
    """Persist uploaded images and start the claim workflow."""
    try:
        prepared_uploads = await _prepare_uploads(files)
    except ValueError as exc:
        return _redirect_home(error=_maybe_failure_text(exc))

    client = await _ensure_client()
    if client is None:
        return _redirect_home(error="Temporal is unavailable; uploads are disabled.")

    records, _ = await _load_case_records()
    existing_case_ids = set(_record_map(records))
    existing_case_ids.update(path.name for path in UPLOAD_ROOT.iterdir() if path.is_dir())
    case_id = _generate_case_id(existing_case_ids)

    case_dir = UPLOAD_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        for index, (extension, content) in enumerate(prepared_uploads, start=1):
            target = case_dir / f"{case_id}_p{index}{extension}"
            target.write_bytes(content)
    except OSError as exc:
        return _redirect_home(
            error=f"Unable to save uploads for {case_id}: {_maybe_failure_text(exc)}",
        )

    timestamp = _utc_now_iso()
    queued_record = CaseRecord(
        case_id=case_id,
        workflow_id=_workflow_id(case_id),
        status=CaseStatus.QUEUED,
        created_at=timestamp,
        updated_at=timestamp,
        image_dir=str(case_dir),
        image_count=len(prepared_uploads),
    )
    records = _replace_case_record(records, queued_record)
    await _save_case_records(records)

    try:
        await _start_case_workflow(
            client,
            queued_record,
            WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )
    except (OSError, RuntimeError, TemporalError) as exc:
        failed_record = _record_with_updates(
            queued_record,
            status=CaseStatus.START_FAILED,
            last_error=_maybe_failure_text(exc),
        )
        records = _replace_case_record(records, failed_record)
        await _save_case_records(records)
        return _redirect_case(
            case_id,
            error="Upload saved, but the workflow could not be started.",
        )

    running_record = _record_with_updates(
        queued_record,
        status=CaseStatus.RUNNING,
        last_error=None,
    )
    records = _replace_case_record(records, running_record)
    await _save_case_records(records)
    return _redirect_case(case_id, message=f"Started workflow for {case_id}.")

"""FastAPI-based human review UI for insurance claim adjudication."""

from __future__ import annotations

import html
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import TemporalError

from .insurance_models import ClaimAdjudicationInput, ClaimAdjudicationOutput, HumanReviewInput
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

CaseIdForm = Annotated[str, Form(...)]
ReviewerForm = Annotated[str, Form(...)]
DecisionForm = Annotated[str, Form(...)]
NotesForm = Annotated[str, Form(...)]
UploadFiles = Annotated[list[UploadFile], File(...)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up the Temporal client for the review UI."""
    app.state.client = await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
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
      max-width: 960px;
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
    .card {{
      border: 1px solid var(--border);
      padding: 16px;
      border-radius: 12px;
      background: var(--card);
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
      margin-top: 16px;
    }}
    .muted {{ color: var(--muted); }}
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
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e0e7ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
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


async def _get_client() -> Client:
    """Return the shared Temporal client."""
    return app.state.client


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


def _load_manifest() -> list[dict]:
    """Load uploaded-case metadata from disk."""
    if not MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text())
        if isinstance(payload, list):
            return payload
    except json.JSONDecodeError:
        return []
    return []


def _write_manifest(entries: list[dict]) -> None:
    """Persist uploaded-case metadata to disk."""
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2))


def _scan_cases() -> list[dict]:
    """Scan the upload directory for cases not present in the manifest."""
    if not UPLOAD_ROOT.exists():
        return []
    entries: list[dict] = []
    for path in UPLOAD_ROOT.iterdir():
        if path.name == MANIFEST_PATH.name or not path.is_dir():
            continue
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        entries.append(
            {
                "case_id": path.name,
                "created_at": created_at,
                "image_dir": str(path),
            }
        )
    return entries


def _generate_case_id() -> str:
    """Generate a case id for new uploaded insurance claims."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d")
    suffix = datetime.now(UTC).strftime("%H%M%S")
    return f"CLM-{timestamp}-{suffix}"


def _safe_extension(filename: str) -> str:
    """Allow only known image extensions for uploaded files."""
    lower = filename.lower()
    if lower.endswith(".png"):
        return ".png"
    if lower.endswith(".jpg"):
        return ".jpg"
    if lower.endswith(".jpeg"):
        return ".jpeg"
    return ".png"


async def _fetch_status(client: Client, case_id: str) -> str:
    """Fetch the current workflow status for a case."""
    handle = client.get_workflow_handle(_workflow_id(case_id))
    try:
        description = await handle.describe()
        if description.status:
            return description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
    except TemporalError:
        return "NOT_FOUND"
    return "UNKNOWN"


def _redirect_with_message(
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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Render the insurance claims console home page."""
    client = await _get_client()
    entries = _load_manifest()
    if not entries:
        entries = _scan_cases()

    entries_by_case = {entry.get("case_id", ""): entry for entry in entries if entry.get("case_id")}

    try:
        async for workflow_info in client.list_workflows(
            query=RUNNING_WORKFLOW_QUERY,
            limit=50,
        ):
            workflow_id = workflow_info.id
            if not workflow_id.startswith("insurance-"):
                continue
            case_id = workflow_id.replace("insurance-", "", 1)
            if case_id not in entries_by_case:
                entries_by_case[case_id] = {
                    "case_id": case_id,
                    "created_at": workflow_info.start_time.isoformat(),
                    "image_dir": "",
                }
    except TemporalError:
        pass

    entries = sorted(
        entries_by_case.values(),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )

    rows = []
    for entry in entries:
        case_id = entry.get("case_id", "")
        status = await _fetch_status(client, case_id) if case_id else "UNKNOWN"
        rows.append(
            f"<tr>"
            f'<td><a href="/case/{_escape(case_id)}">{_escape(case_id)}</a></td>'
            f"<td>{_escape(entry.get('created_at', ''))}</td>"
            f'<td><span class="badge">{_escape(status)}</span></td>'
            f"</tr>"
        )

    body = f"""
<h1>Insurance Claims Console</h1>
<p class="muted">Upload a new claim package or review the status of existing workflows.</p>

<div class="card">
  <h2>Upload Claim Images</h2>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <label>Claim Document Images (PNG/JPG)</label>
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
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows) if rows else '<tr><td colspan="3" class="muted">No cases yet.</td></tr>'}
    </tbody>
  </table>
</div>
"""
    return _page("Insurance Claims Console", body, refresh_seconds=10)


@app.get("/case/", response_class=HTMLResponse)
async def case_redirect(case_id: str) -> HTMLResponse:
    """Support query-string case routing."""
    return await case_view(case_id)


@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_view(
    case_id: str,
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Render a specific case page with packet details and review actions."""
    client = await _get_client()
    handle = client.get_workflow_handle(
        _workflow_id(case_id),
        result_type=ClaimAdjudicationOutput,
    )
    status_label = "UNKNOWN"
    try:
        description = await handle.describe()
        if description.status:
            status_label = description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
    except TemporalError as exc:
        return _page(
            "Case Not Found",
            f"<h1>Case {case_id}</h1><p>Unable to load workflow: {_escape(exc)}</p>",
        )

    final_decision_block = ""
    if status_label == "COMPLETED":
        try:
            result = await handle.result()
            final_decision_block = f"""
<div class="card">
  <h2>Final Outcome</h2>
  <p><span class="badge">{_escape(result.final_decision)}</span></p>
  <p><strong>Risk Score:</strong> {_escape(result.risk_score)}</p>
  <p><strong>Human Review Required:</strong> {_escape(result.human_review_required)}</p>
  <p><strong>Decision Memo (Preview):</strong></p>
  <pre>{_escape(result.decision_memo[:800])}</pre>
</div>
"""
        except TemporalError as exc:
            final_decision_block = f"""
<div class="card">
  <h2>Final Outcome</h2>
  <p class="muted">Unable to load final result: {_escape(exc)}</p>
</div>
"""

    alert_block = ""
    if message:
        alert_block = f'<div class="alert alert-success">{_escape(message)}</div>'
    if error:
        alert_block = f'<div class="alert alert-error">{_escape(error)}</div>'

    try:
        packet = await handle.query(InsuranceClaimAdjudicationWorkflow.get_review_packet)
    except TemporalError:
        packet = None

    allow_submit = False
    recommendation_block = """
<div class="card">
  <h2>Review Packet</h2>
  <p class="muted">No review packet available yet.</p>
</div>
"""
    details_block = ""

    if packet is not None:
        claim = packet.sanitized_claim
        metrics = packet.metrics
        analyses = packet.analyses
        recommendation = packet.decision_recommendation
        allow_submit = status_label == "RUNNING" and recommendation.decision == "CONDITIONAL"

        recommendation_block = f"""
<div class="card">
  <h2>Recommendation</h2>
  <p><span class="badge">{_escape(recommendation.decision)}</span></p>
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
  <pre>{_escape(packet.critic_review)}</pre>
</div>

<div class="card">
  <h2>Risk Flags</h2>
  {_render_list(packet.risk_flags)}
</div>

<div class="card">
  <h2>Policy Violations</h2>
  {_render_list(packet.policy_violations)}
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

    display_name = f" - {_escape(packet.display_name)}" if packet else ""
    body = f"""
<h1>Case {_escape(case_id)}{display_name}</h1>
<p class="muted">Workflow ID: {_escape(_workflow_id(case_id))}</p>
{alert_block}
<div class="card">
  <h2>Status</h2>
  <p><span class="badge">{_escape(status_label)}</span></p>
</div>
{final_decision_block}
{recommendation_block}
{details_block}
{submit_block}
"""
    refresh_seconds = 10 if status_label == "RUNNING" else None
    return _page(f"Case {case_id}", body, refresh_seconds=refresh_seconds)


@app.post("/submit")
async def submit_review(
    case_id: CaseIdForm,
    reviewer: ReviewerForm,
    decision: DecisionForm,
    notes: NotesForm,
) -> Response:
    """Signal the workflow with a human review decision."""
    client = await _get_client()
    case_id = case_id.strip()
    reviewer = reviewer.strip()
    decision = decision.strip().upper()
    notes = notes.strip()

    if decision not in {"APPROVED", "REJECTED"}:
        return _redirect_with_message(case_id, error="Invalid decision. Use APPROVED or REJECTED.")

    handle = client.get_workflow_handle(_workflow_id(case_id))
    try:
        description = await handle.describe()
        status = (
            description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
            if description.status
            else "UNKNOWN"
        )
        if status != "RUNNING":
            return _redirect_with_message(
                case_id,
                error=f"Cannot submit review: workflow status is {status}.",
            )

        packet = await handle.query(InsuranceClaimAdjudicationWorkflow.get_review_packet)
        if packet is None:
            return _redirect_with_message(
                case_id,
                error="Cannot submit review: no review packet available yet.",
            )
        if packet.decision_recommendation.decision != "CONDITIONAL":
            return _redirect_with_message(
                case_id,
                error="Cannot submit review: workflow is not awaiting human review.",
            )

        await handle.signal(
            InsuranceClaimAdjudicationWorkflow.submit_human_review,
            HumanReviewInput(reviewer=reviewer, decision=decision, notes=notes),
        )
    except TemporalError as exc:
        return _redirect_with_message(case_id, error=f"Unable to submit review: {exc}")

    return _redirect_with_message(case_id, message=f"Submitted review for {case_id}.")


@app.post("/upload")
async def upload_case(files: UploadFiles) -> Response:
    """Persist uploaded images and start the claim workflow."""
    if not files:
        return PlainTextResponse("No files uploaded.", status_code=400)

    case_id = _generate_case_id()
    case_dir = UPLOAD_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    for index, upload in enumerate(files, start=1):
        ext = _safe_extension(upload.filename or "")
        filename = f"{case_id}_p{index}{ext}"
        target = case_dir / filename
        content = await upload.read()
        target.write_bytes(content)

    client = await _get_client()
    await client.start_workflow(
        InsuranceClaimAdjudicationWorkflow.run,
        ClaimAdjudicationInput(case_id=case_id, image_dir=str(case_dir)),
        id=_workflow_id(case_id),
        task_queue=TASK_QUEUE,
    )

    entries = _load_manifest()
    created_at = datetime.now(UTC).isoformat()
    entries.append(
        {
            "case_id": case_id,
            "created_at": created_at,
            "image_dir": str(case_dir),
        }
    )
    _write_manifest(entries)

    return RedirectResponse(url=f"/case/{case_id}", status_code=303)

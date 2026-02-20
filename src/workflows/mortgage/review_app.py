"""FastAPI-based human review UI for mortgage underwriting."""

from __future__ import annotations

import html
import os
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import TemporalError

from .mortgage_models import HumanReviewInput
from .mortgage_workflow import MortgageUnderwritingWorkflow

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")

app = FastAPI(title="Mortgage Human Review")


@app.on_event("startup")
async def _startup() -> None:
    app.state.client = await Client.connect(
        TEMPORAL_ADDRESS,
        data_converter=pydantic_data_converter,
    )


def _page(title: str, body: str) -> HTMLResponse:
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
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
    button.secondary {{
      background: #374151;
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
  </style>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""
    return HTMLResponse(html)


def _workflow_id(case_id: str) -> str:
    return f"mortgage-{case_id}"


async def _get_client() -> Client:
    return app.state.client


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _format_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_money(value: float) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _render_list(items: list[str]) -> str:
    if not items:
        return "<p class=\"muted\">None</p>"
    return "<ul>" + "".join([f"<li>{_escape(item)}</li>" for item in items]) + "</ul>"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    body = """
<h1>Mortgage Human Review</h1>
<p class="muted">Enter a case ID to load the review packet.</p>

<div class="card">
  <h2>Load Case</h2>
  <form method="get" action="/case/">
    <label>Case ID</label>
    <input name="case_id" placeholder="MTG-2025-001" />
    <button type="submit">Load</button>
  </form>
</div>
"""
    return _page("Human Review", body)


@app.get("/case/", response_class=HTMLResponse)
async def case_redirect(case_id: str) -> HTMLResponse:
    return await case_view(case_id)


@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_view(case_id: str) -> HTMLResponse:
    client = await _get_client()
    handle = client.get_workflow_handle(_workflow_id(case_id))
    status_label = "UNKNOWN"
    try:
        description = await handle.describe()
        if description.status:
            status_label = description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "")
    except TemporalError as exc:
        return _page(
            "Case Not Found",
            f"<h1>Case {case_id}</h1><p>Unable to load workflow: {exc}</p>",
        )
    try:
        packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
    except TemporalError as exc:
        return _page(
            "Case Not Found",
            f"<h1>Case {case_id}</h1><p>Unable to load workflow: {exc}</p>",
        )

    if packet is None:
        return _page(
            "Case Not Ready",
            f"<h1>Case {case_id}</h1><p>No review packet available yet.</p>",
        )

    applicant = packet.sanitized_applicant
    metrics = packet.metrics
    analyses = packet.analyses
    recommendation = packet.decision_recommendation
    allow_submit = status_label == "RUNNING" and recommendation.decision == "CONDITIONAL"

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

    body = f"""
<h1>Case {_escape(case_id)} — {_escape(packet.display_name)}</h1>
<p class="muted">Workflow ID: {_escape(_workflow_id(case_id))}</p>
<div class="card">
  <h2>Status</h2>
  <p><span class="badge">{_escape(status_label)}</span></p>
</div>
{final_decision_block}

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

<div class="card">
  <h2>Applicant Summary (Sanitized)</h2>
  <div class="grid">
    <div>
      <p><strong>Name:</strong> {_escape(applicant.name or "N/A")}</p>
      <p><strong>Email:</strong> {_escape(applicant.email or "N/A")}</p>
      <p><strong>Phone:</strong> {_escape(applicant.phone or "N/A")}</p>
      <p><strong>Address:</strong> {_escape(applicant.address or "N/A")}</p>
    </div>
    <div>
      <p><strong>Credit Score:</strong> {_escape(applicant.credit_score)}</p>
      <p><strong>Employer:</strong> {_escape(applicant.employment.employer)}</p>
      <p><strong>Position:</strong> {_escape(applicant.employment.position)}</p>
      <p><strong>Years Employed:</strong> {_escape(applicant.employment.years)}</p>
    </div>
    <div>
      <p><strong>Monthly Income:</strong> {_format_money(applicant.employment.monthly_income)}</p>
      <p><strong>Liquid Assets:</strong> {_format_money(applicant.assets.liquid_assets_total)}</p>
      <p><strong>Reserves (months):</strong> {_escape(applicant.assets.reserves_months)}</p>
      <p><strong>Loan Amount:</strong> {_format_money(applicant.loan.amount)}</p>
    </div>
  </div>
</div>

<div class="card">
  <h2>Metrics</h2>
  <div class="grid">
    <p><strong>DTI Ratio:</strong> {_format_percent(metrics.dti_ratio)}</p>
    <p><strong>LTV Ratio:</strong> {_format_percent(metrics.ltv_ratio)}</p>
    <p><strong>Monthly Debt:</strong> {_format_money(metrics.monthly_debt)}</p>
    <p><strong>Monthly Income:</strong> {_format_money(metrics.monthly_income)}</p>
  </div>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Analyst Findings</h2>
  <h3>Credit</h3>
  <pre>{_escape(analyses.credit)}</pre>
  <h3>Income</h3>
  <pre>{_escape(analyses.income)}</pre>
  <h3>Assets</h3>
  <pre>{_escape(analyses.assets)}</pre>
  <h3>Collateral</h3>
  <pre>{_escape(analyses.collateral)}</pre>
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

{"""
<div class="card">
  <h2>Submit Review</h2>
  <form method="post" action="/submit">
    <input type="hidden" name="case_id" value=\"""" + _escape(case_id) + """\" />
    <label>Reviewer</label>
    <input name="reviewer" placeholder="Senior Underwriter" required />

    <label>Decision</label>
    <select name="decision">
      <option value="APPROVED">APPROVED</option>
      <option value="CONDITIONAL">CONDITIONAL</option>
      <option value="REJECTED">REJECTED</option>
    </select>

    <label>Notes</label>
    <textarea name="notes" placeholder="Write your review notes..." required></textarea>

    <button type="submit">Submit Review</button>
  </form>
</div>
""" if allow_submit else ""}
"""
    return _page(f"Case {case_id}", body)


@app.post("/submit")
async def submit_review(
    case_id: str = Form(...),
    reviewer: str = Form(...),
    decision: str = Form(...),
    notes: str = Form(...),
) -> PlainTextResponse:
    client = await _get_client()
    case_id = case_id.strip()
    reviewer = reviewer.strip()
    decision = decision.strip().upper()
    notes = notes.strip()

    if decision not in {"APPROVED", "CONDITIONAL", "REJECTED"}:
        return PlainTextResponse(
            "Invalid decision. Use APPROVED, CONDITIONAL, or REJECTED.",
            status_code=400,
        )

    handle = client.get_workflow_handle(_workflow_id(case_id))

    try:
        description = await handle.describe()
        status = description.status.name.replace("WORKFLOW_EXECUTION_STATUS_", "") if description.status else "UNKNOWN"
        if status != "RUNNING":
            return PlainTextResponse(
                f"Cannot submit review: workflow status is {status}.",
                status_code=409,
            )

        packet = await handle.query(MortgageUnderwritingWorkflow.get_review_packet)
        if packet is None:
            return PlainTextResponse(
                "Cannot submit review: no review packet available yet.",
                status_code=409,
            )
        if packet.decision_recommendation.decision != "CONDITIONAL":
            return PlainTextResponse(
                "Cannot submit review: workflow is not awaiting human review.",
                status_code=409,
            )

        await handle.signal(
            MortgageUnderwritingWorkflow.submit_human_review,
            HumanReviewInput(reviewer=reviewer, decision=decision, notes=notes),
        )
    except TemporalError as exc:
        return PlainTextResponse(
            f"Unable to submit review: {exc}",
            status_code=400,
        )

    return PlainTextResponse(f"Submitted review for {case_id}.")

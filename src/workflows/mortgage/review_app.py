"""FastAPI-based human review UI for mortgage underwriting."""

from __future__ import annotations

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
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    label {{ display: block; margin-top: 12px; font-weight: bold; }}
    input, select, textarea {{ width: 420px; padding: 8px; }}
    textarea {{ height: 120px; }}
    .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  {body}
</body>
</html>
"""
    return HTMLResponse(html)


def _workflow_id(case_id: str) -> str:
    return f"mortgage-{case_id}"


async def _get_client() -> Client:
    return app.state.client


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    body = """
<h1>Mortgage Human Review</h1>
<p class="muted">Enter a case ID to load the review packet, or submit a decision directly.</p>

<div class="card">
  <h2>Load Case</h2>
  <form method="get" action="/case/">
    <label>Case ID</label>
    <input name="case_id" placeholder="MTG-2025-001" />
    <button type="submit">Load</button>
  </form>
</div>

<div class="card" style="margin-top: 24px;">
  <h2>Submit Review</h2>
  <form method="post" action="/submit">
    <label>Case ID</label>
    <input name="case_id" placeholder="MTG-2025-001" required />

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
"""
    return _page("Human Review", body)


@app.get("/case/", response_class=HTMLResponse)
async def case_redirect(case_id: str) -> HTMLResponse:
    return await case_view(case_id)


@app.get("/case/{case_id}", response_class=HTMLResponse)
async def case_view(case_id: str) -> HTMLResponse:
    client = await _get_client()
    handle = client.get_workflow_handle(_workflow_id(case_id))
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

    body = f"""
<h1>Case {case_id} — {packet.display_name}</h1>
<p class="muted">Workflow ID: {_workflow_id(case_id)}</p>

<div class="card">
  <h2>Recommendation</h2>
  <p><strong>Decision:</strong> {packet.decision_recommendation.decision}</p>
  <p><strong>Risk Score:</strong> {packet.decision_recommendation.risk_score}</p>
  <p><strong>Reason:</strong> {packet.decision_recommendation.human_review_reason or "N/A"}</p>
  <pre>{packet.decision_recommendation.memo}</pre>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Risk Flags</h2>
  <ul>
    {"".join([f"<li>{flag}</li>" for flag in packet.risk_flags])}
  </ul>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Policy Violations</h2>
  <ul>
    {"".join([f"<li>{flag}</li>" for flag in packet.policy_violations])}
  </ul>
</div>
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
    handle = client.get_workflow_handle(_workflow_id(case_id))

    await handle.signal(
        MortgageUnderwritingWorkflow.submit_human_review,
        HumanReviewInput(reviewer=reviewer, decision=decision, notes=notes),
    )

    return PlainTextResponse(f"Submitted review for {case_id}.")

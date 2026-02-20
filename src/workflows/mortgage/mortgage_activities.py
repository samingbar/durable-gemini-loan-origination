"""Temporal activities for the mortgage underwriting workflow."""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader
from temporalio import activity

from .mortgage_models import (
    AgentResult,
    AgentTask,
    CriticResult,
    CriticTask,
    DecisionRecommendation,
    DecisionResult,
    DecisionTask,
    SupervisorDecision,
    SupervisorTask,
)
from .mortgage_utils import compute_risk_score, determine_decision, parse_llm_json, score_chunk, tokenize

DEFAULT_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _load_policy_chunks() -> list[str]:
    policy_path = _repo_root() / "notebook-inspiration" / "underwriting_policies.pdf"
    reader = PdfReader(policy_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    raw_text = "\n".join(pages)

    paragraphs = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
    chunks: list[str] = []
    current = []
    current_len = 0
    max_chars = 1200

    for para in paragraphs:
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _retrieve_policies(query: str, top_k: int = 4) -> str:
    chunks = _load_policy_chunks()
    query_tokens = tokenize(query)
    scored = [(score_chunk(query_tokens, chunk), chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)
    best = [chunk for score, chunk in scored[:top_k] if score > 0]
    return "\n\n".join(best) if best else "No relevant policy text found."


@lru_cache(maxsize=1)
def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use Gemini.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


async def _generate_text(prompt: str) -> str:
    client = _gemini_client()
    response = await client.aio.models.generate_content(
        model=_model_name(),
        contents=prompt,
    )
    return response.text or ""


def _format_applicant(task: AgentTask | CriticTask | DecisionTask | SupervisorTask) -> str:
    return json.dumps(task.applicant.model_dump(by_alias=True), indent=2)


@activity.defn
async def retrieve_policy_context(query: str) -> str:
    """Retrieve relevant policy chunks from the underwriting PDF."""

    activity.logger.info("Retrieving policy context for query: %s", query)
    return await asyncio.to_thread(_retrieve_policies, query)


@activity.defn
async def run_agent_analysis(task: AgentTask) -> AgentResult:
    """Run a specialist agent analysis using Gemini."""

    activity.logger.info("Running %s agent analysis", task.agent_name)
    prompt = f"""
You are the {task.agent_name} analyst for a mortgage underwriting team.
Write a short, plain-English analysis (6-10 bullet points) using the policy text and applicant data.

Applicant (sanitized):
{_format_applicant(task)}

Key metrics:
- DTI ratio: {task.metrics.dti_ratio:.2%}
- LTV ratio: {task.metrics.ltv_ratio:.2%}

Relevant policy excerpts:
{task.policy_context}

Your response should include:
- Summary of key strengths and risks
- Any documentation questions
- A recommended stance for your specialty (approve/conditional/reject)
""".strip()

    analysis = await _generate_text(prompt)
    return AgentResult(analysis=analysis)


@activity.defn
async def run_supervisor(task: SupervisorTask) -> SupervisorDecision:
    """Route the workflow to the next agent using Gemini."""

    activity.logger.info("Running supervisor routing decision")
    prompt = f"""
You are the supervisor for a mortgage underwriting team.
Choose the next agent to run based on what is still missing.

Applicant (sanitized):
{_format_applicant(task)}

Key metrics:
- DTI ratio: {task.metrics.dti_ratio:.2%}
- LTV ratio: {task.metrics.ltv_ratio:.2%}

Completed agents: {task.completed_agents}
Remaining agents: {task.remaining_agents}
Risk flags so far: {json.dumps(task.risk_flags, indent=2)}

Relevant policy excerpts:
{task.policy_context}

Respond ONLY as JSON with:
{{
  "next_agent": "credit|income|assets|collateral|critic|decision",
  "rationale": "short reason"
}}
""".strip()

    response = await _generate_text(prompt)
    data = parse_llm_json(response)

    next_agent = str(data.get("next_agent", "")).lower()
    rationale = str(data.get("rationale", "")).strip()

    valid_agents = {"credit", "income", "assets", "collateral", "critic", "decision"}
    if next_agent not in valid_agents:
        next_agent = task.remaining_agents[0] if task.remaining_agents else "decision"
        rationale = rationale or "Fallback routing decision."

    return SupervisorDecision(next_agent=next_agent, rationale=rationale)


@activity.defn
async def run_critic_review(task: CriticTask) -> CriticResult:
    """Run a critic review to check for missing risks or inconsistencies."""

    activity.logger.info("Running critic review")
    prompt = f"""
You are a senior underwriting critic.
Review the specialist analyses for consistency and completeness.
Highlight any missing risk factors, conflicting statements, or policy concerns.

Applicant (sanitized):
{_format_applicant(task)}

Key metrics:
- DTI ratio: {task.metrics.dti_ratio:.2%}
- LTV ratio: {task.metrics.ltv_ratio:.2%}

Risk flags already detected:
{json.dumps(task.risk_flags, indent=2)}

Specialist analyses:
CREDIT:\n{task.analyses.credit}
INCOME:\n{task.analyses.income}
ASSETS:\n{task.analyses.assets}
COLLATERAL:\n{task.analyses.collateral}

Relevant policy excerpts:
{task.policy_context}

Respond with:
- 3-6 bullet points of issues or confirmations
- Any additional documentation needed
""".strip()

    review = await _generate_text(prompt)
    return CriticResult(review=review)


@activity.defn
async def run_decision_memo(task: DecisionTask) -> DecisionResult:
    """Draft a decision memo using Gemini."""

    activity.logger.info("Drafting decision memo")
    prompt = f"""
You are a senior underwriter writing a decision memo.
Summarize the applicant profile, key risks, and policy alignment.
Do NOT include any personal identifiers beyond [APPLICANT_NAME].

Applicant (sanitized):
{_format_applicant(task)}

Key metrics:
- DTI ratio: {task.metrics.dti_ratio:.2%}
- LTV ratio: {task.metrics.ltv_ratio:.2%}

Risk flags:
{json.dumps(task.risk_flags, indent=2)}

Specialist analyses:
CREDIT:\n{task.analyses.credit}
INCOME:\n{task.analyses.income}
ASSETS:\n{task.analyses.assets}
COLLATERAL:\n{task.analyses.collateral}

Relevant policy excerpts:
{task.policy_context}

Write your response ONLY as JSON:
{{
  "decision": "APPROVED|CONDITIONAL|REJECTED|HUMAN_REVIEW",
  "risk_score": 0-100,
  "memo": "8-12 bullet points",
  "conditions": ["list", "of", "conditions"],
  "human_review_reason": "only if decision is HUMAN_REVIEW"
}}
""".strip()

    raw_response = await _generate_text(prompt)
    data = parse_llm_json(raw_response)

    decision = str(data.get("decision", "")).upper()
    risk_score = data.get("risk_score")
    if isinstance(risk_score, str) and risk_score.isdigit():
        risk_score = int(risk_score)
    memo = data.get("memo", "")
    conditions = data.get("conditions", [])
    human_review_reason = data.get("human_review_reason")

    allowed = {"APPROVED", "CONDITIONAL", "REJECTED", "HUMAN_REVIEW"}
    if decision not in allowed or not isinstance(risk_score, int) or not memo:
        decision = determine_decision(task.applicant, task.metrics)
        risk_score = compute_risk_score(task.applicant, task.metrics)
        memo = "Fallback decision due to invalid LLM JSON response."
        conditions = []
        human_review_reason = (
            "Invalid LLM output; requires human confirmation."
            if decision == "HUMAN_REVIEW"
            else None
        )

    recommendation = DecisionRecommendation(
        decision=decision,
        risk_score=int(risk_score),
        memo=str(memo),
        conditions=[str(item) for item in conditions] if isinstance(conditions, list) else [],
        human_review_reason=str(human_review_reason) if human_review_reason else None,
    )
    return DecisionResult(recommendation=recommendation, raw_response=raw_response)

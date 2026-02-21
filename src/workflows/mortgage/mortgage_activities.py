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
    ApplicationOcrTask,
    AgentResult,
    AgentTask,
    CriticResult,
    CriticTask,
    DecisionRecommendation,
    DecisionResult,
    DecisionTask,
    MortgageApplication,
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
    policy_path = _repo_root() / "resources" / "underwriting_policies.pdf"
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


def _list_image_paths(image_dir: Path, case_id: str) -> list[Path]:
    if not image_dir.exists():
        return []

    case_images = sorted(image_dir.glob(f"{case_id}_p*.png"))
    if case_images:
        return case_images

    images: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def _normalize_ocr_payload(payload: dict, case_id: str) -> dict:
    data = dict(payload)

    applicant_info = data.pop("applicant_information", None)
    if isinstance(applicant_info, dict):
        for field in ("case_id", "name", "ssn", "email", "phone", "address", "credit_score"):
            if field not in data and field in applicant_info:
                data[field] = applicant_info[field]

    credit_info = data.pop("credit_information", None)
    if isinstance(credit_info, dict):
        if "credit_score" not in data and "credit_score" in credit_info:
            data["credit_score"] = credit_info["credit_score"]
        if "credit_history" not in data:
            if "credit_history" in credit_info:
                data["credit_history"] = credit_info["credit_history"]
            elif "history" in credit_info:
                data["credit_history"] = credit_info["history"]

    for source_key, target_key in (
        ("employment_information", "employment"),
        ("income_information", "employment"),
        ("loan_information", "loan"),
        ("loan_details", "loan"),
        ("property_information", "property"),
        ("collateral", "property"),
        ("asset_information", "assets"),
        ("assets_information", "assets"),
        ("debt_information", "debts"),
        ("liabilities", "debts"),
    ):
        section = data.pop(source_key, None)
        if isinstance(section, dict) and target_key not in data:
            data[target_key] = section

    if not data.get("case_id"):
        data["case_id"] = case_id

    credit_history = data.get("credit_history")
    if isinstance(credit_history, dict):
        credit_history.setdefault("late_payments_12mo", 0)
        credit_history.setdefault("late_payments_24mo", 0)
        credit_history.setdefault("collections", [])
        credit_history.setdefault("inquiries_6mo", 0)
        credit_history.setdefault("credit_notes", "")

    debts = data.get("debts")
    if isinstance(debts, dict) and "total_monthly_debt" not in debts:
        total = 0.0
        for key in ("car_loan", "student_loan", "credit_cards", "personal_loan"):
            value = debts.get(key, 0) or 0
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        debts["total_monthly_debt"] = total

    assets = data.get("assets")
    if isinstance(assets, dict):
        if "liquid_assets_total" not in assets:
            checking = assets.get("checking", 0) or 0
            savings = assets.get("savings", 0) or 0
            try:
                assets["liquid_assets_total"] = float(checking) + float(savings)
            except (TypeError, ValueError):
                assets["liquid_assets_total"] = 0
        recent_deposits = assets.get("recent_deposits")
        if isinstance(recent_deposits, list):
            for deposit in recent_deposits:
                if isinstance(deposit, dict) and "description" not in deposit:
                    deposit["description"] = "Not provided"

    employment = data.get("employment")
    if isinstance(employment, dict):
        income_details = employment.get("income_details")
        if isinstance(income_details, dict):
            income_details.setdefault("bonus_2023", 0)
            income_details.setdefault("bonus_2024", 0)
            income_details.setdefault("bonus_stable", False)
            income_details.setdefault("employer_confirmation", "")

    loan = data.get("loan")
    property_info = data.get("property")
    if isinstance(property_info, dict):
        if "type" not in property_info and "property_type" in property_info:
            property_info["type"] = property_info["property_type"]
    if isinstance(loan, dict):
        if "property_type" not in loan and isinstance(property_info, dict):
            prop_type = property_info.get("type") or property_info.get("property_type")
            if prop_type:
                loan["property_type"] = prop_type

    if "dti_ratio" not in data:
        monthly_debt = None
        monthly_income = None
        if isinstance(debts, dict):
            monthly_debt = debts.get("total_monthly_debt")
        if isinstance(employment, dict):
            monthly_income = employment.get("monthly_income")
        try:
            if monthly_debt is not None and monthly_income:
                data["dti_ratio"] = float(monthly_debt) / float(monthly_income)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return data


async def _ocr_application_from_images(image_paths: list[Path], case_id: str) -> str:
    client = _gemini_client()
    schema = json.dumps(MortgageApplication.model_json_schema(), indent=2)
    contents: list[types.Part | str] = [
        "Extract the mortgage application data from these images.\n"
        "Return JSON ONLY (no markdown) that matches the MortgageApplication schema exactly.\n"
        "Rules:\n"
        f"- Use case_id exactly as: {case_id}\n"
        "- Output a single JSON object with the exact field names.\n"
        "- Include ALL required fields even if blank.\n"
        "- Optional fields may be null.\n"
        "- If a numeric value is missing, use 0.\n"
        "- If a string value is missing, use an empty string.\n"
        "- If a list is missing, use an empty list.\n"
        "Schema:\n"
        f"{schema}",
    ]

    for path in image_paths:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))

    response = await client.aio.models.generate_content(
        model=_model_name(),
        contents=contents,
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
async def extract_application_from_images(task: ApplicationOcrTask) -> MortgageApplication:
    """Extract mortgage application data from a directory of scanned images."""

    image_dir = Path(task.image_dir)
    image_paths = _list_image_paths(image_dir, task.case_id)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir} for case {task.case_id}")

    raw_text = await _ocr_application_from_images(image_paths, task.case_id)
    data = parse_llm_json(raw_text)
    if not isinstance(data, dict):
        raise RuntimeError("OCR did not return a JSON object")

    normalized = _normalize_ocr_payload(data, task.case_id)
    return MortgageApplication.model_validate(normalized)


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

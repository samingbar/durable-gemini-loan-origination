"""Temporal activities for the insurance claims workflow."""

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

from .insurance_models import (
    AgentResult,
    AgentTask,
    ClaimOcrTask,
    CriticResult,
    CriticTask,
    DecisionRecommendation,
    DecisionResult,
    DecisionTask,
    InsuranceClaim,
)
from .insurance_utils import (
    compute_risk_score,
    determine_decision,
    parse_llm_json,
    score_chunk,
    tokenize,
)

DEFAULT_MODEL = "gemini-2.5-flash"


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    """Resolve the repository root once for resource lookups."""
    return Path(__file__).resolve().parents[3]


def _policy_path() -> Path:
    """Return the configured policy corpus path."""
    configured = os.environ.get("INSURANCE_POLICY_PATH")
    if configured:
        return Path(configured).expanduser()
    return _repo_root() / "resources" / "insurance_claim_policies.pdf"


@lru_cache(maxsize=1)
def _load_policy_chunks() -> list[str]:
    """Load and chunk policy text for simple lexical retrieval."""
    policy_path = _policy_path()
    if not policy_path.exists():
        message = f"Insurance policy corpus not found at {policy_path}"
        raise FileNotFoundError(message)

    if policy_path.suffix.lower() == ".pdf":
        reader = PdfReader(policy_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n".join(pages)
    else:
        raw_text = policy_path.read_text()

    paragraphs = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
    chunks: list[str] = []
    current: list[str] = []
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
    """Retrieve the top policy chunks using simple token overlap."""
    chunks = _load_policy_chunks()
    query_tokens = tokenize(query)
    scored = [(score_chunk(query_tokens, chunk), chunk) for chunk in chunks]
    scored.sort(key=lambda item: item[0], reverse=True)
    best = [chunk for score, chunk in scored[:top_k] if score > 0]
    return "\n\n".join(best) if best else "No relevant policy text found."


@lru_cache(maxsize=1)
def _gemini_client() -> genai.Client:
    """Build and cache a Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        message = "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use Gemini."
        raise RuntimeError(message)
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _model_name() -> str:
    """Return the configured Gemini model name."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


async def _generate_text(prompt: str) -> str:
    """Generate text from Gemini using a consistent helper."""
    client = _gemini_client()
    response = await client.aio.models.generate_content(
        model=_model_name(),
        contents=prompt,
    )
    return response.text or ""


def _list_image_paths(image_dir: Path, case_id: str) -> list[Path]:
    """Return claim image paths in deterministic order."""
    if not image_dir.exists():
        return []

    case_images = sorted(image_dir.glob(f"{case_id}_p*.png"))
    if case_images:
        return case_images

    images: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(image_dir.glob(pattern))
    return sorted(images)


def _normalize_ocr_payload(  # noqa: C901, PLR0912, PLR0915
    payload: dict, case_id: str
) -> dict:
    """Normalize OCR output variants into the InsuranceClaim schema."""
    data = dict(payload)

    claimant_info = data.pop("claimant_information", None)
    if isinstance(claimant_info, dict):
        for field in ("case_id", "name", "policy_number", "email", "phone", "address"):
            if field not in data and field in claimant_info:
                data[field] = claimant_info[field]

    contact_info = data.pop("contact_information", None)
    if isinstance(contact_info, dict):
        for field in ("email", "phone", "address"):
            if field not in data and field in contact_info:
                data[field] = contact_info[field]

    policy_info = data.pop("policy_information", None)
    if isinstance(policy_info, dict):
        if "policy" not in data:
            data["policy"] = policy_info
        if "policy_number" not in data and "policy_number" in policy_info:
            data["policy_number"] = policy_info["policy_number"]

    for source_key, target_key in (
        ("incident_information", "incident"),
        ("incident_details", "incident"),
        ("loss_information", "loss"),
        ("damage_information", "loss"),
        ("claim_loss", "loss"),
        ("documentation", "documents"),
        ("supporting_documents", "documents"),
        ("document_information", "documents"),
        ("parties_information", "parties"),
        ("party_information", "parties"),
    ):
        section = data.pop(source_key, None)
        if isinstance(section, dict) and target_key not in data:
            data[target_key] = section

    if not data.get("case_id"):
        data["case_id"] = case_id

    policy = data.get("policy")
    if isinstance(policy, dict):
        policy.setdefault("line_of_business", "")
        policy.setdefault("policy_type", "")
        policy.setdefault("coverage_confirmed", False)
        policy.setdefault("coverage_limit", 0)
        policy.setdefault("deductible", 0)
        policy.setdefault("premium_status", "")
        policy.setdefault("policy_status", "")
        policy.setdefault("years_insured", 0)
        policy.setdefault("prior_claims_3y", 0)
        policy.setdefault("exclusions_noted", [])

    incident = data.get("incident")
    if isinstance(incident, dict):
        incident.setdefault("date_of_loss", "")
        incident.setdefault("reported_date", "")
        incident.setdefault("claim_type", "")
        incident.setdefault("description", "")
        incident.setdefault("location", "")
        incident.setdefault("police_report_filed", False)
        incident.setdefault("weather_related", False)

    loss = data.get("loss")
    if isinstance(loss, dict):
        loss.setdefault("claimed_amount", 0)
        loss.setdefault("estimated_damage", 0)
        loss.setdefault("emergency_mitigation", 0)
        loss.setdefault("depreciation_applied", 0)
        loss.setdefault("salvage_value", 0)
        loss.setdefault("repair_status", "")
        loss.setdefault("damaged_items", [])
        loss.setdefault("loss_notes", "")

        damaged_items = loss.get("damaged_items")
        if isinstance(damaged_items, list):
            for item in damaged_items:
                if isinstance(item, dict):
                    item.setdefault("description", "")
                    item.setdefault("category", "")
                    item.setdefault("estimated_cost", 0)

    documents = data.get("documents")
    if isinstance(documents, dict):
        documents.setdefault("photos_received", False)
        documents.setdefault("repair_estimates_count", 0)
        documents.setdefault("receipts_count", 0)
        documents.setdefault("witness_statements_count", 0)
        documents.setdefault("proof_of_ownership", False)
        documents.setdefault("adjuster_notes", "")
        documents.setdefault("missing_documents", [])

    parties = data.get("parties")
    if not isinstance(parties, dict):
        parties = {}
        data["parties"] = parties

    if isinstance(incident, dict):
        if "third_party_involved" in incident and "third_party_involved" not in parties:
            parties["third_party_involved"] = incident["third_party_involved"]
        if "injuries_reported" in incident and "injuries_reported" not in parties:
            parties["injuries_reported"] = incident["injuries_reported"]

    parties.setdefault("third_party_involved", False)
    parties.setdefault("third_party_details", "")
    parties.setdefault("injuries_reported", False)
    parties.setdefault("claimant_statement", "")
    parties.setdefault("witness_summary", "")

    return data


async def _ocr_claim_from_images(image_paths: list[Path], case_id: str) -> str:
    """Extract the claim JSON from uploaded images with a single multimodal prompt."""
    client = _gemini_client()
    schema = json.dumps(InsuranceClaim.model_json_schema(), indent=2)
    contents: list[types.Part | str] = [
        "Extract the insurance claim data from these images.\n"
        "Return JSON ONLY (no markdown) that matches the InsuranceClaim schema exactly.\n"
        "Rules:\n"
        f"- Use case_id exactly as: {case_id}\n"
        "- Output a single JSON object with the exact field names.\n"
        "- Include ALL required fields even if blank.\n"
        "- Optional fields may be null.\n"
        "- If a numeric value is missing, use 0.\n"
        "- If a boolean value is missing, use false.\n"
        "- If a string value is missing, use an empty string.\n"
        "- If a list is missing, use an empty list.\n"
        "Schema:\n"
        f"{schema}",
    ]

    for path in image_paths:
        mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))

    response = await client.aio.models.generate_content(
        model=_model_name(),
        contents=contents,
    )
    return response.text or ""


def _format_claim(task: AgentTask | CriticTask | DecisionTask) -> str:
    """Format the claim as JSON to preserve structure in prompts."""
    return json.dumps(task.claim.model_dump(), indent=2)


@activity.defn
async def retrieve_policy_context(query: str) -> str:
    """Retrieve relevant claim policy chunks from the policy corpus."""
    activity.logger.info("Retrieving insurance policy context for query: %s", query)
    return await asyncio.to_thread(_retrieve_policies, query)


@activity.defn
async def extract_claim_from_images(task: ClaimOcrTask) -> InsuranceClaim:
    """Extract insurance claim data from a directory of scanned images."""
    image_dir = Path(task.image_dir)
    image_paths = _list_image_paths(image_dir, task.case_id)
    if not image_paths:
        message = f"No images found in {image_dir} for case {task.case_id}"
        raise RuntimeError(message)

    raw_text = await _ocr_claim_from_images(image_paths, task.case_id)
    data = parse_llm_json(raw_text)
    if not isinstance(data, dict):
        message = "OCR did not return a JSON object"
        raise TypeError(message)

    normalized = _normalize_ocr_payload(data, task.case_id)
    return InsuranceClaim.model_validate(normalized)


@activity.defn
async def run_agent_analysis(task: AgentTask) -> AgentResult:
    """Run an insurance specialist analysis using Gemini."""
    activity.logger.info("Running %s claim analysis", task.agent_name)
    prompt = f"""
You are the {task.agent_name} specialist on an insurance claims team.
Write a short, plain-English analysis (6-10 bullet points) using the policy text and claim data.

Claim (sanitized):
{_format_claim(task)}

Key metrics:
- Claimed-to-coverage ratio: {task.metrics.claimed_to_coverage_ratio:.2%}
- Estimate gap ratio: {task.metrics.estimate_gap_ratio:.2%}
- Reporting lag: {task.metrics.reporting_lag_days} days
- Documentation completeness: {task.metrics.documentation_completeness:.2%}
- Net claim exposure: ${task.metrics.net_claim_exposure:,.2f}

Relevant policy excerpts:
{task.policy_context}

Your response should include:
- Key strengths and risks for your specialty
- Any missing documentation or inconsistencies
- A recommended stance for your specialty (approve/conditional/reject)
""".strip()

    analysis = await _generate_text(prompt)
    return AgentResult(analysis=analysis)


@activity.defn
async def run_critic_review(task: CriticTask) -> CriticResult:
    """Run a critic review to look for missed risks or contradictions."""
    activity.logger.info("Running insurance claim critic review")
    prompt = f"""
You are a senior insurance claims critic.
Review the specialist analyses for consistency, missing risks, and policy alignment.

Claim (sanitized):
{_format_claim(task)}

Key metrics:
- Claimed-to-coverage ratio: {task.metrics.claimed_to_coverage_ratio:.2%}
- Estimate gap ratio: {task.metrics.estimate_gap_ratio:.2%}
- Reporting lag: {task.metrics.reporting_lag_days} days
- Documentation completeness: {task.metrics.documentation_completeness:.2%}
- Net claim exposure: ${task.metrics.net_claim_exposure:,.2f}

Risk flags already detected:
{json.dumps(task.risk_flags, indent=2)}

Specialist analyses:
COVERAGE:\n{task.analyses.coverage}
LIABILITY:\n{task.analyses.liability}
DAMAGES:\n{task.analyses.damages}
FRAUD:\n{task.analyses.fraud}

Relevant policy excerpts:
{task.policy_context}

Respond with:
- 3-6 bullet points of issues or confirmations
- Any additional documentation required before settlement
""".strip()

    review = await _generate_text(prompt)
    return CriticResult(review=review)


@activity.defn
async def run_decision_memo(task: DecisionTask) -> DecisionResult:
    """Draft a structured claim decision memo using Gemini."""
    activity.logger.info("Drafting insurance claim decision memo")
    prompt = f"""
You are a senior insurance adjuster writing a claim decision memo.
Summarize the claim profile, key risks, and policy alignment.
Do NOT include any personal identifiers beyond [CLAIMANT_NAME].

Claim (sanitized):
{_format_claim(task)}

Key metrics:
- Claimed-to-coverage ratio: {task.metrics.claimed_to_coverage_ratio:.2%}
- Estimate gap ratio: {task.metrics.estimate_gap_ratio:.2%}
- Reporting lag: {task.metrics.reporting_lag_days} days
- Documentation completeness: {task.metrics.documentation_completeness:.2%}
- Net claim exposure: ${task.metrics.net_claim_exposure:,.2f}

Risk flags:
{json.dumps(task.risk_flags, indent=2)}

Specialist analyses:
COVERAGE:\n{task.analyses.coverage}
LIABILITY:\n{task.analyses.liability}
DAMAGES:\n{task.analyses.damages}
FRAUD:\n{task.analyses.fraud}

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
        decision = determine_decision(task.claim, task.metrics)
        risk_score = compute_risk_score(task.claim, task.metrics)
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

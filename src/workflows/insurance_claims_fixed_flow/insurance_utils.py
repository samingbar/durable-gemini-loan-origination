"""Utility functions for insurance claims adjudication."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from .insurance_models import ClaimMetrics, InsuranceClaim

if TYPE_CHECKING:
    from collections.abc import Iterable

PROTECTED_TERMS = [
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "religion",
    "marital",
    "disability",
    "nationality",
    "citizenship",
    "pregnant",
    "pregnancy",
]

ACTIVE_POLICY_STATUSES = {"ACTIVE", "IN_FORCE"}
CURRENT_PREMIUM_STATUSES = {"CURRENT", "PAID", "UP_TO_DATE", "GOOD_STANDING"}
REPORTABLE_CLAIM_TYPES = {"THEFT", "COLLISION", "VANDALISM"}

MASK_SUFFIX_LENGTH = 4
MIN_NAME_PARTS = 2
OWNERSHIP_RECEIPT_THRESHOLD = 2500.0
REVIEW_REPORTING_LAG_DAYS = 30
HIGH_REPORTING_LAG_DAYS = 60
REVIEW_DOCUMENT_COMPLETENESS = 0.75
LOW_DOCUMENT_COMPLETENESS = 0.50
REVIEW_ESTIMATE_GAP_RATIO = 0.25
HIGH_ESTIMATE_GAP_RATIO = 0.40
REVIEW_PRIOR_CLAIMS = 2
HIGH_PRIOR_CLAIMS = 4
REVIEW_CLAIM_TO_COVERAGE_RATIO = 0.80
CLAIM_TO_COVERAGE_HARD_STOP = 1.0


def _normalize_status(value: str) -> str:
    """Normalize free-text status values into a stable token."""
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _parse_date(value: str) -> date | None:
    """Parse a few common date formats used in OCR output."""
    cleaned = value.strip()
    if not cleaned:
        return None

    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC).date()
        except ValueError:
            continue

    return None


def _is_policy_active(status: str) -> bool:
    """Return whether the policy status is considered active."""
    return _normalize_status(status) in ACTIVE_POLICY_STATUSES


def _is_premium_current(status: str) -> bool:
    """Return whether premium status is current enough to honor a claim."""
    return _normalize_status(status) in CURRENT_PREMIUM_STATUSES


def _requires_police_report(claim_type: str) -> bool:
    """Return whether a claim type should include a police report."""
    return _normalize_status(claim_type) in REPORTABLE_CLAIM_TYPES


def calculate_claimed_to_coverage_ratio(claimed_amount: float, coverage_limit: float) -> float:
    """Calculate the claimed amount as a fraction of the coverage limit."""
    if coverage_limit <= 0:
        return 0.0
    return claimed_amount / coverage_limit


def calculate_estimate_gap_ratio(claimed_amount: float, estimated_damage: float) -> float:
    """Calculate the relative gap between the claim and the damage estimate."""
    denominator = max(claimed_amount, estimated_damage, 1)
    return abs(claimed_amount - estimated_damage) / denominator


def calculate_reporting_lag_days(date_of_loss: str, reported_date: str) -> int:
    """Calculate how many days elapsed between loss and report dates."""
    loss_date = _parse_date(date_of_loss)
    report_date = _parse_date(reported_date)
    if loss_date is None or report_date is None:
        return 0

    return max(0, (report_date - loss_date).days)


def compute_documentation_completeness(claim: InsuranceClaim) -> float:
    """Compute a simple 0-1 completeness score for claim documentation."""
    checklist = [
        claim.documents.photos_received,
        claim.documents.repair_estimates_count > 0 or claim.loss.claimed_amount <= 0,
        claim.documents.receipts_count > 0
        or claim.documents.proof_of_ownership
        or claim.loss.claimed_amount < OWNERSHIP_RECEIPT_THRESHOLD,
    ]

    if _requires_police_report(claim.incident.claim_type):
        checklist.append(claim.incident.police_report_filed)

    if claim.parties.third_party_involved or claim.parties.injuries_reported:
        checklist.append(claim.documents.witness_statements_count > 0)

    if not checklist:
        return 1.0

    return sum(1 for item in checklist if item) / len(checklist)


def sanitize_pii(claim: InsuranceClaim) -> InsuranceClaim:
    """Remove or mask PII for safe LLM processing."""
    sanitized = claim.model_copy(deep=True)

    if sanitized.name:
        sanitized.name = "[CLAIMANT_NAME]"

    if sanitized.address:
        sanitized.address = "[ADDRESS]"

    if sanitized.email:
        sanitized.email = "[EMAIL]"

    if sanitized.phone:
        phone = re.sub(r"\D", "", sanitized.phone)
        sanitized.phone = (
            f"***-***-{phone[-MASK_SUFFIX_LENGTH:]}"
            if len(phone) >= MASK_SUFFIX_LENGTH
            else "***-***-XXXX"
        )

    if sanitized.policy_number:
        digits = re.sub(r"\W", "", sanitized.policy_number)
        suffix = digits[-MASK_SUFFIX_LENGTH:] if len(digits) >= MASK_SUFFIX_LENGTH else "XXXX"
        sanitized.policy_number = f"***{suffix}"

    return sanitized


def detect_bias_signals(analysis: str) -> list[str]:
    """Detect potentially biased or protected-class references in free text."""
    lowered = analysis.lower()
    return [
        f"Potential bias reference detected: {term}" for term in PROTECTED_TERMS if term in lowered
    ]


def compute_metrics(claim: InsuranceClaim) -> ClaimMetrics:
    """Compute key insurance-claim ratios and indicators."""
    return ClaimMetrics(
        claimed_to_coverage_ratio=calculate_claimed_to_coverage_ratio(
            claim.loss.claimed_amount,
            claim.policy.coverage_limit,
        ),
        estimate_gap_ratio=calculate_estimate_gap_ratio(
            claim.loss.claimed_amount,
            claim.loss.estimated_damage,
        ),
        reporting_lag_days=calculate_reporting_lag_days(
            claim.incident.date_of_loss,
            claim.incident.reported_date,
        ),
        documentation_completeness=compute_documentation_completeness(claim),
        net_claim_exposure=max(0.0, claim.loss.claimed_amount - claim.policy.deductible),
    )


def derive_risk_flags(  # noqa: C901, PLR0912
    claim: InsuranceClaim, metrics: ClaimMetrics
) -> list[str]:
    """Generate a deterministic set of claim risk flags."""
    flags: list[str] = []

    if not _is_policy_active(claim.policy.policy_status):
        flags.append("Policy is not active")
    if not _is_premium_current(claim.policy.premium_status):
        flags.append("Premium status is not current")
    if not claim.policy.coverage_confirmed:
        flags.append("Coverage for this loss is not confirmed")
    if claim.loss.claimed_amount > claim.policy.coverage_limit:
        flags.append("Claimed amount exceeds coverage limit")
    if claim.loss.claimed_amount <= claim.policy.deductible:
        flags.append("Claimed amount does not exceed deductible")

    if metrics.reporting_lag_days > REVIEW_REPORTING_LAG_DAYS:
        flags.append("Claim reported more than 30 days after the loss")

    if metrics.documentation_completeness < REVIEW_DOCUMENT_COMPLETENESS:
        flags.append("Documentation package is incomplete")
    if claim.documents.missing_documents:
        flags.append(
            f"Missing supporting documents: {', '.join(claim.documents.missing_documents)}"
        )

    if metrics.estimate_gap_ratio > REVIEW_ESTIMATE_GAP_RATIO:
        flags.append("Claimed amount materially exceeds the available damage estimate")

    if claim.policy.prior_claims_3y >= REVIEW_PRIOR_CLAIMS:
        flags.append("Multiple claims filed within the last 3 years")

    if claim.parties.third_party_involved:
        flags.append("Third-party involvement increases liability complexity")

    if claim.parties.injuries_reported:
        flags.append("Injury reported on the claim")

    if (
        _requires_police_report(claim.incident.claim_type)
        and not claim.incident.police_report_filed
    ):
        flags.append("Police report missing for a reportable loss type")

    if metrics.claimed_to_coverage_ratio > REVIEW_CLAIM_TO_COVERAGE_RATIO:
        flags.append("Claim consumes more than 80% of the available coverage")

    return flags


def determine_decision(  # noqa: C901, PLR0911
    claim: InsuranceClaim, metrics: ClaimMetrics
) -> str:
    """Apply deterministic thresholds to determine the claim decision."""
    if not _is_policy_active(claim.policy.policy_status):
        return "REJECTED"
    if not _is_premium_current(claim.policy.premium_status):
        return "REJECTED"
    if not claim.policy.coverage_confirmed:
        return "REJECTED"
    if claim.loss.claimed_amount > claim.policy.coverage_limit:
        return "REJECTED"
    if claim.loss.claimed_amount <= claim.policy.deductible:
        return "REJECTED"

    if metrics.reporting_lag_days > REVIEW_REPORTING_LAG_DAYS:
        return "CONDITIONAL"
    if metrics.documentation_completeness < REVIEW_DOCUMENT_COMPLETENESS:
        return "CONDITIONAL"
    if metrics.estimate_gap_ratio > REVIEW_ESTIMATE_GAP_RATIO:
        return "CONDITIONAL"
    if claim.policy.prior_claims_3y >= REVIEW_PRIOR_CLAIMS:
        return "CONDITIONAL"
    if claim.parties.third_party_involved:
        return "CONDITIONAL"
    if claim.parties.injuries_reported:
        return "CONDITIONAL"
    if metrics.claimed_to_coverage_ratio > REVIEW_CLAIM_TO_COVERAGE_RATIO:
        return "CONDITIONAL"

    return "APPROVED"


def hard_stop_violations(claim: InsuranceClaim, metrics: ClaimMetrics) -> list[str]:
    """Return policy violations that should force human review or denial."""
    violations: list[str] = []

    if not _is_policy_active(claim.policy.policy_status):
        violations.append("Policy is not active (hard stop)")
    if not _is_premium_current(claim.policy.premium_status):
        violations.append("Premium status is not current (hard stop)")
    if not claim.policy.coverage_confirmed:
        violations.append("Coverage for this loss is not confirmed (hard stop)")
    if claim.loss.claimed_amount > claim.policy.coverage_limit:
        violations.append("Claimed amount exceeds coverage limit (hard stop)")
    if claim.loss.claimed_amount <= claim.policy.deductible:
        violations.append("Claimed amount does not exceed deductible (hard stop)")
    if metrics.claimed_to_coverage_ratio > CLAIM_TO_COVERAGE_HARD_STOP:
        violations.append("Claim-to-coverage ratio exceeds 100% (hard stop)")

    return violations


def compute_risk_score(  # noqa: C901, PLR0912
    claim: InsuranceClaim, metrics: ClaimMetrics
) -> int:
    """Compute a simple 0-100 risk score where higher is safer."""
    score = 100

    if not _is_policy_active(claim.policy.policy_status):
        score -= 35
    if not _is_premium_current(claim.policy.premium_status):
        score -= 25
    if not claim.policy.coverage_confirmed:
        score -= 35

    if claim.loss.claimed_amount > claim.policy.coverage_limit:
        score -= 30
    elif metrics.claimed_to_coverage_ratio > REVIEW_CLAIM_TO_COVERAGE_RATIO:
        score -= 10

    if claim.loss.claimed_amount <= claim.policy.deductible:
        score -= 15

    if metrics.reporting_lag_days > HIGH_REPORTING_LAG_DAYS:
        score -= 20
    elif metrics.reporting_lag_days > REVIEW_REPORTING_LAG_DAYS:
        score -= 10

    if metrics.documentation_completeness < LOW_DOCUMENT_COMPLETENESS:
        score -= 20
    elif metrics.documentation_completeness < REVIEW_DOCUMENT_COMPLETENESS:
        score -= 10

    if metrics.estimate_gap_ratio > HIGH_ESTIMATE_GAP_RATIO:
        score -= 20
    elif metrics.estimate_gap_ratio > REVIEW_ESTIMATE_GAP_RATIO:
        score -= 10

    if claim.policy.prior_claims_3y >= HIGH_PRIOR_CLAIMS:
        score -= 15
    elif claim.policy.prior_claims_3y >= REVIEW_PRIOR_CLAIMS:
        score -= 8

    if claim.parties.third_party_involved:
        score -= 8

    if claim.parties.injuries_reported:
        score -= 10

    if (
        _requires_police_report(claim.incident.claim_type)
        and not claim.incident.police_report_filed
    ):
        score -= 10

    return max(0, min(100, score))


def format_display_name(claim: InsuranceClaim) -> str:
    """Format the claimant name as first and last initials."""
    if not claim.name:
        return "[CLAIMANT]"

    parts = [part for part in claim.name.strip().split() if part]
    if len(parts) < MIN_NAME_PARTS:
        return "[CLAIMANT]"

    first_initial = parts[0][0].upper()
    last_initial = parts[-1][0].upper()
    return f"{first_initial}. {last_initial}."


def parse_llm_json(text: str) -> dict:
    """Parse a JSON object from an LLM response with a safe fallback."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}

    return {}


def tokenize(text: str) -> list[str]:
    """Tokenize a string into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def score_chunk(query_tokens: Iterable[str], chunk: str) -> int:
    """Score a chunk using simple token overlap."""
    chunk_tokens = set(tokenize(chunk))
    return sum(1 for token in query_tokens if token in chunk_tokens)

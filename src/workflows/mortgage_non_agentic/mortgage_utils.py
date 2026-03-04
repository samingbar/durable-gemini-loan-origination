"""Utility functions for mortgage underwriting."""

from __future__ import annotations

import json
import re
from typing import Iterable

from .mortgage_models import MortgageApplication, UnderwritingMetrics

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


def calculate_dti_ratio(monthly_debt: float, monthly_income: float) -> float:
    """Calculate Debt-to-Income ratio as a decimal (e.g., 0.43)."""

    if monthly_income <= 0:
        raise ValueError("Monthly income must be greater than 0")
    return monthly_debt / monthly_income


def calculate_ltv_ratio(loan_amount: float, property_value: float) -> float:
    """Calculate Loan-to-Value ratio as a decimal (e.g., 0.80)."""

    if property_value <= 0:
        raise ValueError("Property value must be greater than 0")
    return loan_amount / property_value


def sanitize_pii(applicant: MortgageApplication) -> MortgageApplication:
    """Remove or mask PII for safe LLM processing."""

    sanitized = applicant.model_copy(deep=True)

    if sanitized.ssn:
        ssn = sanitized.ssn
        sanitized.ssn = f"***-**-{ssn[-4:]}" if len(ssn) >= 4 else "***-**-XXXX"

    if sanitized.name:
        sanitized.name = "[APPLICANT_NAME]"

    if sanitized.address:
        sanitized.address = "[ADDRESS]"

    if sanitized.phone:
        phone = re.sub(r"\D", "", sanitized.phone)
        sanitized.phone = f"***-***-{phone[-4:]}" if len(phone) >= 4 else "***-***-XXXX"

    if sanitized.email:
        sanitized.email = "[EMAIL]"

    return sanitized


def detect_bias_signals(analysis: str) -> list[str]:
    """Detect potential fair-lending bias signals in free text."""

    found = []
    lowered = analysis.lower()
    for term in PROTECTED_TERMS:
        if term in lowered:
            found.append(f"Potential bias reference detected: {term}")
    return found


def compute_metrics(applicant: MortgageApplication) -> UnderwritingMetrics:
    """Compute key underwriting ratios."""

    monthly_debt = applicant.debts.total_monthly_debt
    monthly_income = applicant.employment.monthly_income
    dti_ratio = calculate_dti_ratio(monthly_debt, monthly_income)
    ltv_ratio = calculate_ltv_ratio(applicant.loan.amount, applicant.property.appraised_value)
    return UnderwritingMetrics(
        dti_ratio=dti_ratio,
        ltv_ratio=ltv_ratio,
        monthly_debt=monthly_debt,
        monthly_income=monthly_income,
    )


def derive_risk_flags(applicant: MortgageApplication, metrics: UnderwritingMetrics) -> list[str]:
    """Generate a simple list of risk flags for the application."""

    flags: list[str] = []

    if applicant.credit_score < 620:
        flags.append("Credit score below minimum (620)")
    if applicant.credit_history.bankruptcies > 0:
        flags.append("Bankruptcy history on record")
    if applicant.credit_history.late_payments_12mo > 0:
        flags.append("Recent late payments in last 12 months")
    if applicant.credit_history.collections:
        flags.append("Collections present on credit report")

    if metrics.dti_ratio > 0.50:
        flags.append("DTI above 50% (very high)")
    elif metrics.dti_ratio > 0.43:
        flags.append("DTI above 43% (high)")

    if metrics.ltv_ratio > 0.95:
        flags.append("LTV above 95% (very high)")
    elif metrics.ltv_ratio > 0.90:
        flags.append("LTV above 90% (high)")

    if applicant.assets.reserves_months <= 0:
        flags.append("No verified reserves")
    elif applicant.assets.reserves_months < 2:
        flags.append("Reserves below 2 months")
    elif applicant.assets.reserves_months < 6:
        flags.append("Reserves below 6 months")

    if applicant.employment.employment_gap.lower() == "yes":
        flags.append("Employment gap reported")

    if applicant.property.required_repairs > 0:
        flags.append("Property requires repairs")

    for deposit in applicant.assets.recent_deposits:
        if "unknown" in deposit.description.lower() or "unexplained" in deposit.description.lower():
            flags.append("Unexplained large deposit")
            break

    return flags


def determine_decision(applicant: MortgageApplication, metrics: UnderwritingMetrics) -> str:
    """Apply simple policy thresholds to determine a decision."""

    if applicant.credit_score < 620:
        return "REJECTED"
    if metrics.dti_ratio > 0.50:
        return "REJECTED"
    if applicant.credit_history.bankruptcies > 0 and applicant.credit_score < 660:
        return "REJECTED"
    if applicant.assets.reserves_months <= 0:
        return "REJECTED"

    if metrics.dti_ratio > 0.43:
        return "CONDITIONAL"
    if metrics.ltv_ratio > 0.90:
        return "CONDITIONAL"
    if applicant.employment.employment_gap.lower() == "yes":
        return "CONDITIONAL"
    if applicant.property.required_repairs > 0:
        return "CONDITIONAL"
    if applicant.assets.reserves_months < 6:
        return "CONDITIONAL"

    return "APPROVED"


def hard_stop_violations(applicant: MortgageApplication, metrics: UnderwritingMetrics) -> list[str]:
    """Return policy violations that require human review."""

    violations: list[str] = []

    if applicant.credit_score < 620:
        violations.append("Credit score below 620 (hard stop)")
    if metrics.dti_ratio > 0.50:
        violations.append("DTI above 50% (hard stop)")
    if applicant.credit_history.bankruptcies > 0 and applicant.credit_score < 660:
        violations.append("Bankruptcy with sub-660 credit score (hard stop)")
    if applicant.assets.reserves_months <= 0:
        violations.append("No verified reserves (hard stop)")

    return violations


def compute_risk_score(applicant: MortgageApplication, metrics: UnderwritingMetrics) -> int:
    """Compute a simple 0-100 risk score (higher is safer)."""

    score = 100

    if applicant.credit_score < 620:
        score -= 40
    elif applicant.credit_score < 680:
        score -= 20
    elif applicant.credit_score < 720:
        score -= 10

    if metrics.dti_ratio > 0.50:
        score -= 30
    elif metrics.dti_ratio > 0.43:
        score -= 15

    if metrics.ltv_ratio > 0.95:
        score -= 20
    elif metrics.ltv_ratio > 0.90:
        score -= 10

    if applicant.assets.reserves_months <= 0:
        score -= 15
    elif applicant.assets.reserves_months < 2:
        score -= 10
    elif applicant.assets.reserves_months < 6:
        score -= 5

    if applicant.credit_history.bankruptcies > 0:
        score -= 25
    if applicant.credit_history.collections:
        score -= 10

    return max(0, min(100, score))


def format_display_name(applicant: MortgageApplication) -> str:
    """Format display name as first and last initials."""

    if not applicant.name:
        return "[APPLICANT]"

    parts = [part for part in applicant.name.strip().split() if part]
    if len(parts) < 2:
        return "[APPLICANT]"

    first_initial = parts[0][0].upper()
    last_initial = parts[-1][0].upper()
    return f"{first_initial}. {last_initial}."


def parse_llm_json(text: str) -> dict:
    """Parse a JSON object from an LLM response with a safe fallback."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\\s*", "", cleaned, flags=re.IGNORECASE)
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

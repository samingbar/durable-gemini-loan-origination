"""Unit tests for insurance claim utility functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.workflows.insurance_claims_fixed_flow.insurance_models import InsuranceClaim
from src.workflows.insurance_claims_fixed_flow.insurance_utils import (
    calculate_claimed_to_coverage_ratio,
    calculate_estimate_gap_ratio,
    calculate_reporting_lag_days,
    compute_documentation_completeness,
    compute_metrics,
    compute_risk_score,
    derive_risk_flags,
    determine_decision,
    format_display_name,
    hard_stop_violations,
    parse_llm_json,
    sanitize_pii,
)

EXPECTED_REPORTING_LAG_DAYS = 17
REVIEW_DOCUMENT_COMPLETENESS = 0.75
EXPECTED_LLM_RISK_SCORE = 88
WEAK_CASE_MAX_SCORE = 40


def _load_cases() -> list[dict]:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "resources" / "insurance_claim_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"]


def test_calculate_ratios_and_dates() -> None:
    """Calculate the core insurance claim ratios and lag days."""
    assert calculate_claimed_to_coverage_ratio(12000, 50000) == pytest.approx(0.24)
    assert calculate_estimate_gap_ratio(29000, 21000) == pytest.approx(8000 / 29000)
    assert calculate_reporting_lag_days("2026-02-01", "2026-02-18") == EXPECTED_REPORTING_LAG_DAYS


def test_documentation_completeness() -> None:
    """Score complete and incomplete claim packets differently."""
    approved_case, conditional_case, _ = _load_cases()
    approved_claim = InsuranceClaim(**approved_case)
    conditional_claim = InsuranceClaim(**conditional_case)

    assert compute_documentation_completeness(approved_claim) == pytest.approx(1.0)
    assert compute_documentation_completeness(conditional_claim) < REVIEW_DOCUMENT_COMPLETENESS


def test_sanitize_pii() -> None:
    """Mask claimant identifiers before LLM usage."""
    claim = InsuranceClaim(**_load_cases()[0])
    sanitized = sanitize_pii(claim)

    assert sanitized.name == "[CLAIMANT_NAME]"
    assert sanitized.address == "[ADDRESS]"
    assert sanitized.email == "[EMAIL]"
    assert sanitized.phone.endswith(claim.phone[-4:])
    assert sanitized.policy_number.endswith("1137")


def test_decisions_match_expected() -> None:
    """Match the deterministic decision against each fixture's expectation."""
    for case in _load_cases():
        claim = InsuranceClaim(**case)
        metrics = compute_metrics(claim)
        decision = determine_decision(claim, metrics)
        assert decision == case["expected_decision"]


def test_hard_stop_violations() -> None:
    """Return no hard stops for the strong case and some for the weak case."""
    strong = InsuranceClaim(**_load_cases()[0])
    weak = InsuranceClaim(**_load_cases()[2])

    assert hard_stop_violations(strong, compute_metrics(strong)) == []
    assert hard_stop_violations(weak, compute_metrics(weak))


def test_parse_llm_json() -> None:
    """Parse both raw JSON and fenced JSON snippets."""
    payload = {
        "decision": "APPROVED",
        "risk_score": EXPECTED_LLM_RISK_SCORE,
        "memo": "All good.",
        "conditions": [],
    }
    text = json.dumps(payload)
    assert parse_llm_json(text)["decision"] == "APPROVED"

    wrapped = f"""```json
{text}
```"""
    assert parse_llm_json(wrapped)["risk_score"] == EXPECTED_LLM_RISK_SCORE


def test_risk_flags_and_score_for_rejected_case() -> None:
    """Expose coverage, policy, and severity issues for the rejected case."""
    claim = InsuranceClaim(**_load_cases()[2])
    metrics = compute_metrics(claim)
    flags = derive_risk_flags(claim, metrics)

    assert "Policy is not active" in flags
    assert "Premium status is not current" in flags
    assert "Coverage for this loss is not confirmed" in flags
    assert any("coverage limit" in flag.lower() for flag in flags)
    assert compute_risk_score(claim, metrics) <= WEAK_CASE_MAX_SCORE


def test_format_display_name() -> None:
    """Format claimant names as initials for the review UI."""
    claim = InsuranceClaim(**_load_cases()[0])
    assert format_display_name(claim) == "E. R."

    claim.name = "Single"
    assert format_display_name(claim) == "[CLAIMANT]"

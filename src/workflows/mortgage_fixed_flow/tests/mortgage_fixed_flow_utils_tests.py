# ruff: noqa: PLR2004
"""Unit tests for mortgage utility functions."""

from __future__ import annotations

import json
from pathlib import Path

from src.workflows.mortgage_fixed_flow.mortgage_models import MortgageApplication
from src.workflows.mortgage_fixed_flow.mortgage_utils import (
    calculate_dti_ratio,
    calculate_ltv_ratio,
    compute_metrics,
    compute_risk_score,
    derive_risk_flags,
    detect_bias_signals,
    determine_decision,
    format_display_name,
    hard_stop_violations,
    parse_llm_json,
    sanitize_pii,
    score_chunk,
    tokenize,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = REPO_ROOT / "resources" / "mortgage_test_cases.json"


def _load_cases() -> list[dict]:
    payload = json.loads(FIXTURE_PATH.read_text())
    return payload["test_cases"]


def test_calculate_ratios() -> None:
    """Return expected DTI and LTV ratios."""
    assert calculate_dti_ratio(2000, 10000) == 0.2
    assert calculate_ltv_ratio(400000, 500000) == 0.8


def test_sanitize_pii() -> None:
    """Mask direct identifiers before LLM use."""
    case = _load_cases()[0]
    applicant = MortgageApplication(**case)
    sanitized = sanitize_pii(applicant)

    assert sanitized.name == "[APPLICANT_NAME]"
    assert sanitized.address == "[ADDRESS]"
    assert sanitized.email == "[EMAIL]"
    assert sanitized.phone.endswith(case["phone"][-4:])
    assert sanitized.ssn.endswith(case["ssn"][-4:])


def test_decisions_match_expected() -> None:
    """Keep deterministic decision rules aligned with fixture expectations."""
    for case in _load_cases():
        applicant = MortgageApplication(**case)
        metrics = compute_metrics(applicant)
        decision = determine_decision(applicant, metrics)
        assert decision == case["expected_decision"]


def test_hard_stop_violations() -> None:
    """Flag hard-stop conditions only for weak cases."""
    cases = _load_cases()
    strong = MortgageApplication(**cases[0])
    weak = MortgageApplication(**cases[2])

    assert hard_stop_violations(strong, compute_metrics(strong)) == []
    assert hard_stop_violations(weak, compute_metrics(weak))


def test_parse_llm_json() -> None:
    """Handle plain JSON, fenced JSON, and invalid payloads safely."""
    payload = {
        "decision": "APPROVED",
        "risk_score": 88,
        "memo": "All good.",
        "conditions": [],
    }
    text = json.dumps(payload)
    assert parse_llm_json(text)["decision"] == "APPROVED"

    wrapped = f"""```json
{text}
```"""
    assert parse_llm_json(wrapped)["risk_score"] == 88
    assert parse_llm_json("not-json") == {}


def test_risk_flags_and_score_for_weak_case() -> None:
    """Derive expected flags and a low score for a weak application."""
    case = _load_cases()[2]
    applicant = MortgageApplication(**case)
    metrics = compute_metrics(applicant)
    flags = derive_risk_flags(applicant, metrics)

    assert "Credit score below minimum (620)" in flags
    assert "Bankruptcy history on record" in flags
    assert any("DTI above" in flag for flag in flags)
    assert compute_risk_score(applicant, metrics) <= 60


def test_format_display_name() -> None:
    """Collapse full names to initials and guard short names."""
    case = _load_cases()[0]
    applicant = MortgageApplication(**case)
    assert format_display_name(applicant) == "S. J."

    applicant.name = "Single"
    assert format_display_name(applicant) == "[APPLICANT]"


def test_detect_bias_signals() -> None:
    """Detect protected-class references in analysis text."""
    flags = detect_bias_signals("The applicant's age and marital history are noted.")
    assert "Potential bias reference detected: age" in flags
    assert "Potential bias reference detected: marital" in flags


def test_tokenize_and_score_chunk() -> None:
    """Token scoring should be case-insensitive and overlap-based."""
    assert tokenize("Credit-score minimum: 620!") == ["credit", "score", "minimum", "620"]
    assert score_chunk(["credit", "minimum"], "Minimum credit score is 620.") == 2

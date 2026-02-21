"""Unit tests for mortgage utility functions."""

from __future__ import annotations

import json
from pathlib import Path

from src.workflows.mortgage.mortgage_models import MortgageApplication
from src.workflows.mortgage.mortgage_utils import (
    calculate_dti_ratio,
    calculate_ltv_ratio,
    compute_metrics,
    compute_risk_score,
    determine_decision,
    derive_risk_flags,
    format_display_name,
    hard_stop_violations,
    parse_llm_json,
    sanitize_pii,
)


def _load_cases() -> list[dict]:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "resources" / "mortgage_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"]


def test_calculate_ratios() -> None:
    assert calculate_dti_ratio(2000, 10000) == 0.2
    assert calculate_ltv_ratio(400000, 500000) == 0.8


def test_sanitize_pii() -> None:
    case = _load_cases()[0]
    applicant = MortgageApplication(**case)
    sanitized = sanitize_pii(applicant)

    assert sanitized.name == "[APPLICANT_NAME]"
    assert sanitized.address == "[ADDRESS]"
    assert sanitized.email == "[EMAIL]"
    assert sanitized.phone.endswith(case["phone"][-4:])
    assert sanitized.ssn.endswith(case["ssn"][-4:])


def test_decisions_match_expected() -> None:
    for case in _load_cases():
        applicant = MortgageApplication(**case)
        metrics = compute_metrics(applicant)
        decision = determine_decision(applicant, metrics)
        assert decision == case["expected_decision"]


def test_hard_stop_violations() -> None:
    cases = _load_cases()
    strong = MortgageApplication(**cases[0])
    weak = MortgageApplication(**cases[2])

    assert hard_stop_violations(strong, compute_metrics(strong)) == []
    assert hard_stop_violations(weak, compute_metrics(weak))


def test_parse_llm_json() -> None:
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


def test_risk_flags_and_score_for_weak_case() -> None:
    case = _load_cases()[2]
    applicant = MortgageApplication(**case)
    metrics = compute_metrics(applicant)
    flags = derive_risk_flags(applicant, metrics)

    assert "Credit score below minimum (620)" in flags
    assert "Bankruptcy history on record" in flags
    assert any("DTI above" in flag for flag in flags)
    assert compute_risk_score(applicant, metrics) <= 60


def test_format_display_name() -> None:
    case = _load_cases()[0]
    applicant = MortgageApplication(**case)
    assert format_display_name(applicant) == "S. J."

    applicant.name = "Single"
    assert format_display_name(applicant) == "[APPLICANT]"

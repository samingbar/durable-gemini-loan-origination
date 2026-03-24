# ruff: noqa: PLR2004, SLF001
"""Unit tests for mortgage fixed-flow activities with mocked LLM calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.workflows.mortgage_fixed_flow import mortgage_activities as activities
from src.workflows.mortgage_fixed_flow.mortgage_models import (
    AgentTask,
    ApplicationOcrTask,
    CriticTask,
    DecisionTask,
    MortgageApplication,
    UnderwritingAnalyses,
)
from src.workflows.mortgage_fixed_flow.mortgage_utils import (
    compute_metrics,
    derive_risk_flags,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sample_applicant() -> MortgageApplication:
    return MortgageApplication(
        case_id="MTG-TEST-001",
        name="Jane Doe",
        ssn="123-45-6789",
        email="jane@example.com",
        phone="555-123-4567",
        address="123 Main St",
        credit_score=700,
        credit_history={
            "bankruptcies": 0,
            "foreclosures": 0,
            "late_payments_12mo": 0,
            "late_payments_24mo": 0,
            "collections": [],
            "inquiries_6mo": 1,
            "oldest_tradeline_years": 5,
            "total_tradelines": 6,
            "credit_notes": "Clean history",
        },
        employment={
            "employer": "Acme",
            "position": "Engineer",
            "years": 3.0,
            "monthly_income": 9000,
            "type": "W2",
            "employment_gap": "None",
            "gap_explanation": "N/A",
            "employment_history": [
                {
                    "employer": "Acme",
                    "position": "Engineer",
                    "years": 3.0,
                    "income": 108000,
                }
            ],
            "income_details": {
                "base_salary": 108000,
                "bonus_2023": 5000,
                "bonus_2024": 6000,
                "bonus_stable": True,
                "employer_confirmation": "Stable",
            },
        },
        debts={
            "car_loan": 300,
            "student_loan": 200,
            "credit_cards": 400,
            "total_monthly_debt": 900,
        },
        assets={
            "checking": 20000,
            "savings": 40000,
            "liquid_assets_total": 60000,
            "401k": 80000,
            "recent_deposits": [],
            "deposit_explanations": "All regular",
            "reserves_months": 6,
        },
        loan={
            "amount": 300000,
            "down_payment": 60000,
            "closing_costs": 8000,
            "estimated_payment": 2200,
            "property_type": "Single Family",
            "use": "Primary Residence",
            "monthly_piti": 2200,
        },
        property={
            "purchase_price": 360000,
            "appraised_value": 360000,
            "condition": "C3",
            "type": "Single Family",
            "required_repairs": 0,
            "repair_details": "None",
        },
        dti_ratio=0.1,
        expected_decision="APPROVED",
    )


@pytest.mark.asyncio
async def test_retrieve_policy_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the best matching policy chunk."""
    monkeypatch.setattr(
        activities,
        "_load_policy_chunks",
        lambda: ["Credit score minimum 620", "Other policy"],
    )
    result = await activities.retrieve_policy_context("credit score minimum")
    assert "credit score" in result.lower()


@pytest.mark.asyncio
async def test_extract_application_from_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """OCR only case-scoped files and validate the result."""
    applicant = _sample_applicant()
    case_id = applicant.case_id
    image_path = tmp_path / f"{case_id}_p1.png"
    image_path.write_bytes(b"fake-image")
    (tmp_path / "OTHER_p1.png").write_bytes(b"other-image")

    async def fake_ocr(image_paths: list[Path], case_id_value: str) -> str:
        assert case_id_value == case_id
        assert [path.name for path in image_paths] == [image_path.name]
        return json.dumps(applicant.model_dump(by_alias=True))

    monkeypatch.setattr(activities, "_ocr_application_from_images", fake_ocr)

    task = ApplicationOcrTask(case_id=case_id, image_dir=str(tmp_path))
    result = await activities.extract_application_from_images(task)
    assert result.case_id == case_id


def test_normalize_ocr_payload_fills_common_gaps() -> None:
    """Normalize alternate OCR section names and derive missing totals."""
    normalized = activities._normalize_ocr_payload(
        {
            "applicant_information": {
                "name": "Jane Doe",
                "credit_score": 700,
            },
            "credit_information": {
                "history": {
                    "bankruptcies": 0,
                    "foreclosures": 0,
                    "oldest_tradeline_years": 5,
                    "total_tradelines": 6,
                }
            },
            "employment_information": {
                "employer": "Acme",
                "position": "Engineer",
                "years": 3.0,
                "monthly_income": 9000,
                "type": "W2",
                "employment_gap": "None",
                "gap_explanation": "",
                "employment_history": [],
                "income_details": {
                    "base_salary": 108000,
                },
            },
            "debt_information": {
                "car_loan": 300,
                "student_loan": 200,
                "credit_cards": 400,
            },
            "asset_information": {
                "checking": 10000,
                "savings": 5000,
                "recent_deposits": [{"date": "2025-01-01", "amount": 5000}],
                "deposit_explanations": "",
                "reserves_months": 3,
                "401k": 25000,
            },
            "loan_details": {
                "amount": 300000,
                "down_payment": 60000,
                "closing_costs": 8000,
                "estimated_payment": 2200,
                "use": "Primary Residence",
                "monthly_piti": 2200,
            },
            "property_information": {
                "purchase_price": 360000,
                "appraised_value": 360000,
                "condition": "C3",
                "property_type": "Single Family",
                "required_repairs": 0,
                "repair_details": "None",
            },
        },
        "MTG-NORMALIZE",
    )

    assert normalized["case_id"] == "MTG-NORMALIZE"
    assert normalized["debts"]["total_monthly_debt"] == 900
    assert normalized["assets"]["liquid_assets_total"] == 15000
    assert normalized["assets"]["recent_deposits"][0]["description"] == "Not provided"
    assert normalized["employment"]["income_details"]["bonus_2023"] == 0
    assert normalized["loan"]["property_type"] == "Single Family"
    assert normalized["property"]["type"] == "Single Family"
    assert normalized["dti_ratio"] == pytest.approx(0.1)


def test_list_image_paths_falls_back_to_supported_extensions(tmp_path: Path) -> None:
    """Use all supported image types when case-scoped PNGs are absent."""
    jpg = tmp_path / "scan-02.jpg"
    png = tmp_path / "scan-01.png"
    jpeg = tmp_path / "scan-03.jpeg"
    for path in (jpg, png, jpeg):
        path.write_bytes(b"img")

    image_paths = activities._list_image_paths(tmp_path, "MISSING")

    assert [path.name for path in image_paths] == ["scan-01.png", "scan-02.jpg", "scan-03.jpeg"]


@pytest.mark.asyncio
async def test_run_agent_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the model output as the analysis payload."""

    async def fake_generate_text(prompt: str) -> str:
        assert "Credit analyst" in prompt
        return "analysis ok"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    applicant = _sample_applicant()
    metrics = compute_metrics(applicant)
    task = AgentTask(
        agent_name="Credit",
        applicant=applicant,
        metrics=metrics,
        policy_context="policy",
    )
    result = await activities.run_agent_analysis(task)
    assert result.analysis == "analysis ok"


@pytest.mark.asyncio
async def test_run_decision_memo_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse structured JSON responses into a recommendation."""
    payload = {
        "decision": "APPROVED",
        "risk_score": 88,
        "memo": "All good",
        "conditions": [],
    }

    async def fake_generate_text(prompt: str) -> str:
        assert "Write your response ONLY as JSON" in prompt
        return json.dumps(payload)

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    applicant = _sample_applicant()
    metrics = compute_metrics(applicant)
    analyses = UnderwritingAnalyses(credit="c", income="i", assets="a", collateral="c")
    risk_flags = derive_risk_flags(applicant, metrics)

    task = DecisionTask(
        applicant=applicant,
        metrics=metrics,
        analyses=analyses,
        risk_flags=risk_flags,
        policy_context="policy",
    )

    result = await activities.run_decision_memo(task)
    assert result.recommendation.decision == "APPROVED"
    assert result.recommendation.risk_score == 88


@pytest.mark.asyncio
async def test_run_decision_memo_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to deterministic heuristics on invalid LLM output."""

    async def fake_generate_text(_prompt: str) -> str:
        return "invalid"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    applicant = _sample_applicant()
    metrics = compute_metrics(applicant)
    analyses = UnderwritingAnalyses(credit="c", income="i", assets="a", collateral="c")
    risk_flags = derive_risk_flags(applicant, metrics)

    task = DecisionTask(
        applicant=applicant,
        metrics=metrics,
        analyses=analyses,
        risk_flags=risk_flags,
        policy_context="policy",
    )

    result = await activities.run_decision_memo(task)
    assert result.recommendation.decision == "APPROVED"
    assert result.recommendation.memo == "Fallback decision due to invalid LLM JSON response."


@pytest.mark.asyncio
async def test_run_critic_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the critic model response."""

    async def fake_generate_text(prompt: str) -> str:
        assert "senior underwriting critic" in prompt
        return "critic ok"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    applicant = _sample_applicant()
    metrics = compute_metrics(applicant)
    analyses = UnderwritingAnalyses(credit="c", income="i", assets="a", collateral="c")
    risk_flags = derive_risk_flags(applicant, metrics)
    task = CriticTask(
        applicant=applicant,
        metrics=metrics,
        analyses=analyses,
        risk_flags=risk_flags,
        policy_context="policy",
    )

    result = await activities.run_critic_review(task)
    assert result.review == "critic ok"

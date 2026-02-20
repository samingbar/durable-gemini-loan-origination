"""Unit tests for mortgage activities with mocked LLM calls."""

from __future__ import annotations

import json

import pytest

from src.workflows.mortgage import mortgage_activities as activities
from src.workflows.mortgage.mortgage_models import (
    AgentTask,
    CriticTask,
    DecisionTask,
    MortgageApplication,
    SupervisorTask,
    UnderwritingAnalyses,
)
from src.workflows.mortgage.mortgage_utils import compute_metrics, derive_risk_flags

pytestmark = pytest.mark.asyncio


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


async def test_retrieve_policy_context(monkeypatch) -> None:
    monkeypatch.setattr(
        activities,
        "_load_policy_chunks",
        lambda: ["Credit score minimum 620", "Other policy"],
    )
    result = await activities.retrieve_policy_context("credit score minimum")
    assert "credit score" in result.lower()


async def test_run_supervisor_parses_json(monkeypatch) -> None:
    async def fake_generate_text(prompt: str) -> str:
        return '{"next_agent":"income","rationale":"Need income"}'

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    applicant = _sample_applicant()
    metrics = compute_metrics(applicant)
    risk_flags = derive_risk_flags(applicant, metrics)

    task = SupervisorTask(
        applicant=applicant,
        metrics=metrics,
        completed_agents=["credit"],
        remaining_agents=["income", "assets", "collateral"],
        risk_flags=risk_flags,
        policy_context="policy",
    )

    decision = await activities.run_supervisor(task)
    assert decision.next_agent == "income"


async def test_run_supervisor_fallback(monkeypatch) -> None:
    async def fake_generate_text(prompt: str) -> str:
        return "not-json"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    applicant = _sample_applicant()
    metrics = compute_metrics(applicant)
    risk_flags = derive_risk_flags(applicant, metrics)

    task = SupervisorTask(
        applicant=applicant,
        metrics=metrics,
        completed_agents=[],
        remaining_agents=["credit", "income"],
        risk_flags=risk_flags,
        policy_context="policy",
    )

    decision = await activities.run_supervisor(task)
    assert decision.next_agent == "credit"


async def test_run_agent_analysis(monkeypatch) -> None:
    async def fake_generate_text(prompt: str) -> str:
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


async def test_run_decision_memo_json(monkeypatch) -> None:
    payload = {
        "decision": "APPROVED",
        "risk_score": 88,
        "memo": "All good",
        "conditions": [],
    }
    async def fake_generate_text(prompt: str) -> str:
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


async def test_run_decision_memo_fallback(monkeypatch) -> None:
    async def fake_generate_text(prompt: str) -> str:
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
    assert result.recommendation.memo


async def test_run_critic_review(monkeypatch) -> None:
    async def fake_generate_text(prompt: str) -> str:
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

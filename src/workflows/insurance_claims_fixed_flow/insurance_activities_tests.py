"""Unit tests for insurance claim activities with mocked LLM calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.workflows.insurance_claims_fixed_flow import insurance_activities as activities
from src.workflows.insurance_claims_fixed_flow.insurance_models import (
    AgentTask,
    ClaimAnalyses,
    ClaimOcrTask,
    CriticTask,
    DecisionTask,
    InsuranceClaim,
)
from src.workflows.insurance_claims_fixed_flow.insurance_utils import (
    compute_metrics,
    derive_risk_flags,
)

pytestmark = pytest.mark.asyncio
EXPECTED_RISK_SCORE = 88


def _load_case(index: int) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "resources" / "insurance_claim_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"][index]


def _sample_claim() -> InsuranceClaim:
    return InsuranceClaim(**_load_case(0))


async def test_retrieve_policy_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieve the highest-scoring policy chunk."""
    monkeypatch.setattr(
        activities,
        "_load_policy_chunks",
        lambda: ["Coverage limit requires active policy", "Other policy text"],
    )
    result = await activities.retrieve_policy_context("coverage limit active policy")
    assert "coverage limit" in result.lower()


async def test_extract_claim_from_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Normalize OCR output and validate it as an InsuranceClaim."""
    claim = _sample_claim()
    case_id = claim.case_id
    image_path = tmp_path / f"{case_id}_p1.png"
    image_path.write_bytes(b"fake-image")
    (tmp_path / "OTHER_p1.png").write_bytes(b"other-image")

    nested_payload = {
        "claimant_information": {
            "name": claim.name,
            "policy_number": claim.policy_number,
            "email": claim.email,
            "phone": claim.phone,
            "address": claim.address,
        },
        "policy_information": claim.policy.model_dump(),
        "incident_information": claim.incident.model_dump(),
        "loss_information": claim.loss.model_dump(),
        "documentation": claim.documents.model_dump(),
        "parties_information": claim.parties.model_dump(),
    }

    async def fake_ocr(image_paths: list[Path], case_id_value: str) -> str:
        assert case_id_value == case_id
        assert [path.name for path in image_paths] == [image_path.name]
        return json.dumps(nested_payload)

    monkeypatch.setattr(activities, "_ocr_claim_from_images", fake_ocr)

    task = ClaimOcrTask(case_id=case_id, image_dir=str(tmp_path))
    result = await activities.extract_claim_from_images(task)
    assert result.case_id == case_id
    assert result.policy.policy_type == claim.policy.policy_type


async def test_run_agent_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the generated specialist analysis."""

    async def fake_generate_text(_prompt: str) -> str:
        return "analysis ok"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    claim = _sample_claim()
    metrics = compute_metrics(claim)
    task = AgentTask(
        agent_name="Coverage",
        claim=claim,
        metrics=metrics,
        policy_context="policy",
    )
    result = await activities.run_agent_analysis(task)
    assert result.analysis == "analysis ok"


async def test_run_decision_memo_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse a structured JSON decision memo response."""
    payload = {
        "decision": "APPROVED",
        "risk_score": EXPECTED_RISK_SCORE,
        "memo": "All good",
        "conditions": [],
    }

    async def fake_generate_text(_prompt: str) -> str:
        return json.dumps(payload)

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    claim = _sample_claim()
    metrics = compute_metrics(claim)
    analyses = ClaimAnalyses(
        coverage="coverage ok",
        liability="liability ok",
        damages="damages ok",
        fraud="fraud ok",
    )
    risk_flags = derive_risk_flags(claim, metrics)

    task = DecisionTask(
        claim=claim,
        metrics=metrics,
        analyses=analyses,
        risk_flags=risk_flags,
        policy_context="policy",
    )

    result = await activities.run_decision_memo(task)
    assert result.recommendation.decision == "APPROVED"
    assert result.recommendation.risk_score == EXPECTED_RISK_SCORE


async def test_run_decision_memo_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back to deterministic claim logic when the LLM output is invalid."""

    async def fake_generate_text(_prompt: str) -> str:
        return "invalid"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    claim = _sample_claim()
    metrics = compute_metrics(claim)
    analyses = ClaimAnalyses(
        coverage="coverage ok",
        liability="liability ok",
        damages="damages ok",
        fraud="fraud ok",
    )
    risk_flags = derive_risk_flags(claim, metrics)

    task = DecisionTask(
        claim=claim,
        metrics=metrics,
        analyses=analyses,
        risk_flags=risk_flags,
        policy_context="policy",
    )

    result = await activities.run_decision_memo(task)
    assert result.recommendation.decision == "APPROVED"
    assert result.recommendation.memo


async def test_run_critic_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the generated critic review."""

    async def fake_generate_text(_prompt: str) -> str:
        return "critic ok"

    monkeypatch.setattr(activities, "_generate_text", fake_generate_text)

    claim = _sample_claim()
    metrics = compute_metrics(claim)
    risk_flags = derive_risk_flags(claim, metrics)
    analyses = ClaimAnalyses(
        coverage="coverage ok",
        liability="liability ok",
        damages="damages ok",
        fraud="fraud ok",
    )

    task = CriticTask(
        claim=claim,
        metrics=metrics,
        analyses=analyses,
        risk_flags=risk_flags,
        policy_context="policy",
    )
    result = await activities.run_critic_review(task)
    assert result.review == "critic ok"

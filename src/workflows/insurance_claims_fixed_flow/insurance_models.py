"""Pydantic models for the insurance claims workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["APPROVED", "CONDITIONAL", "REJECTED", "HUMAN_REVIEW"]
HumanDecision = Literal["APPROVED", "REJECTED"]


class DamagedItem(BaseModel):
    """Single damaged item or covered component."""

    description: str
    category: str
    estimated_cost: float


class PolicyDetails(BaseModel):
    """Policy details relevant to the claim decision."""

    line_of_business: str
    policy_type: str
    coverage_confirmed: bool
    coverage_limit: float
    deductible: float
    premium_status: str
    policy_status: str
    years_insured: float
    prior_claims_3y: int
    exclusions_noted: list[str]


class IncidentDetails(BaseModel):
    """Loss event details."""

    date_of_loss: str
    reported_date: str
    claim_type: str
    description: str
    location: str
    police_report_filed: bool
    weather_related: bool


class LossDetails(BaseModel):
    """Financial and damage details for the claim."""

    claimed_amount: float
    estimated_damage: float
    emergency_mitigation: float = 0
    depreciation_applied: float = 0
    salvage_value: float = 0
    repair_status: str
    damaged_items: list[DamagedItem]
    loss_notes: str


class ClaimDocuments(BaseModel):
    """Supporting claim documentation."""

    photos_received: bool
    repair_estimates_count: int
    receipts_count: int
    witness_statements_count: int
    proof_of_ownership: bool
    adjuster_notes: str
    missing_documents: list[str]


class ClaimParties(BaseModel):
    """People and statements involved in the claim."""

    third_party_involved: bool
    third_party_details: str
    injuries_reported: bool
    claimant_statement: str
    witness_summary: str


class InsuranceClaim(BaseModel):
    """Full insurance claim payload produced by OCR and validation."""

    case_id: str
    name: str | None = None
    policy_number: str | None = None
    email: str | None = None
    phone: str | None = None
    address: str | None = None
    policy: PolicyDetails
    incident: IncidentDetails
    loss: LossDetails
    documents: ClaimDocuments
    parties: ClaimParties
    expected_decision: str | None = None


class ClaimAdjudicationInput(BaseModel):
    """Input to the insurance claims workflow."""

    case_id: str
    image_dir: str


class ClaimMetrics(BaseModel):
    """Computed claim metrics used by deterministic and LLM steps."""

    claimed_to_coverage_ratio: float
    estimate_gap_ratio: float
    reporting_lag_days: int
    documentation_completeness: float
    net_claim_exposure: float


class ClaimAnalyses(BaseModel):
    """Outputs from the specialist claim agents."""

    coverage: str
    liability: str
    damages: str
    fraud: str


class DecisionRecommendation(BaseModel):
    """LLM-generated decision recommendation."""

    decision: Decision
    risk_score: int
    memo: str
    conditions: list[str] = Field(default_factory=list)
    human_review_reason: str | None = None


class HumanReviewInput(BaseModel):
    """Human review signal payload."""

    reviewer: str
    decision: HumanDecision
    notes: str


class HumanReviewResult(BaseModel):
    """Recorded human review outcome."""

    reviewer: str
    decision: HumanDecision
    notes: str
    timestamp: str


class HumanReviewPacket(BaseModel):
    """Payload exposed to the human review UI."""

    case_id: str
    display_name: str
    sanitized_claim: InsuranceClaim
    metrics: ClaimMetrics
    analyses: ClaimAnalyses
    critic_review: str
    decision_recommendation: DecisionRecommendation
    risk_flags: list[str]
    policy_violations: list[str]
    risk_score: int


class ClaimAdjudicationOutput(BaseModel):
    """Final workflow output."""

    case_id: str
    sanitized_claim: InsuranceClaim
    metrics: ClaimMetrics
    analyses: ClaimAnalyses
    critic_review: str
    decision_memo: str
    final_decision: Decision
    risk_score: int
    risk_flags: list[str]
    bias_flags: list[str]
    policy_violations: list[str]
    human_review_required: bool
    human_review: HumanReviewResult | None = None
    timestamp: str


class AgentTask(BaseModel):
    """Activity input for a specialist agent."""

    agent_name: str
    claim: InsuranceClaim
    metrics: ClaimMetrics
    policy_context: str


class AgentResult(BaseModel):
    """Activity output for a specialist agent."""

    analysis: str


class ClaimOcrTask(BaseModel):
    """Activity input for OCR claim extraction."""

    case_id: str
    image_dir: str


class CriticTask(BaseModel):
    """Activity input for critic review."""

    claim: InsuranceClaim
    metrics: ClaimMetrics
    analyses: ClaimAnalyses
    risk_flags: list[str]
    policy_context: str


class CriticResult(BaseModel):
    """Activity output for critic review."""

    review: str


class DecisionTask(BaseModel):
    """Activity input for decision memo drafting."""

    claim: InsuranceClaim
    metrics: ClaimMetrics
    analyses: ClaimAnalyses
    risk_flags: list[str]
    policy_context: str


class DecisionResult(BaseModel):
    """Activity output for the structured decision memo."""

    recommendation: DecisionRecommendation
    raw_response: str

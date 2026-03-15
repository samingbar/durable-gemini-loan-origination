"""Temporal workflow for insurance claims adjudication."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .insurance_activities import (
        extract_claim_from_images,
        retrieve_policy_context,
        run_agent_analysis,
        run_critic_review,
        run_decision_memo,
    )
from .insurance_models import (
    AgentTask,
    ClaimAdjudicationInput,
    ClaimAdjudicationOutput,
    ClaimAnalyses,
    ClaimOcrTask,
    CriticTask,
    DecisionRecommendation,
    DecisionTask,
    HumanReviewInput,
    HumanReviewPacket,
    HumanReviewResult,
)
from .insurance_utils import (
    compute_metrics,
    derive_risk_flags,
    detect_bias_signals,
    format_display_name,
    hard_stop_violations,
    sanitize_pii,
)


@workflow.defn
class InsuranceClaimAdjudicationWorkflow:
    """Orchestrate the insurance claims process."""

    def __init__(self) -> None:
        """Initialize workflow state for the human review gate."""
        self._human_review: HumanReviewResult | None = None
        self._review_packet: HumanReviewPacket | None = None

    @workflow.signal
    async def submit_human_review(self, review: HumanReviewInput) -> None:
        """Receive human review decision and notes."""
        self._human_review = HumanReviewResult(
            reviewer=review.reviewer,
            decision=review.decision,
            notes=review.notes,
            timestamp=workflow.now().isoformat(),
        )

    @workflow.query
    def get_review_packet(self) -> HumanReviewPacket | None:
        """Expose the review packet for the human review UI."""
        return self._review_packet

    @workflow.run
    async def run(self, input_data: ClaimAdjudicationInput) -> ClaimAdjudicationOutput:
        """Run the deterministic insurance claim workflow."""
        workflow.logger.info(
            "Starting insurance claim adjudication for case %s", input_data.case_id
        )

        claim = await workflow.execute_activity(
            extract_claim_from_images,
            ClaimOcrTask(case_id=input_data.case_id, image_dir=input_data.image_dir),
            start_to_close_timeout=timedelta(minutes=2),
        )

        sanitized = sanitize_pii(claim)
        sanitized_payload = sanitized.model_dump()
        metrics = compute_metrics(claim)
        risk_flags = derive_risk_flags(claim, metrics)

        policy_queries = {
            "coverage": "coverage confirmation, deductibles, policy status, claim limits",
            "liability": "third party review, injuries, witness requirements, police reports",
            "damages": "damage estimate validation, repair documentation, mitigation costs",
            "fraud": "fraud indicators, reporting delays, repeat claims, inconsistent estimates",
            "decision": (
                "overall claim settlement thresholds, denial triggers, human review criteria"
            ),
        }

        policy_futures = {
            key: workflow.start_activity(
                retrieve_policy_context,
                query,
                start_to_close_timeout=timedelta(seconds=15),
            )
            for key, query in policy_queries.items()
        }
        policy_context = {key: await future for key, future in policy_futures.items()}

        analyses = ClaimAnalyses(coverage="", liability="", damages="", fraud="")
        for agent in ["coverage", "liability", "damages", "fraud"]:
            result = await workflow.execute_activity(
                run_agent_analysis,
                AgentTask(
                    agent_name=agent.title(),
                    claim=sanitized_payload,
                    metrics=metrics,
                    policy_context=policy_context[agent],
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )
            setattr(analyses, agent, result.analysis)

        critic_review = await workflow.execute_activity(
            run_critic_review,
            CriticTask(
                claim=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                risk_flags=risk_flags,
                policy_context=policy_context["decision"],
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )

        decision_result = await workflow.execute_activity(
            run_decision_memo,
            DecisionTask(
                claim=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                risk_flags=risk_flags,
                policy_context=policy_context["decision"],
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        decision_recommendation = DecisionRecommendation.model_validate(
            decision_result.recommendation.model_dump()
        )

        real_name = claim.name or "[CLAIMANT_NAME]"
        memo_with_name = decision_recommendation.memo.replace("[CLAIMANT_NAME]", real_name)

        bias_flags = []
        for text in [
            analyses.coverage,
            analyses.liability,
            analyses.damages,
            analyses.fraud,
            critic_review.review,
            decision_recommendation.memo,
        ]:
            bias_flags.extend(detect_bias_signals(text))
        bias_flags = sorted(set(bias_flags))

        policy_violations = hard_stop_violations(claim, metrics)
        if decision_recommendation.decision == "HUMAN_REVIEW":
            decision_recommendation = decision_recommendation.model_copy(
                update={
                    "decision": "CONDITIONAL",
                    "human_review_reason": decision_recommendation.human_review_reason
                    or "LLM requested human review.",
                },
            )
        if policy_violations and decision_recommendation.decision == "APPROVED":
            decision_recommendation = decision_recommendation.model_copy(
                update={
                    "decision": "CONDITIONAL",
                    "human_review_reason": "Policy hard-stop violations require review.",
                },
            )

        final_decision = decision_recommendation.decision
        risk_score = decision_recommendation.risk_score

        human_review_required = final_decision == "CONDITIONAL"
        human_review: HumanReviewResult | None = None

        if human_review_required:
            self._review_packet = HumanReviewPacket(
                case_id=input_data.case_id,
                display_name=format_display_name(claim),
                sanitized_claim=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                critic_review=critic_review.review,
                decision_recommendation=decision_recommendation,
                risk_flags=risk_flags,
                policy_violations=policy_violations,
                risk_score=risk_score,
            )
            await workflow.wait_condition(lambda: self._human_review is not None)
            human_review = self._human_review
            final_decision = human_review.decision

        return ClaimAdjudicationOutput(
            case_id=input_data.case_id,
            sanitized_claim=sanitized_payload,
            metrics=metrics,
            analyses=analyses,
            critic_review=critic_review.review,
            decision_memo=memo_with_name,
            final_decision=final_decision,
            risk_score=risk_score,
            risk_flags=risk_flags,
            bias_flags=bias_flags,
            policy_violations=policy_violations,
            human_review_required=human_review_required,
            human_review=human_review,
            timestamp=workflow.now().isoformat(),
        )

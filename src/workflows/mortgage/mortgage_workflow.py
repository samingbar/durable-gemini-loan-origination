"""Temporal workflow for mortgage underwriting."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from .mortgage_activities import (
        extract_application_from_images,
        retrieve_policy_context,
        run_agent_analysis,
        run_critic_review,
        run_decision_memo,
        run_supervisor,
    )
from .mortgage_models import (
    AgentTask,
    CriticTask,
    DecisionTask,
    DecisionRecommendation,
    HumanReviewInput,
    HumanReviewPacket,
    HumanReviewResult,
    ApplicationOcrTask,
    SupervisorTask,
    UnderwritingAnalyses,
    UnderwritingInput,
    UnderwritingOutput,
)
from .mortgage_utils import (
    compute_metrics,
    detect_bias_signals,
    derive_risk_flags,
    format_display_name,
    hard_stop_violations,
    sanitize_pii,
)


@workflow.defn
class MortgageUnderwritingWorkflow:
    """Orchestrate the mortgage underwriting process."""

    def __init__(self) -> None:
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
    async def run(self, input_data: UnderwritingInput) -> UnderwritingOutput:
        workflow.logger.info("Starting underwriting for case %s", input_data.case_id)

        applicant = await workflow.execute_activity(
            extract_application_from_images,
            ApplicationOcrTask(case_id=input_data.case_id, image_dir=input_data.image_dir),
            start_to_close_timeout=timedelta(minutes=2),
        )

        sanitized = sanitize_pii(applicant)
        sanitized_payload = sanitized.model_dump(by_alias=True)
        metrics = compute_metrics(applicant)
        risk_flags = derive_risk_flags(applicant, metrics)

        policy_queries = {
            "credit": "credit score minimums, bankruptcy, late payments",
            "income": "income verification requirements, self-employment, DTI limits",
            "assets": "asset verification, reserves requirements, large deposits",
            "collateral": "property condition, appraisal requirements, repairs",
            "decision": "overall underwriting decision thresholds",
            "supervisor": "underwriting workflow routing and documentation sequencing",
        }

        policy_futures = {
            key: workflow.start_activity(
                retrieve_policy_context,
                query,
                start_to_close_timeout=timedelta(seconds=15),
            )
            for key, query in policy_queries.items()
        }
        policy_context = {key: await fut for key, fut in policy_futures.items()}

        analyses = UnderwritingAnalyses(credit="", income="", assets="", collateral="")
        completed: set[str] = set()
        max_steps = 6

        for _ in range(max_steps):
            remaining = [
                agent
                for agent in ["credit", "income", "assets", "collateral"]
                if agent not in completed
            ]
            if not remaining:
                break

            supervisor_decision = await workflow.execute_activity(
                run_supervisor,
                SupervisorTask(
                    applicant=sanitized_payload,
                    metrics=metrics,
                    completed_agents=sorted(completed),
                    remaining_agents=remaining,
                    risk_flags=risk_flags,
                    policy_context=policy_context["supervisor"],
                ),
                start_to_close_timeout=timedelta(seconds=30),
            )

            next_agent = supervisor_decision.next_agent
            if next_agent not in remaining:
                next_agent = remaining[0]

            agent_task = AgentTask(
                agent_name=next_agent.title(),
                applicant=sanitized_payload,
                metrics=metrics,
                policy_context=policy_context[next_agent],
            )
            result = await workflow.execute_activity(
                run_agent_analysis,
                agent_task,
                start_to_close_timeout=timedelta(seconds=60),
            )

            setattr(analyses, next_agent, result.analysis)
            completed.add(next_agent)

        remaining = [
            agent
            for agent in ["credit", "income", "assets", "collateral"]
            if agent not in completed
        ]
        for agent in remaining:
            result = await workflow.execute_activity(
                run_agent_analysis,
                AgentTask(
                    agent_name=agent.title(),
                    applicant=sanitized_payload,
                    metrics=metrics,
                    policy_context=policy_context[agent],
                ),
                start_to_close_timeout=timedelta(seconds=60),
            )
            setattr(analyses, agent, result.analysis)
            completed.add(agent)

        critic_policy = policy_context["decision"]

        critic_review = await workflow.execute_activity(
            run_critic_review,
            CriticTask(
                applicant=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                risk_flags=risk_flags,
                policy_context=critic_policy,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )

        decision_result = await workflow.execute_activity(
            run_decision_memo,
            DecisionTask(
                applicant=sanitized_payload,
                metrics=metrics,
                analyses=analyses,
                risk_flags=risk_flags,
                policy_context=critic_policy,
            ),
            start_to_close_timeout=timedelta(seconds=60),
        )
        decision_recommendation = DecisionRecommendation.model_validate(
            decision_result.recommendation.model_dump()
        )
        real_name = applicant.name or "[APPLICANT_NAME]"
        memo_with_name = decision_recommendation.memo.replace("[APPLICANT_NAME]", real_name)

        bias_flags = []
        for text in [
            analyses.credit,
            analyses.income,
            analyses.assets,
            analyses.collateral,
            critic_review.review,
            decision_recommendation.memo,
        ]:
            bias_flags.extend(detect_bias_signals(text))
        bias_flags = sorted(set(bias_flags))

        policy_violations = hard_stop_violations(applicant, metrics)
        if policy_violations and decision_recommendation.decision == "APPROVED":
            decision_recommendation = decision_recommendation.model_copy(
                update={
                    "decision": "HUMAN_REVIEW",
                    "human_review_reason": "Policy hard-stop violations require review.",
                },
            )

        final_decision = decision_recommendation.decision
        risk_score = decision_recommendation.risk_score

        human_review_required = final_decision == "HUMAN_REVIEW"
        human_review: HumanReviewResult | None = None

        if human_review_required:
            self._review_packet = HumanReviewPacket(
                case_id=input_data.case_id,
                display_name=format_display_name(applicant),
                sanitized_applicant=sanitized_payload,
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

        return UnderwritingOutput(
            case_id=input_data.case_id,
            sanitized_applicant=sanitized_payload,
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

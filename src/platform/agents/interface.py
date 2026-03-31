"""Provider interfaces for model-backed operational AI tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class InsuranceAgentProvider(Protocol):
    """Contract for insurance-claims intelligence providers."""

    @property
    def provider_name(self) -> str:
        """Return the provider name used for observability."""

    async def generate_text(self, prompt: str) -> str:
        """Generate plain text for a prompt."""

    async def extract_insurance_claim_json(
        self,
        image_paths: list[Path],
        *,
        case_id: str,
        schema: str,
    ) -> str:
        """Extract a structured insurance-claim JSON payload from images."""

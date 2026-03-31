"""HTTP adapter for Agent Bricks-compatible insurance inference endpoints."""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

import aiohttp

from .interface import InsuranceAgentProvider

if TYPE_CHECKING:
    from pathlib import Path


def _require_env(name: str) -> str:
    """Return a required environment variable value."""
    value = os.environ.get(name)
    if value:
        return value
    message = f"Set {name} to enable the Agent Bricks provider."
    raise RuntimeError(message)


def _auth_headers() -> dict[str, str]:
    """Build authorization headers for Agent Bricks requests."""
    token = os.environ.get("AGENT_BRICKS_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    """POST JSON and return the decoded response body."""
    async with (
        aiohttp.ClientSession(headers=_auth_headers()) as session,
        session.post(
            url,
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        data = await response.json()
        if not isinstance(data, dict):
            message = f"Expected a JSON object from {url}"
            raise TypeError(message)
        return data


def _response_text(payload: dict[str, object], *, url: str) -> str:
    """Extract the text payload from an Agent Bricks response."""
    for key in ("text", "output_text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    message = f"Agent Bricks response from {url} did not include text output."
    raise KeyError(message)


class AgentBricksInsuranceProvider(InsuranceAgentProvider):
    """Insurance provider that delegates inference to Agent Bricks endpoints."""

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "agent_bricks"

    async def generate_text(self, prompt: str) -> str:
        """Generate prompt output through an Agent Bricks text endpoint."""
        url = _require_env("AGENT_BRICKS_TEXT_URL")
        response = await _post_json(url, {"prompt": prompt})
        return _response_text(response, url=url)

    async def extract_insurance_claim_json(
        self,
        image_paths: list[Path],
        *,
        case_id: str,
        schema: str,
    ) -> str:
        """Extract structured insurance-claim data through Agent Bricks."""
        url = _require_env("AGENT_BRICKS_INSURANCE_OCR_URL")
        images = [
            {
                "name": path.name,
                "mime_type": (
                    "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                ),
                "data_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
            for path in image_paths
        ]
        response = await _post_json(
            url,
            {
                "case_id": case_id,
                "schema": schema,
                "images": images,
            },
        )
        return _response_text(response, url=url)

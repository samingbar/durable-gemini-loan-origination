"""Gemini-backed implementation of the insurance agent provider interface."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from .interface import InsuranceAgentProvider

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


@lru_cache(maxsize=1)
def _gemini_client() -> genai.Client:
    """Build and cache a Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        message = "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use Gemini."
        raise RuntimeError(message)
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _model_name() -> str:
    """Return the configured Gemini model name."""
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


class GeminiInsuranceProvider(InsuranceAgentProvider):
    """Insurance intelligence provider backed by Gemini."""

    @property
    def provider_name(self) -> str:
        """Return the provider identifier."""
        return "gemini"

    async def generate_text(self, prompt: str) -> str:
        """Generate prompt output through Gemini."""
        response = await _gemini_client().aio.models.generate_content(
            model=_model_name(),
            contents=prompt,
        )
        return response.text or ""

    async def extract_insurance_claim_json(
        self,
        image_paths: list[Path],
        *,
        case_id: str,
        schema: str,
    ) -> str:
        """Extract structured claim data from image pages with Gemini."""
        contents: list[types.Part | str] = [
            "Extract the insurance claim data from these images.\n"
            "Return JSON ONLY (no markdown) that matches the InsuranceClaim schema exactly.\n"
            "Rules:\n"
            f"- Use case_id exactly as: {case_id}\n"
            "- Output a single JSON object with the exact field names.\n"
            "- Include ALL required fields even if blank.\n"
            "- Optional fields may be null.\n"
            "- If a numeric value is missing, use 0.\n"
            "- If a boolean value is missing, use false.\n"
            "- If a string value is missing, use an empty string.\n"
            "- If a list is missing, use an empty list.\n"
            "Schema:\n"
            f"{schema}",
        ]

        for path in image_paths:
            mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type))

        response = await _gemini_client().aio.models.generate_content(
            model=_model_name(),
            contents=contents,
        )
        return response.text or ""

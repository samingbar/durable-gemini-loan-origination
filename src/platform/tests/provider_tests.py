"""Tests for model-provider adapters used by the insurance flow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.platform.agents import agent_bricks_provider, gemini_provider

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_GEMINI_CONTENTS = 2


@pytest.mark.asyncio
async def test_gemini_provider_generate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return text from the mocked Gemini client."""

    class FakeResponse:
        text = "hello from gemini"

    class FakeModels:
        async def generate_content(self, **kwargs: object) -> FakeResponse:
            assert kwargs["contents"] == "prompt"
            return FakeResponse()

    class FakeAio:
        models = FakeModels()

    class FakeClient:
        aio = FakeAio()

    monkeypatch.setattr(gemini_provider, "_gemini_client", lambda: FakeClient())

    provider = gemini_provider.GeminiInsuranceProvider()
    assert await provider.generate_text("prompt") == "hello from gemini"


@pytest.mark.asyncio
async def test_gemini_provider_extract_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Build a multimodal Gemini request from local image files."""
    image_path = tmp_path / "claim_p1.png"
    image_path.write_bytes(b"img")

    class FakeResponse:
        text = '{"case_id":"CLM-1"}'

    class FakeModels:
        async def generate_content(self, **kwargs: object) -> FakeResponse:
            contents = kwargs["contents"]
            assert isinstance(contents, list)
            assert len(contents) == EXPECTED_GEMINI_CONTENTS
            return FakeResponse()

    class FakeAio:
        models = FakeModels()

    class FakeClient:
        aio = FakeAio()

    monkeypatch.setattr(gemini_provider, "_gemini_client", lambda: FakeClient())

    provider = gemini_provider.GeminiInsuranceProvider()
    result = await provider.extract_insurance_claim_json(
        [image_path],
        case_id="CLM-1",
        schema="{}",
    )
    assert '"CLM-1"' in result


@pytest.mark.asyncio
async def test_agent_bricks_provider_generate_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate prompt generation to the configured Agent Bricks text URL."""
    captured: dict[str, object] = {}

    async def fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        return {"text": "hello from bricks"}

    monkeypatch.setenv("AGENT_BRICKS_TEXT_URL", "https://agent-bricks.local/text")
    monkeypatch.setattr(agent_bricks_provider, "_post_json", fake_post_json)

    provider = agent_bricks_provider.AgentBricksInsuranceProvider()
    assert await provider.generate_text("prompt") == "hello from bricks"
    assert captured["url"] == "https://agent-bricks.local/text"
    assert captured["payload"] == {"prompt": "prompt"}


@pytest.mark.asyncio
async def test_agent_bricks_provider_extract_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Encode images and send them to the configured Agent Bricks OCR URL."""
    captured: dict[str, object] = {}
    image_path = tmp_path / "claim_p1.png"
    image_path.write_bytes(b"image-bytes")

    async def fake_post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        return {"output_text": '{"case_id":"CLM-7"}'}

    monkeypatch.setenv(
        "AGENT_BRICKS_INSURANCE_OCR_URL",
        "https://agent-bricks.local/ocr",
    )
    monkeypatch.setattr(agent_bricks_provider, "_post_json", fake_post_json)

    provider = agent_bricks_provider.AgentBricksInsuranceProvider()
    result = await provider.extract_insurance_claim_json(
        [image_path],
        case_id="CLM-7",
        schema="{}",
    )

    assert '"CLM-7"' in result
    assert captured["url"] == "https://agent-bricks.local/ocr"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["case_id"] == "CLM-7"
    assert isinstance(payload["images"], list)
    assert payload["images"][0]["data_base64"]

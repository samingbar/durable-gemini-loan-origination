"""File-backed downstream claims-system adapter for local demos."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _outbox_path() -> Path:
    """Return the local outbox path used to emulate a claims system."""
    configured = os.environ.get("CLAIMS_SYSTEM_OUTBOX_PATH")
    if configured:
        return Path(configured).expanduser()
    return _repo_root() / "datasets" / "operations" / "claim_system_updates.jsonl"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


class ClaimsDownstreamAdapter:
    """Persist idempotent claim updates to a local outbox file."""

    target_system = "claims-system-demo"

    def __init__(self, outbox_path: Path | None = None) -> None:
        """Initialize the adapter with an optional outbox path override."""
        self._outbox_path = outbox_path or _outbox_path()
        self._outbox_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def outbox_path(self) -> Path:
        """Return the local outbox file path."""
        return self._outbox_path

    def _read_existing(self, idempotency_key: str) -> dict[str, object] | None:
        """Return an existing outbox event for an idempotency key, if any."""
        if not self._outbox_path.exists():
            return None
        for line in self._outbox_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("idempotency_key") == idempotency_key:
                return payload
        return None

    def publish_claim_decision(  # noqa: PLR0913
        self,
        *,
        case_id: str,
        workflow_id: str,
        idempotency_key: str,
        final_decision: str,
        risk_score: int,
        decision_memo: str,
        reviewer: str | None = None,
    ) -> dict[str, object]:
        """Persist a downstream claim-decision update with idempotency."""
        existing = self._read_existing(idempotency_key)
        if existing is not None:
            existing["action_status"] = "ALREADY_PUBLISHED"
            return existing

        external_record_id = f"CLAIM-SYNC-{case_id}"
        payload = {
            "action_status": "PUBLISHED",
            "idempotency_key": idempotency_key,
            "case_id": case_id,
            "workflow_id": workflow_id,
            "external_record_id": external_record_id,
            "final_decision": final_decision,
            "risk_score": risk_score,
            "decision_memo": decision_memo,
            "reviewer": reviewer,
            "target_system": self.target_system,
            "published_at": _utc_now_iso(),
        }
        with self._outbox_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(payload, sort_keys=True)}\n")
        return payload

"""Native pre-compress archives: completed retention followed by exact text recovery."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hindsight_client_api.models.retain_request import RetainRequest

# Refuse oversized checkpoints rather than archiving a lossy preview.
MAX_CHECKPOINT_BYTES = 1_048_576


@dataclass(frozen=True)
class Checkpoint:
    document_id: str
    content: str
    digest: str

    @classmethod
    def from_messages(cls, messages: list[dict[str, Any]], session_id: str, profile: str) -> Checkpoint:
        content = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the 1 MiB limit")
        digest = hashlib.sha256(encoded).hexdigest()
        session = hashlib.sha256(json.dumps([profile, session_id]).encode("utf-8")).hexdigest()
        return cls(f"hermes-checkpoint-{session}-{digest}", content, digest)

    def verify(self, document: Any, bank_id: str) -> None:
        if document.id != self.document_id or document.bank_id != bank_id or document.original_text != self.content:
            raise ValueError("checkpoint document does not match the evidence")


async def retain_checkpoint(
    client: Any,
    checkpoint: Checkpoint,
    *,
    bank_ids: list[str],
    request: RetainRequest,
    deadline: float,
) -> None:
    """Verify every destination under the caller's single deadline; never retry a write."""
    from hindsight_client_api.exceptions import ApiException

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("checkpoint deadline exceeded")
        return seconds

    # Preflight ALL destinations before any write; unknown strategies silently fall back server-side.
    for bank_id in bank_ids:
        response = await client.banks.get_bank_config(bank_id=bank_id, _request_timeout=remaining())
        if response.bank_id != bank_id or response.config.get("store_document_text") is not True:
            raise ValueError("checkpoint requires full document text storage")
        strategy = request.items[0].strategy or response.config.get("retain_default_strategy")
        if strategy:
            overrides = (response.config.get("retain_strategies") or {}).get(strategy)
            if not isinstance(overrides, dict) or overrides.get("store_document_text", True) is not True:
                raise ValueError("checkpoint retain strategy is unavailable or disables document text storage")

    for bank_id in bank_ids:
        try:
            document = await client.documents.get_document(
                bank_id=bank_id, document_id=checkpoint.document_id, _request_timeout=remaining()
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
        else:
            checkpoint.verify(document, bank_id)
            continue

        response = await client.memory.retain_memories(bank_id, request, _request_timeout=remaining())
        if response.success is not True or response.var_async is not False:
            raise ValueError("checkpoint retain did not complete synchronously")
        document = await client.documents.get_document(
            bank_id=bank_id, document_id=checkpoint.document_id, _request_timeout=remaining()
        )
        checkpoint.verify(document, bank_id)

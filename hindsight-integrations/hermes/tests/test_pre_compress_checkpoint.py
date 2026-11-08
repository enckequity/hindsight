"""Strict compaction requires a recoverable document, not an accepted retain job."""

import asyncio
import hashlib
import json
import threading
import time
import types

import pytest
from hindsight_client_api.exceptions import ApiException

from conftest import FakeClient

MESSAGES = [{"role": "user", "content": "Keep café ☕"}, {"role": "assistant", "content": "Recorded."}]


class CheckpointClient(FakeClient):
    """In-memory boundary for native config, retain and document operations."""

    def __init__(self):
        super().__init__()
        self.stored = {}
        self.events = []
        self.config = {"store_document_text": True, "retain_strategies": {}}
        self.banks = types.SimpleNamespace(get_bank_config=self.get_config)
        self.memory = types.SimpleNamespace(retain_memories=self.retain_memories)
        self.documents = types.SimpleNamespace(get_document=self.get_document)
        self.corrupt = None
        self.error_phase = None
        self.delay = 0
        self.cancelled = threading.Event()
        self.response_success = True
        self.response_async = False
        self.missing_after_write = False
        self.error_status = 402
        self.uncertain_write = False
        self.timeouts = []

    async def step(self, phase, bank):
        self.events.append((phase, bank))
        if self.error_phase == phase:
            raise ApiException(status=self.error_status, reason="test refusal")
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def get_config(self, bank_id, **kwargs):
        self.timeouts.append(kwargs["_request_timeout"])
        await self.step("config", bank_id)
        return types.SimpleNamespace(bank_id=bank_id, config=self.config)

    async def get_document(self, bank_id, document_id, **kwargs):
        self.timeouts.append(kwargs["_request_timeout"])
        await self.step("get", bank_id)
        if (bank_id, document_id) not in self.stored or self.missing_after_write:
            raise ApiException(status=404)
        text = self.stored[(bank_id, document_id)]
        return types.SimpleNamespace(
            id=document_id if self.corrupt != "id" else "wrong",
            bank_id=bank_id if self.corrupt != "bank" else "wrong",
            original_text=None if self.corrupt == "null" else text + ("!" if self.corrupt == "text" else ""),
            memory_unit_count=0,
        )

    async def retain_memories(self, bank_id, request, **kwargs):
        self.timeouts.append(kwargs["_request_timeout"])
        return await self.aretain_batch(
            bank_id=bank_id,
            items=[request.items[0].to_dict()],
            document_id=request.items[0].document_id,
            retain_async=request.var_async,
        )

    async def aretain_batch(self, **kwargs):
        await self.step("retain", kwargs["bank_id"])
        self.retains.append(kwargs)
        item = kwargs["items"][0]
        self.stored[(kwargs["bank_id"], kwargs["document_id"])] = item["content"]
        if self.uncertain_write:
            raise ConnectionError("Cannot connect to host after an uncertain write")
        return types.SimpleNamespace(success=self.response_success, var_async=self.response_async)


def connected(provider, config=None, **kwargs):
    client = CheckpointClient()
    instance, _ = provider({"checkpoint_enabled": True, **(config or {})}, client=client, **kwargs)
    instance._get_client()
    return instance, client


def test_checkpoint_round_trip_is_exact_separate_and_reused(provider):
    instance, client = connected(provider, {"retain_tags": ["tenant:test"], "additional_banks": ["archive"]})
    session_document = instance._document_id
    assert instance.pre_compress_checkpoint_api_version == 2
    result = instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 2
    for call in client.retains:
        assert call["retain_async"] is False
        assert "operation_id" not in call
        assert call["document_id"] != session_document
        text = call["items"][0]["content"]
        assert json.loads(text) == MESSAGES
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest in call["document_id"]
        assert "tenant:test" in call["items"][0]["tags"]
        assert digest in result
    assert "Keep café" not in result
    assert instance._document_id == session_document
    assert instance._session_turns == []
    assert instance._pending_retain_ops == set()
    assert instance._writer_thread is None
    assert client.events[-1] == ("get", "archive")
    instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 2
    instance.on_pre_compress([{"role": "user", "content": "Different"}], require_checkpoint=True)
    assert len(client.retains) == 4


@pytest.mark.parametrize("corrupt", ["id", "bank", "null", "text"])
@pytest.mark.parametrize("required", [False, True])
def test_checkpoint_bad_readback_never_reports_success(provider, corrupt, required):
    instance, client = connected(provider)
    client.corrupt = corrupt
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=required)
    assert len(client.retains) == 1


@pytest.mark.parametrize("phase", ["config", "get", "retain"])
def test_checkpoint_refusal_does_not_retry_or_recreate_client(provider, phase):
    instance, client = connected(provider)
    instance._mode = "local_embedded"
    client.error_phase = phase
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert instance._client is client
    assert client.events.count((phase, "hermes")) == 1


@pytest.mark.parametrize(
    "config,context",
    [
        ({}, "primary"),
        ({"checkpoint_enabled": "false"}, "primary"),
        ({"checkpoint_enabled": True, "auto_retain": False}, "primary"),
        ({"checkpoint_enabled": True}, "subagent"),
        ({"checkpoint_enabled": True}, "kanban"),
        ({"checkpoint_enabled": True}, "cron"),
    ],
)
def test_checkpoint_disabled_or_disallowed_context_is_incapable(provider, config, context):
    client = CheckpointClient()
    instance, _ = provider(config, client=client, agent_context=context)
    instance._get_client()
    assert instance.pre_compress_checkpoint_api_version == 1
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.events == []


def test_checkpoint_requires_connected_compatible_client(provider):
    instance, _ = provider({"checkpoint_enabled": True}, client=CheckpointClient())
    assert instance.pre_compress_checkpoint_api_version == 1
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert instance._client is None
    instance._client = FakeClient()
    assert instance.pre_compress_checkpoint_api_version == 1


@pytest.mark.parametrize(
    "config",
    [
        {"store_document_text": False},
        {"store_document_text": "true"},
        {"store_document_text": True, "retain_default_strategy": "missing"},
        {
            "store_document_text": True,
            "retain_default_strategy": "archive",
            "retain_strategies": {"archive": {"store_document_text": False}},
        },
    ],
)
def test_checkpoint_configuration_must_preserve_full_text(provider, config):
    instance, client = connected(provider)
    client.config = config
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.retains == []


def test_checkpoint_unknown_explicit_strategy_is_not_silently_used(provider):
    instance, client = connected(provider, {"retain_strategy": "missing"})
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.retains == []


def test_checkpoint_all_banks_share_one_deadline_and_cancel_pending_work(provider):
    instance, client = connected(provider, {"additional_banks": ["archive"]})
    instance._timeout = 0.1
    client.delay = 0.025
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert time.monotonic() - started < 0.5
    assert client.cancelled.wait(0.5)
    assert len(client.retains) <= 1
    # Cancellation cannot be reported as a rollback of an already committed write.
    assert not any(bank == "archive" and phase == "retain" for phase, bank in client.events)


def test_checkpoint_size_limit_rejects_without_truncation(provider):
    instance, client = connected(provider)
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress([{"role": "user", "content": "é" * 1_048_576}], require_checkpoint=True)
    assert client.events == []


def test_checkpoint_interleaved_sessions_never_overwrite_each_other(provider):
    first, a = connected(provider, {"bank_id": "one"}, session_id="session-a")
    second, b = connected(provider, {"bank_id": "two"}, session_id="session-b")
    first.on_pre_compress(MESSAGES, require_checkpoint=True)
    second.on_pre_compress(MESSAGES, require_checkpoint=True)
    first.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(a.retains) == len(b.retains) == 1
    assert {bank for bank, _ in a.stored} == {"one"}
    assert {bank for bank, _ in b.stored} == {"two"}
    assert a.retains[0]["document_id"] != b.retains[0]["document_id"]


@pytest.mark.parametrize("success,queued", [(False, False), (True, True), (None, False)])
def test_checkpoint_retain_receipt_must_be_successful_and_synchronous(provider, success, queued):
    instance, client = connected(provider)
    client.response_success, client.response_async = success, queued
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 1


def test_checkpoint_document_still_missing_after_completed_retain_blocks(provider):
    instance, client = connected(provider)
    client.missing_after_write = True
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 1
    assert client.events[-1] == ("get", "hermes")


def test_checkpoint_uncertain_write_is_recovered_by_readback_not_resubmission(provider):
    instance, client = connected(provider)
    instance._mode = "local_embedded"
    client.uncertain_write = True
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 1
    assert instance._client is client
    # The request may have committed despite the lost response. A fresh hook invocation reads first.
    client.uncertain_write = False
    instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_checkpoint_read_error_is_not_treated_as_absence(provider, status):
    instance, client = connected(provider)
    client.error_phase, client.error_status = "get", status
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.retains == []


def test_checkpoint_every_request_uses_remaining_deadline(provider):
    instance, client = connected(provider)
    instance._timeout = 0.5
    client.delay = 0.005
    instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert 0 < client.timeouts[-1] < client.timeouts[0] < 0.5
    assert all(later < earlier for earlier, later in zip(client.timeouts, client.timeouts[1:]))


def test_checkpoint_full_text_disabled_in_second_bank_prevents_all_writes(provider):
    instance, client = connected(provider, {"additional_banks": ["archive"]})
    original = client.banks.get_bank_config

    async def bank_config(bank_id, **kwargs):
        response = await original(bank_id, **kwargs)
        if bank_id == "archive":
            response.config = {"store_document_text": False}
        return response

    client.banks.get_bank_config = bank_config
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.retains == []


def test_checkpoint_empty_evidence_is_an_exact_archive_not_an_empty_success(provider):
    instance, client = connected(provider)
    result = instance.on_pre_compress([], require_checkpoint=True)
    assert len(client.retains) == 1
    assert client.retains[0]["items"][0]["content"] == "[]"
    assert "checkpoint" in result


def test_checkpoint_cron_requires_both_explicit_opt_ins(provider):
    instance, client = connected(provider, {"retain_cron": True}, agent_context="cron")
    assert instance.pre_compress_checkpoint_api_version == 2
    instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert len(client.retains) == 1


def test_checkpoint_malformed_success_response_does_not_allow_compaction(provider):
    instance, client = connected(provider)
    retain = client.memory.retain_memories

    async def missing_async_flag(*args, **kwargs):
        await retain(*args, **kwargs)
        return types.SimpleNamespace(success=True)

    client.memory.retain_memories = missing_async_flag
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)


def test_checkpoint_sdk_must_preserve_policy_fields_before_any_write(provider, monkeypatch):
    from hindsight_client_api.models.memory_item import MemoryItem

    instance, client = connected(provider, {"retain_tags": ["private:fixture"]})
    convert = MemoryItem.from_dict

    def old_schema(raw):
        item = convert(raw)
        item.tags = None
        return item

    monkeypatch.setattr(MemoryItem, "from_dict", old_schema)
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.retains == []


def test_default_operation_bridge_still_retries_a_stale_repeatable_connection(provider, monkeypatch):
    instance, first = connected(provider)
    replacement = CheckpointClient()
    monkeypatch.setattr(instance, "_new_cloud_client", lambda: replacement)
    attempted = []

    async def operation(client):
        attempted.append(client)
        if client is first:
            raise ConnectionError("Server disconnected")
        return "reconnected"

    assert instance._run_hindsight_operation(operation) == "reconnected"
    assert attempted == [first, replacement]
    assert instance._client is replacement


def test_default_nonrepeatable_operation_keeps_its_existing_no_retry_behavior(provider):
    instance, client = connected(provider)
    attempted = []

    async def operation(current):
        attempted.append(current)
        raise ConnectionError("Server disconnected")

    with pytest.raises(ConnectionError):
        instance._run_hindsight_operation(operation, repeatable=False)
    assert attempted == [client]
    assert instance._client is client


def test_checkpoint_older_host_without_normalized_evidence_protocol_is_incapable(provider, monkeypatch):
    import sys

    instance, client = connected(provider)
    monkeypatch.delattr(sys.modules["agent.memory_provider"], "PRE_COMPRESS_CHECKPOINT_API_VERSION", raising=False)
    assert instance.pre_compress_checkpoint_api_version == 1
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert client.events == []


def test_checkpoint_same_bank_session_and_evidence_stay_distinct_across_bound_profiles(provider):
    first, a = connected(provider, hermes_home="/fixture/profile-a")
    second, b = connected(provider, hermes_home="/fixture/profile-b")
    first.on_pre_compress(MESSAGES, require_checkpoint=True)
    second.on_pre_compress(MESSAGES, require_checkpoint=True)
    assert a.retains[0]["document_id"] != b.retains[0]["document_id"]


def test_checkpoint_exact_utf8_limit_is_accepted_but_one_more_byte_is_refused(provider):
    instance, client = connected(provider)
    # Canonical JSON for one user message, with its content left empty.
    overhead = len('[{"content":"","role":"user"}]'.encode("utf-8"))
    evidence = [{"role": "user", "content": "x" * (1_048_576 - overhead)}]
    instance.on_pre_compress(evidence, require_checkpoint=True)
    assert len(client.retains[0]["items"][0]["content"].encode("utf-8")) == 1_048_576
    events = list(client.events)
    evidence[0]["content"] += "x"
    with pytest.raises(RuntimeError, match="checkpoint"):
        instance.on_pre_compress(evidence, require_checkpoint=True)
    assert client.events == events

"""Optional real-Hermes contract probe, run in a fresh process by pytest."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

# The host is deliberately optional: Hermes is not a PyPI dependency of this integration.
sys.path.insert(0, os.environ["HERMES_AGENT_SOURCE"])

from agent.conversation_compression import _pre_compress_memory_context
from agent.conversation_compression_checkpoint import CompressionCheckpointUnavailable
from agent.memory_manager import MemoryManager
from hindsight_client_api.exceptions import ApiException

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "checkpoint_provider", root / "__init__.py", submodule_search_locations=[str(root)]
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


class Client:
    def __init__(self):
        self.stored = {}
        self.fail = False
        self.banks = types.SimpleNamespace(get_bank_config=self.config)
        self.documents = types.SimpleNamespace(get_document=self.get)
        self.memory = types.SimpleNamespace(retain_memories=self.retain)

    async def config(self, bank_id, **kwargs):
        return types.SimpleNamespace(bank_id=bank_id, config={"store_document_text": True})

    async def get(self, bank_id, document_id, **kwargs):
        if (bank_id, document_id) not in self.stored:
            raise ApiException(status=404)
        return types.SimpleNamespace(id=document_id, bank_id=bank_id, original_text=self.stored[(bank_id, document_id)])

    async def retain(self, bank_id, request, **kwargs):
        if self.fail:
            raise ApiException(status=402, reason="fixture-only refusal")
        item = request.items[0]
        self.stored[(bank_id, item.document_id)] = item.content.actual_instance
        return types.SimpleNamespace(success=True, var_async=False)


plugin._load_config = lambda: {"mode": "cloud", "apiKey": "fixture-only", "checkpoint_enabled": True}
plugin._check_api_supports_update_mode_append = lambda *args, **kwargs: True
client = Client()
provider = plugin.HindsightMemoryProvider()
provider._new_cloud_client = lambda: client
manager = MemoryManager()
manager.add_provider(provider)
manager.initialize_all("contract-session", hermes_home=os.environ["HERMES_HOME"], agent_identity="checkpoint-fixture")
assert not manager.supports_pre_compress_checkpoint()
provider._get_client()
assert manager.supports_pre_compress_checkpoint()

agent = types.SimpleNamespace(_memory_manager=manager)
fake_secret = "sk-" + "x" * 48
messages = [
    {"role": "system", "content": "system fixture omitted"},
    {
        "role": "user",
        "content": "direct fixture " + fake_secret,
        "display_metadata": {"withheld_prompt": "withheld fixture omitted"},
    },
    {"role": "assistant", "content": "assistant prose", "tool_calls": [{"id": "fixture"}]},
    {"role": "tool", "content": "tool fixture omitted"},
    {"role": "assistant", "content": "summary fixture omitted", "_compressed_summary": True},
]
original = json.dumps(messages)
receipt = _pre_compress_memory_context(agent, messages, checkpoint_required=True)
assert "checkpoint" in receipt
assert len(client.stored) == 1
stored = next(iter(client.stored.values()))
assert fake_secret not in stored
assert "withheld fixture omitted" not in stored
assert "system fixture omitted" not in stored
assert "tool fixture omitted" not in stored
assert "summary fixture omitted" not in stored
assert "tool_calls" not in stored
assert [message["role"] for message in json.loads(stored)] == ["user", "assistant"]
assert json.dumps(messages) == original

client.fail = True
messages.append({"role": "user", "content": "new evidence fixture"})
original = json.dumps(messages)
try:
    _pre_compress_memory_context(agent, messages, checkpoint_required=True)
except CompressionCheckpointUnavailable:
    pass
else:
    raise AssertionError("strict host gate accepted an unverified checkpoint")
assert json.dumps(messages) == original
assert len(client.stored) == 1

# The real manager must refuse a caller from another profile before touching the provider.
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

token = set_hermes_home_override(str(Path(os.environ["HERMES_HOME"]) / "other-profile"))
try:
    try:
        _pre_compress_memory_context(agent, messages, checkpoint_required=True)
    except CompressionCheckpointUnavailable:
        pass
    else:
        raise AssertionError("strict host gate accepted another profile's caller")
finally:
    reset_hermes_home_override(token)
assert len(client.stored) == 1
print("Real host: normalized redacted checkpoint succeeded; required failure preserved original evidence.")

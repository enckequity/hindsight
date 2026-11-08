"""The real SDK must send a synchronous request and recover text with a fresh client."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from hindsight_client import Hindsight

import hindsight_hermes as plugin


def test_checkpoint_real_sdk_round_trip_survives_client_recreation(provider):
    documents = {}
    writes = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith("/config"):
                self.reply(200, {"bank_id": "hermes", "config": {"store_document_text": True}, "overrides": {}})
                return
            document_id = unquote(self.path.split("/documents/", 1)[1])
            if document_id not in documents:
                self.reply(404, {"detail": "Document not found"})
                return
            text = documents[document_id]
            self.reply(
                200,
                {
                    "id": document_id,
                    "bank_id": "hermes",
                    "original_text": text,
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                    "created_at": "2026-11-08T00:00:00Z",
                    "updated_at": "2026-11-08T00:00:00Z",
                    "memory_unit_count": 0,
                },
            )

        def do_POST(self):
            assert self.path == "/v1/default/banks/hermes/memories"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            writes.append(body)
            documents[body["items"][0]["document_id"]] = body["items"][0]["content"]
            self.reply(200, {"success": True, "bank_id": "hermes", "items_count": 1, "async": False})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clients = []
    try:

        def connected():
            client = Hindsight(base_url=f"http://127.0.0.1:{server.server_port}", api_key="fixture-only", timeout=2)
            clients.append(client)
            instance, _ = provider({"checkpoint_enabled": True}, client=client)
            instance._get_client()
            return instance

        evidence = [{"role": "user", "content": "Exact café fixture"}]
        first = connected()
        receipt = first.on_pre_compress(evidence, require_checkpoint=True)
        plugin._run_sync(clients[0].aclose())
        restored = connected()
        assert restored.on_pre_compress(evidence, require_checkpoint=True) == receipt
        assert len(writes) == 1
        assert writes[0]["async"] is False
        assert "operation_id" not in writes[0]
        assert json.loads(next(iter(documents.values()))) == evidence
        assert writes[0]["items"][0]["document_id"] != first._document_id
    finally:
        for client in clients:
            plugin._run_sync(client.aclose())
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()

"""Tests OllamaClient against a tiny fake Ollama server (no real model needed).
This checks our request/response conversion, NOT how good any local model is."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from claim_agent.agent import ClaimIntakeAgent
from claim_agent.llm import OllamaClient
from claim_agent.store import Store
from claim_agent.tools import ToolBox
from claim_agent.tracing import Tracer


def start_fake_ollama(replies):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            data = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_full_agent_run_through_fake_ollama():
    replies = [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "create_claim", "arguments": {
                "policy_id": "P1001", "claim_type": "medical", "amount": 800, "description": "clinic"}}}]}},
        {"message": {"role": "assistant", "content": "Claim created."}},
    ]
    server, seen = start_fake_ollama(replies)
    try:
        client = OllamaClient(model="fake", host=f"http://127.0.0.1:{server.server_port}", timeout=5)
        store = Store()
        agent = ClaimIntakeAgent(client, ToolBox(store), Tracer(), backoff_s=0)
        result = agent.run("Claim RM800 on P1001, NRIC 900101-14-5678")
    finally:
        server.shutdown()

    assert result.status == "completed" and len(store.claims) == 1
    first, second = seen
    assert first["messages"][0]["role"] == "system"
    assert first["tools"][0]["type"] == "function" and "parameters" in first["tools"][0]["function"]
    assert "900101" not in json.dumps(first)                    # redaction still applies
    tool_msg = [m for m in second["messages"] if m["role"] == "tool"][0]
    assert tool_msg["tool_name"] == "create_claim"              # name carried onto the result


def test_arguments_as_json_string_are_parsed():
    replies = [{"message": {"content": "", "tool_calls": [
        {"function": {"name": "get_policy", "arguments": "{\"policy_id\": \"P1001\"}"}}]}}]
    server, _ = start_fake_ollama(replies)
    try:
        client = OllamaClient(model="fake", host=f"http://127.0.0.1:{server.server_port}", timeout=5)
        resp = client.complete("sys", [{"role": "user", "content": "hi"}], [])
    finally:
        server.shutdown()
    assert resp.tool_calls[0].args == {"policy_id": "P1001"}


def test_unreachable_server_gives_helpful_error():
    client = OllamaClient(model="x", host="http://127.0.0.1:9", timeout=2)
    try:
        client.complete("s", [{"role": "user", "content": "hi"}], [])
        assert False, "should have raised"
    except RuntimeError as e:
        assert "Is it running" in str(e)

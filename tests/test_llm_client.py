"""The client against real sockets: slow servers vs. unreachable ones."""

import socket
import threading

import pytest

from repo_debug_agent.config import LLMSettings
from repo_debug_agent.llm import LLMError, LLMUnavailableError, OpenAICompatibleLLM


def client(port: int, timeout: float = 0.5) -> OpenAICompatibleLLM:
    settings = LLMSettings(
        provider="local",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key=None,
        triage_model="m",
        fix_model="m",
        timeout=timeout,
        max_retries=0,
    )
    return OpenAICompatibleLLM(settings)


@pytest.fixture
def silent_server():
    """Accepts connections and never answers, like a server busy with queued work."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    held = []
    stop = threading.Event()

    def accept():
        server.settimeout(0.1)
        while not stop.is_set():
            try:
                held.append(server.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1]
    stop.set()
    thread.join()
    for conn in held:
        conn.close()
    server.close()


def test_a_slow_server_fails_the_request_but_not_the_run(silent_server):
    with pytest.raises(LLMError, match="timed out") as caught:
        client(silent_server).complete([{"role": "user", "content": "hi"}], model="m")
    assert not isinstance(caught.value, LLMUnavailableError)


def test_an_unreachable_server_stops_the_run():
    with socket.socket() as probe:  # a port with nothing listening
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(LLMUnavailableError):
        client(port).complete([{"role": "user", "content": "hi"}], model="m")


@pytest.fixture
def picky_server():
    """An OpenAI-compatible endpoint that rejects json_schema output (as many hosted
    models do) but accepts plain JSON mode. Records what each request asked for."""
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fmt = (body.get("response_format") or {}).get("type")
            seen.append(fmt)
            if fmt == "json_schema":
                payload = {"error": {"message": "response_format json_schema is not supported"}}
                status = 400
            else:
                payload = {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"issues": []}'},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                status = 200
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], seen
    server.shutdown()
    server.server_close()


def test_schema_output_falls_back_to_json_mode_and_remembers(picky_server):
    port, seen = picky_server
    llm = client(port, timeout=5)
    msgs = [{"role": "user", "content": "review"}]
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    assert llm.complete(msgs, model="m", json_mode=True, json_schema=schema) == '{"issues": []}'
    assert llm.complete(msgs, model="m", json_mode=True, json_schema=schema) == '{"issues": []}'
    # first call: schema rejected, then JSON mode; second call goes straight to JSON mode
    assert seen == ["json_schema", "json_object", "json_object"]

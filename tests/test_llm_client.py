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

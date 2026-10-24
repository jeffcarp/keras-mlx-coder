from fastapi.testclient import TestClient
import serve_keras_opencode
from serve_keras_opencode import app

client = TestClient(app)


def test_list_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    ids = [m["id"] for m in payload["data"]]
    assert serve_keras_opencode.active_preset in ids
    assert "default" in ids


def test_chat_completions_mocked(monkeypatch):
    async def mock_generate_stream(prompt, max_tokens=1024):
        yield "def "
        yield "hello():\n"
        yield "    return True"

    monkeypatch.setattr(serve_keras_opencode, "generate_stream", mock_generate_stream)

    # Test non-streaming completion
    payload = {
        "model": "qwen2.5-coder",
        "messages": [{"role": "user", "content": "Write hello world."}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "def hello():\n    return True"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming_mocked(monkeypatch):
    async def mock_generate_stream(prompt, max_tokens=1024):
        yield "def "
        yield "hello():"

    monkeypatch.setattr(serve_keras_opencode, "generate_stream", mock_generate_stream)

    payload = {
        "model": "qwen2.5-coder",
        "messages": [{"role": "user", "content": "Write a func."}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = [line for line in response.text.split("\n\n") if line.strip()]
    assert any("[DONE]" in line for line in lines)

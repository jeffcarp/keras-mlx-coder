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


def test_compute_sequence_length():
    # Small prompt gets bucketed to 1024
    assert serve_keras_opencode.compute_sequence_length(50, 100, 4096) == 1024
    # Medium prompt scales up to nearest 1024 bucket
    assert serve_keras_opencode.compute_sequence_length(1500, 1000, 4096) == 3072
    # Large prompt + max_tokens capped at max_sequence_length
    assert serve_keras_opencode.compute_sequence_length(2000, 16384, 4096) == 4096
    # Exact ceiling
    assert serve_keras_opencode.compute_sequence_length(5000, 5000, 8192) == 8192


def test_chat_completions_null_max_tokens(monkeypatch):
    captured_max_tokens = []

    async def mock_generate_stream(prompt, max_tokens=1024):
        captured_max_tokens.append(max_tokens)
        yield "def foo(): pass"

    monkeypatch.setattr(serve_keras_opencode, "generate_stream", mock_generate_stream)

    payload = {
        "model": "qwen2.5-coder",
        "messages": [{"role": "user", "content": "Write a func."}],
        "max_tokens": None,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert captured_max_tokens == [1024]


def test_patch_kerashub_bpe_tokenizer():
    serve_keras_opencode._patch_kerashub_bpe_tokenizer()
    try:
        from keras_hub.src.tokenizers import byte_pair_tokenizer
        import tensorflow as tf

        unsplittable = ["<|vision_end|>", "<|im_end|>", "<|im_start|>"]
        test_str = tf.constant(
            ["<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"]
        )
        split_result = byte_pair_tokenizer.split_strings_for_bpe(test_str, unsplittable)
        tokens = [t.decode("utf-8") for t in split_result.flat_values.numpy()]
        assert "<|im_start|>" in tokens
        assert "<|im_end|>" in tokens
        assert "<|vision_end|>H" not in tokens
        assert "<|vision_end|>D" not in tokens
    except ImportError:
        pass


def test_extract_tool_calls():
    raw = (
        "Checking src directory:\n"
        '<tool_call>\n{"name": "list_dir", "arguments": {"path": "src"}}\n</tool_call>'
    )
    clean, tool_calls = serve_keras_opencode.extract_tool_calls(raw)
    assert clean == "Checking src directory:"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "list_dir"
    assert '"path": "src"' in tool_calls[0]["function"]["arguments"]


def test_chat_completions_tool_calling(monkeypatch):
    async def mock_generate_stream(prompt, max_tokens=1024):
        yield "Checking directory\n"
        yield '<tool_call>\n{"name": "list_files", "arguments": {"dir": "src"}}\n</tool_call>'

    monkeypatch.setattr(serve_keras_opencode, "generate_stream", mock_generate_stream)

    payload = {
        "model": "qwen2.5-coder",
        "messages": [{"role": "user", "content": "List files."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "parameters": {
                        "type": "object",
                        "properties": {"dir": {"type": "string"}},
                    },
                },
            }
        ],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert len(choice["message"]["tool_calls"]) == 1
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "list_files"


def test_chat_completions_tool_calling_streaming(monkeypatch):
    async def mock_generate_stream(prompt, max_tokens=1024):
        yield "Checking\n"
        yield '<tool_call>\n{"name": "list_files", "arguments": {"dir": "src"}}\n</tool_call>'

    monkeypatch.setattr(serve_keras_opencode, "generate_stream", mock_generate_stream)

    payload = {
        "model": "qwen2.5-coder",
        "messages": [{"role": "user", "content": "List files."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "parameters": {
                        "type": "object",
                        "properties": {"dir": {"type": "string"}},
                    },
                },
            }
        ],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "tool_calls" in response.text

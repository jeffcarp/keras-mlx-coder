import importlib
from fastapi.testclient import TestClient

serve_module = importlib.import_module("examples.6_serve_qwen_opencode")
format_chatml = serve_module.format_chatml
app = serve_module.app


def test_format_chatml_single_message():
    messages = [{"role": "user", "content": "Write hello world in Python"}]
    expected = (
        "<|im_start|>user\nWrite hello world in Python<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert format_chatml(messages) == expected


def test_format_chatml_multi_turn():
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Add 3 to it."},
    ]
    expected = (
        "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4<|im_end|>\n"
        "<|im_start|>user\nAdd 3 to it.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert format_chatml(messages) == expected


def test_list_models_endpoint():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "qwen2.5-coder" in model_ids
    assert "qwen2.5-coder-1.5b" in model_ids

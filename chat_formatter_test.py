from chat_formatter import (
    ChatMLFormatter,
    GemmaFormatter,
    Llama3Formatter,
    PlainFormatter,
    get_chat_formatter,
)


def test_chatml_formatting():
    formatter = ChatMLFormatter()
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a function."},
    ]
    result = formatter.format_chat(messages, add_generation_prompt=True)
    expected = (
        "<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n"
        "<|im_start|>user\nWrite a function.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert result == expected


def test_chatml_formatting_no_generation_prompt():
    formatter = ChatMLFormatter()
    messages = [{"role": "user", "content": "Hi"}]
    result = formatter.format_chat(messages, add_generation_prompt=False)
    assert result == "<|im_start|>user\nHi<|im_end|>\n"


def test_gemma_formatting():
    formatter = GemmaFormatter()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "Next step?"},
    ]
    result = formatter.format_chat(messages, add_generation_prompt=True)
    expected = (
        "<start_of_turn>user\nBe concise.\n\nHello<end_of_turn>\n"
        "<start_of_turn>model\nHi there.<end_of_turn>\n"
        "<start_of_turn>user\nNext step?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    assert result == expected


def test_llama3_formatting():
    formatter = Llama3Formatter()
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User prompt."},
    ]
    result = formatter.format_chat(messages, add_generation_prompt=True)
    expected = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\nSystem prompt.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nUser prompt.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert result == expected


def test_plain_formatting():
    formatter = PlainFormatter()
    messages = [
        {"role": "user", "content": "Ping"},
    ]
    result = formatter.format_chat(messages, add_generation_prompt=True)
    assert result == "User: Ping\n\nAssistant: "


def test_get_chat_formatter_auto_detection():
    # Direct keys
    assert isinstance(get_chat_formatter("chatml"), ChatMLFormatter)
    assert isinstance(get_chat_formatter("gemma"), GemmaFormatter)
    assert isinstance(get_chat_formatter("llama3"), Llama3Formatter)
    assert isinstance(get_chat_formatter("plain"), PlainFormatter)

    # Preset names
    assert isinstance(
        get_chat_formatter("qwen2.5_coder_instruct_1.5b"), ChatMLFormatter
    )
    assert isinstance(
        get_chat_formatter("deepseek-coder-6.7b-instruct"), ChatMLFormatter
    )
    assert isinstance(get_chat_formatter("hf://google/gemma-3-1b-it"), GemmaFormatter)
    assert isinstance(get_chat_formatter("llama_3_instruct_8b"), Llama3Formatter)

    # Empty / unknown defaults to chatml
    assert isinstance(get_chat_formatter(None), ChatMLFormatter)
    assert isinstance(get_chat_formatter("unknown_custom_model"), ChatMLFormatter)


def test_chatml_formatting_with_tools():
    formatter = ChatMLFormatter()
    messages = [
        {"role": "user", "content": "List files in src/."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files in directory",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    result = formatter.format_chat(messages, tools=tools, add_generation_prompt=True)
    assert "<|im_start|>system" in result
    assert "# Tools" in result
    assert "<tools>" in result
    assert "list_dir" in result
    assert "<tool_call>" in result
    assert "<|im_start|>user\nList files in src/.<|im_end|>\n" in result
    assert result.endswith("<|im_start|>assistant\n")


def test_chatml_formatting_with_tool_calls_and_responses():
    formatter = ChatMLFormatter()
    messages = [
        {"role": "user", "content": "List files."},
        {
            "role": "assistant",
            "content": "Checking...",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": '{"path": "src"}'},
                }
            ],
        },
        {"role": "tool", "content": '["main.py", "utils.py"]'},
    ]
    result = formatter.format_chat(messages, add_generation_prompt=True)
    assert (
        '<tool_call>\n{"name": "list_dir", "arguments": {"path": "src"}}\n</tool_call>'
        in result
    )
    assert '<tool_response>\n["main.py", "utils.py"]\n</tool_response>' in result
    assert result.endswith("<|im_start|>assistant\n")


def test_stop_tokens_resolution():
    class MockTokenizer:
        end_token_id = 999
        end_token2_id = 998

        def token_to_id(self, token):
            lookup = {"<|im_end|>": 151645, "<|endoftext|>": 151643}
            return lookup.get(token)

    formatter = ChatMLFormatter()
    stop_ids = formatter.get_stop_token_ids(MockTokenizer())
    assert 999 in stop_ids
    assert 998 in stop_ids
    assert 151645 in stop_ids
    assert 151643 in stop_ids


def test_multipart_content_formatting():
    formatter = ChatMLFormatter()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello world"}]}]
    result = formatter.format_chat(messages, add_generation_prompt=True)
    assert "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n" == result

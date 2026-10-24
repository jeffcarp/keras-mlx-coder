"""Chat template formatting for various open-source LLM families.

Formats OpenAI-style message lists ([{"role": "...", "content": "..."}]) into
the target model's prompt representation (ChatML, Gemma, Llama 3, etc.) and
provides stop-token resolution.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set


class ChatFormatter(ABC):
    """Base class for model chat formatting."""

    name: str = "base"
    stop_strings: List[str] = []

    @abstractmethod
    def format_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Formats a list of message dicts into a prompt string."""
        pass

    def get_stop_token_ids(self, tokenizer: Any) -> Set[int]:
        """Resolves stop token IDs from the tokenizer."""
        stop_ids: Set[int] = set()

        # Primary tokenizer end tokens
        if hasattr(tokenizer, "end_token_id") and tokenizer.end_token_id is not None:
            stop_ids.add(tokenizer.end_token_id)
        if hasattr(tokenizer, "end_token2_id") and tokenizer.end_token2_id is not None:
            stop_ids.add(tokenizer.end_token2_id)

        # Lookup stop string IDs if tokenizer has token_to_id
        if hasattr(tokenizer, "token_to_id"):
            for s in self.stop_strings:
                try:
                    tid = tokenizer.token_to_id(s)
                    if tid is not None:
                        stop_ids.add(tid)
                except Exception:
                    pass

        return stop_ids


class ChatMLFormatter(ChatFormatter):
    """ChatML format used by Qwen 2.5, DeepSeek, Yi, etc.

    Example:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
    """

    name = "chatml"
    stop_strings = ["<|im_end|>", "<|endoftext|>"]

    def format_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        formatted = ""
        has_system = bool(messages and messages[0].get("role") == "system")
        system_content = (
            messages[0].get("content", "")
            if has_system
            else "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )

        if tools:
            tool_signatures = "\n".join(json.dumps(t) for t in tools)
            system_prompt = (
                f"{system_content}\n\n"
                f"# Tools\n\n"
                f"You may call one or more functions to assist with the user query.\n\n"
                f"You are provided with function signatures within <tools></tools> XML tags:\n"
                f"<tools>\n{tool_signatures}\n</tools>\n\n"
                f"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                f"<tool_call>\n"
                f'{{"name": <function-name>, "arguments": <args-json-object>}}\n'
                f"</tool_call>"
            )
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            start_idx = 1 if has_system else 0
        elif has_system:
            formatted += f"<|im_start|>system\n{system_content}<|im_end|>\n"
            start_idx = 1
        else:
            start_idx = 0

        for msg in messages[start_idx:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "tool":
                formatted += f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            elif role == "assistant" and tool_calls:
                call_strs = []
                if content:
                    call_strs.append(content)
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = str(args)
                    call_strs.append(
                        f'<tool_call>\n{{"name": "{fn.get("name")}", "arguments": {args_str}}}\n</tool_call>'
                    )
                formatted += (
                    f"<|im_start|>assistant\n{chr(10).join(call_strs)}<|im_end|>\n"
                )
            else:
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        return formatted

    def get_stop_token_ids(self, tokenizer: Any) -> Set[int]:
        stop_ids = super().get_stop_token_ids(tokenizer)
        # Known standard Qwen / ChatML token IDs for safety
        stop_ids.update([151643, 151645])
        return stop_ids


class GemmaFormatter(ChatFormatter):
    """Gemma turn format used by Gemma, Gemma 2, and Gemma 3.

    Example:
        <start_of_turn>user
        System: You are helpful.

        Hello<end_of_turn>
        <start_of_turn>model
    """

    name = "gemma"
    stop_strings = ["<end_of_turn>", "<eos>"]

    def format_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        formatted = ""
        system_content = ""

        # Gemma doesn't have a native system role; prepend system prompt to first user turn
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content += f"{content}\n\n"
            elif role == "user":
                turn_content = (
                    f"{system_content}{content}" if system_content else content
                )
                system_content = ""  # Only prepend to the first user message
                formatted += f"<start_of_turn>user\n{turn_content}<end_of_turn>\n"
            elif role in ("assistant", "model"):
                formatted += f"<start_of_turn>model\n{content}<end_of_turn>\n"

        if add_generation_prompt:
            formatted += "<start_of_turn>model\n"
        return formatted

    def get_stop_token_ids(self, tokenizer: Any) -> Set[int]:
        stop_ids = super().get_stop_token_ids(tokenizer)
        # Gemma standard end of turn token ID
        stop_ids.update([1, 107])
        return stop_ids


class Llama3Formatter(ChatFormatter):
    """Llama 3 header format.

    Example:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    name = "llama3"
    stop_strings = ["<|eot_id|>", "<|end_of_text|>"]

    def format_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        formatted = "<|begin_of_text|>"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        if add_generation_prompt:
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted

    def get_stop_token_ids(self, tokenizer: Any) -> Set[int]:
        stop_ids = super().get_stop_token_ids(tokenizer)
        # Llama 3 standard eot token ID
        stop_ids.update([128001, 128009])
        return stop_ids


class PlainFormatter(ChatFormatter):
    """Plain-text conversational fallback."""

    name = "plain"
    stop_strings = ["\nUser:", "\nSystem:"]

    def format_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            formatted += f"{role}: {content}\n\n"

        if add_generation_prompt:
            formatted += "Assistant: "
        return formatted


FORMATTERS: Dict[str, ChatFormatter] = {
    "chatml": ChatMLFormatter(),
    "gemma": GemmaFormatter(),
    "llama3": Llama3Formatter(),
    "plain": PlainFormatter(),
}


def get_chat_formatter(preset_or_name: Optional[str] = None) -> ChatFormatter:
    """Returns the appropriate ChatFormatter based on format name or preset string.

    Args:
        preset_or_name: Format name ('chatml', 'gemma', 'llama3') or KerasHub
            preset name (e.g. 'qwen2.5_coder_instruct_1.5b', 'gemma3_instruct_1b').
            Defaults to 'chatml' if unspecified or unrecognized.
    """
    if not preset_or_name:
        return FORMATTERS["chatml"]

    key = preset_or_name.lower().strip()

    # Exact format name match
    if key in FORMATTERS:
        return FORMATTERS[key]

    # Auto-detect from preset name
    if any(q in key for q in ("qwen", "deepseek", "yi", "chatml")):
        return FORMATTERS["chatml"]
    if any(g in key for g in ("gemma",)):
        return FORMATTERS["gemma"]
    if any(l_name in key for l_name in ("llama", "codellama")):
        return FORMATTERS["llama3"]

    # Fallback to chatml as standard for modern open models
    return FORMATTERS["chatml"]

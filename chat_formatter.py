"""Chat template formatting for various open-source LLM families.

Formats OpenAI-style message lists ([{"role": "...", "content": "..."}]) into
the target model's prompt representation (ChatML, Gemma, Llama 3, etc.) and
provides stop-token resolution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set


class ChatFormatter(ABC):
    """Base class for model chat formatting."""

    name: str = "base"
    stop_strings: List[str] = []

    @abstractmethod
    def format_chat(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
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
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
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
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
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
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
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
        self, messages: List[Dict[str, str]], add_generation_prompt: bool = True
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

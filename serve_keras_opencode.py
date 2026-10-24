"""OpenAI-Compatible Local API Server for SOTA Coding Models (KerasHub + MLX).

Exposes an OpenAI-compatible streaming /v1/chat/completions endpoint
accelerated on Apple Silicon GPU via KerasHub and the Keras-MLX pluggable
backend with fused Metal kernels (mx.compile).

Supports any KerasHub CausalLM preset (Qwen 2.5 Coder, Gemma, Llama, etc.).
"""

import argparse
import asyncio
import atexit
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from chat_formatter import ChatFormatter, get_chat_formatter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Global state for server
active_preset: str = os.environ.get("MODEL_PRESET", "qwen2.5_coder_instruct_1.5b")
active_format: Optional[str] = os.environ.get("CHAT_FORMAT", None)
active_max_sequence_length: int = int(os.environ.get("MAX_SEQUENCE_LENGTH", 4096))
model: Any = None
tokenizer: Any = None
formatter: Optional[ChatFormatter] = None
stop_token_ids: Set[int] = set()


def extract_tool_calls(
    text: str, valid_tool_names: Optional[Set[str]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extracts tool calls from XML tags, code blocks, or raw JSON objects."""
    raw_calls = []
    clean_text = text

    # 1. XML <tool_call> ... </tool_call>
    tc_matches = list(
        re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    )
    if tc_matches:
        for m in tc_matches:
            try:
                data = json.loads(m.group(1).strip())
                if isinstance(data, dict) and "name" in data:
                    raw_calls.append(data)
            except Exception:
                pass
        clean_text = re.sub(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", "", clean_text, flags=re.DOTALL
        ).strip()

    # 2. Markdown ```json ... ``` blocks containing name & arguments
    if not raw_calls:
        code_matches = list(
            re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        )
        for m in code_matches:
            try:
                data = json.loads(m.group(1).strip())
                if (
                    isinstance(data, dict)
                    and "name" in data
                    and ("arguments" in data or "parameters" in data)
                ):
                    raw_calls.append(data)
            except Exception:
                pass
        if raw_calls:
            clean_text = re.sub(
                r"```(?:json)?\s*(\{.*?\})\s*```", "", clean_text, flags=re.DOTALL
            ).strip()

    # 3. Raw JSON object
    if not raw_calls:
        raw_match = re.search(
            r"^\s*(\{.*\"name\"\s*:.*\"arguments\"\s*:.*\})\s*$", text, re.DOTALL
        )
        if raw_match:
            try:
                data = json.loads(raw_match.group(1).strip())
                if isinstance(data, dict) and "name" in data:
                    raw_calls.append(data)
                    clean_text = ""
            except Exception:
                pass

    tool_calls = []
    for call in raw_calls:
        name = call.get("name")
        if valid_tool_names and name not in valid_tool_names:
            continue
        args = call.get("arguments", call.get("parameters", {}))
        args_str = json.dumps(args) if isinstance(args, dict) else str(args or "{}")
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:9]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_str,
                },
            }
        )

    return clean_text, tool_calls


def compute_sequence_length(
    prompt_tokens: int,
    max_tokens: int,
    max_sequence_length: int = 4096,
) -> int:
    """Computes bucketed sequence length capped at max_sequence_length to avoid OOM."""
    needed = min(max(prompt_tokens + max_tokens + 64, 1024), max_sequence_length)
    return min(((needed + 1023) // 1024) * 1024, max_sequence_length)


app = FastAPI(title="KerasHub MLX Local OpenAI Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _register_tf_mlx_converter():
    """Registers mlx.core.array converter for TensorFlow to support KerasHub tokenizers."""
    try:
        import mlx.core as mx
        import numpy as np
        import tensorflow as tf

        tf.register_tensor_conversion_function(
            mx.array,
            lambda val, dtype=None, name=None, as_ref=False: tf.convert_to_tensor(
                np.array(val), dtype=dtype, name=name
            ),
        )
    except Exception:
        pass


def _patch_kerashub_bpe_tokenizer():
    """Patches KerasHub BytePairTokenizer regex replacement order for special tokens.

    KerasHub assigns prefix-overlapping alternates to unsplittable tokens (e.g.
    'ĴB' for '<|vision_end|>', 'ĴBD' for '<|im_end|>', 'ĴBH' for '<|im_start|>').
    Without sorting alternates longest-first on the reverse replacement pass,
    'ĴB' matches inside 'ĴBH' and 'ĴBD', corrupting '<|im_start|>' into '<|vision_end|>H'
    and '<|im_end|>' into '<|vision_end|>D'. Sorting alternates longest-first resolves this.
    """
    try:
        import re
        from keras_hub.src.tokenizers import byte_pair_tokenizer
        import tensorflow as tf

        orig_split = byte_pair_tokenizer.split_strings_for_bpe

        def patched_split_strings_for_bpe(inputs, unsplittable_tokens=None):
            if not unsplittable_tokens:
                return orig_split(inputs, unsplittable_tokens)
            alts = byte_pair_tokenizer.create_alts_for_unsplittable_tokens(
                unsplittable_tokens
            )
            token_pairs = sorted(
                zip(unsplittable_tokens, alts), key=lambda x: len(x[0]), reverse=True
            )
            alt_pairs = sorted(
                zip(unsplittable_tokens, alts), key=lambda x: len(x[1]), reverse=True
            )
            for token, alt in token_pairs:
                escaped_token = re.escape(token)
                inputs = byte_pair_tokenizer.tf_text.regex_split(
                    inputs, escaped_token, escaped_token
                )
                inputs = tf.strings.regex_replace(inputs, escaped_token, alt)
            raw_tokens = byte_pair_tokenizer.tf_text.regex_split(
                inputs,
                byte_pair_tokenizer.SPLIT_PATTERN_1,
                byte_pair_tokenizer.SPLIT_PATTERN_1,
            )
            raw_tokens = byte_pair_tokenizer.tf_text.regex_split(
                raw_tokens,
                byte_pair_tokenizer.SPLIT_PATTERN_2,
                byte_pair_tokenizer.SPLIT_PATTERN_2,
            )
            for token, alt in alt_pairs:
                escaped_alt = re.escape(alt)
                raw_tokens = tf.strings.regex_replace(raw_tokens, escaped_alt, token)
            while raw_tokens.shape.rank > 2:
                raw_tokens = raw_tokens.merge_dims(1, 2)
            return byte_pair_tokenizer.remove_strings_from_inputs(raw_tokens, "६")

        byte_pair_tokenizer.split_strings_for_bpe = patched_split_strings_for_bpe
    except Exception:
        pass


def init_model(
    preset: Optional[str] = None,
    chat_format: Optional[str] = None,
    dtype: str = "bfloat16",
    max_sequence_length: Optional[int] = None,
):
    """Initializes, compiles, and configures the KerasHub CausalLM model."""
    global \
        active_preset, \
        active_format, \
        active_max_sequence_length, \
        model, \
        tokenizer, \
        formatter, \
        stop_token_ids

    if model is not None:
        return model

    # Ensure backend is configured
    os.environ["KERAS_BACKEND"] = "mlx"
    _register_tf_mlx_converter()
    _patch_kerashub_bpe_tokenizer()
    import keras  # noqa: E402
    import keras_hub  # noqa: E402
    import mlx.core as mx  # noqa: E402

    if hasattr(mx, "clear_streams"):
        atexit.register(mx.clear_streams)
    if hasattr(mx, "compile_clear_cache"):
        atexit.register(mx.compile_clear_cache)

    active_preset = preset or active_preset
    active_format = chat_format or active_format
    if max_sequence_length is not None:
        active_max_sequence_length = max_sequence_length

    print("=" * 65)
    print(f"Starting KerasHub Local Server with Model: {active_preset}")
    print("=" * 65)
    print(f"Keras Backend: {keras.config.backend()}")
    print(f"MLX Device:    {mx.default_device()}")
    print(f"Dtype:         {dtype}")

    print(f"\nLoading model from preset '{active_preset}'...")
    t0 = time.time()
    model = keras_hub.models.CausalLM.from_preset(active_preset, dtype=dtype)
    print(f"Model loaded in {time.time() - t0:.2f}s!")
    print(f"Total Parameters: {model.count_params():,}")

    # Compile the single-token forward step to fuse Metal compute kernels
    print("Compiling call_with_cache with mx.compile for kernel fusion...")
    model.call_with_cache = mx.compile(model.call_with_cache)

    tokenizer = model.preprocessor.tokenizer
    formatter = get_chat_formatter(active_format or active_preset)
    stop_token_ids = formatter.get_stop_token_ids(tokenizer)
    print(f"Chat Formatter:   {formatter.name}")
    print(f"Stop Token IDs:   {len(stop_token_ids)} tokens registered")

    return model


async def generate_stream(prompt: str, max_tokens: int = 1024):
    """Generates tokens streamingly via compiled Keras-MLX decoding on the event loop."""
    import keras  # noqa: E402
    import mlx.core as mx  # noqa: E402

    init_model()

    prompt_tokens = len(tokenizer(prompt)) if tokenizer else len(prompt.split())
    seq_len = compute_sequence_length(
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        max_sequence_length=active_max_sequence_length,
    )

    inputs = model.preprocessor.generate_preprocess([prompt], sequence_length=seq_len)
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]

    hidden_states, cache = model._build_cache(token_ids)
    prompt_length = int(keras.ops.sum(keras.ops.cast(padding_mask, "int32")).item())

    current_prompt = token_ids
    current_cache = cache
    index = prompt_length
    max_len = min(prompt_length + max_tokens, seq_len)

    while index < max_len:
        cache_update_index = index - 1
        prompt_slice = keras.ops.slice(current_prompt, [0, cache_update_index], [1, 1])
        idx_arr = mx.array(cache_update_index, dtype=mx.int32)

        logits, _, current_cache = model.call_with_cache(
            prompt_slice,
            current_cache,
            idx_arr,
        )
        mx.eval(logits, current_cache)

        next_token = int(mx.argmax(logits[:, 0, :], axis=-1).item())
        if next_token in stop_token_ids:
            break

        piece = tokenizer.detokenize([next_token])
        if (
            piece
            and not piece.startswith("<|")
            and not piece.startswith("<start_")
            and not piece.startswith("<end_")
        ):
            yield piece
            # Yield to event loop to stream chunk immediately and handle disconnects
            await asyncio.sleep(0)

        token_tensor = mx.reshape(
            mx.array(next_token, dtype=current_prompt.dtype), (1, 1)
        )
        current_prompt = keras.ops.slice_update(
            current_prompt, [0, index], token_tensor
        )
        index += 1


async def warmup_model():
    """Warms up the compiled Metal graph with a short dummy prompt."""
    print("Warming up compiled graph...")
    dummy_prompt = formatter.format_chat([{"role": "user", "content": "Hello"}])
    async for _ in generate_stream(dummy_prompt, max_tokens=5):
        pass
    print("Server ready to accept requests!\n")


@app.get("/v1/models")
async def list_models():
    """Returns available model descriptors matching the loaded preset."""
    created_time = int(time.time())
    active_id = active_preset
    aliases = [active_id, "default", "qwen2.5-coder"]
    unique_ids = list(dict.fromkeys(aliases))

    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": created_time,
                "owned_by": "keras-mlx",
            }
            for mid in unique_ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completion endpoint supporting SSE streaming and tool calling."""
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    stream = body.get("stream", False)
    raw_max_tokens = body.get("max_tokens")
    max_tokens = int(raw_max_tokens) if raw_max_tokens is not None else 1024
    model_name = body.get("model", active_preset)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    active_formatter = formatter or get_chat_formatter(active_format or active_preset)
    prompt = active_formatter.format_chat(
        messages, tools=tools, add_generation_prompt=True
    )
    valid_tool_names = (
        {
            t["function"]["name"]
            for t in tools
            if "function" in t and "name" in t["function"]
        }
        if tools
        else None
    )

    if stream:

        async def event_stream():
            try:
                full_chunks = []

                if not tools:
                    async for chunk in generate_stream(prompt, max_tokens=max_tokens):
                        full_chunks.append(chunk)
                        payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"

                    done_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # When tools are provided, buffer chunks to determine if output is text or tool call
                async for chunk in generate_stream(prompt, max_tokens=max_tokens):
                    full_chunks.append(chunk)

                combined = "".join(full_chunks)
                clean_text, tool_calls = extract_tool_calls(
                    combined, valid_tool_names=valid_tool_names
                )

                if clean_text:
                    text_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": clean_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(text_payload)}\n\n"

                if tool_calls:
                    for tc_idx, tc in enumerate(tool_calls):
                        tc_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "index": tc_idx,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": tc["function"],
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tc_payload)}\n\n"

                    done_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                else:
                    done_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"

                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                # Client disconnected or cancelled request cleanly
                pass
            except Exception as e:
                print(f"Error during streaming generation: {e}")
                err_payload = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(err_payload)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    chunks = []
    async for chunk in generate_stream(prompt, max_tokens=max_tokens):
        chunks.append(chunk)
    full_content = "".join(chunks)

    clean_content, tool_calls = extract_tool_calls(
        full_content, valid_tool_names=valid_tool_names
    )
    finish_reason = "tool_calls" if tool_calls else "stop"

    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": clean_content if (clean_content or not tool_calls) else None,
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls

    prompt_tok_count = len(tokenizer(prompt)) if tokenizer else len(prompt.split())
    return JSONResponse(
        {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tok_count,
                "completion_tokens": len(chunks),
                "total_tokens": prompt_tok_count + len(chunks),
            },
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="KerasHub MLX Local OpenAI-Compatible Server for Coding Models"
    )
    parser.add_argument(
        "--preset",
        default=os.environ.get("MODEL_PRESET", "qwen2.5_coder_instruct_1.5b"),
        help="KerasHub model preset (e.g. qwen2.5_coder_instruct_1.5b, qwen2.5_coder_instruct_7b)",
    )
    parser.add_argument(
        "--chat-format",
        default=os.environ.get("CHAT_FORMAT", None),
        choices=["chatml", "gemma", "llama3", "plain", None],
        help="Chat template format (default: auto-detected from model preset)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Compute dtype for model weights (default: bfloat16)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=int(os.environ.get("MAX_SEQUENCE_LENGTH", 4096)),
        help="Maximum sequence length buffer cap (default: 4096). Caps Metal memory allocation.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host address to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to listen on (default: 8000)",
    )
    args = parser.parse_args()

    init_model(
        preset=args.preset,
        chat_format=args.chat_format,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
    )
    asyncio.run(warmup_model())
    print(f"Listening on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

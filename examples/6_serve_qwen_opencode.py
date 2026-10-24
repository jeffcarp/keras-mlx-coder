"""OpenAI-Compatible Local API Server for Qwen 2.5 Coder (KerasHub + MLX).

Exposes a streaming /v1/chat/completions endpoint accelerated on Apple Silicon GPU
via KerasHub and the Keras-MLX pluggable backend with fused Metal kernels.
Enables local coding agents (OpenCode, Aider, Continue) to drive local models.
"""

import asyncio
import atexit
import json
import os
import time
import uuid

os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402
import keras_hub  # noqa: E402
import mlx.core as mx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
import uvicorn  # noqa: E402

if hasattr(mx, "clear_streams"):
    atexit.register(mx.clear_streams)
if hasattr(mx, "compile_clear_cache"):
    atexit.register(mx.compile_clear_cache)

model = None
tokenizer = None
stop_token_ids = set()


def init_model(preset=None):
    """Initializes and compiles the model if not already loaded."""
    global model, tokenizer, stop_token_ids
    if model is not None:
        return model

    preset = preset or os.environ.get("MODEL_PRESET", "qwen2.5_coder_instruct_1.5b")
    print("=" * 65)
    print("Starting KerasHub Qwen 2.5 Coder API Server (MLX Backend)")
    print("=" * 65)
    print(f"Keras Backend: {keras.config.backend()}")
    print(f"MLX Device:    {mx.default_device()}")

    print(f"\nLoading model from preset '{preset}' (dtype=bfloat16)...")
    t0 = time.time()
    model = keras_hub.models.Qwen2CausalLM.from_preset(preset, dtype="bfloat16")
    print(f"Model loaded in {time.time() - t0:.2f}s!")
    print(f"Total Parameters: {model.count_params():,}")

    # Compile single-token decoding forward step for kernel fusion
    print("Compiling call_with_cache with mx.compile for kernel fusion...")
    model.call_with_cache = mx.compile(model.call_with_cache)

    tokenizer = model.preprocessor.tokenizer
    stop_token_ids = {tokenizer.end_token_id}
    if hasattr(tokenizer, "end_token2_id") and tokenizer.end_token2_id is not None:
        stop_token_ids.add(tokenizer.end_token2_id)
    # Qwen special end tokens: <|im_end|> (151645) and <|endoftext|> (151643)
    for special_id in [151643, 151645]:
        stop_token_ids.add(special_id)

    return model


app = FastAPI(title="KerasHub Qwen Local Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_chatml(messages):
    """Formats an OpenAI messages list into Qwen ChatML format."""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


async def generate_stream(prompt, max_tokens=1024):
    """Generates tokens streamingly via compiled Keras-MLX decoding on the event loop."""
    init_model()
    prompt_tokens = len(tokenizer(prompt))
    # Dynamically scale buffer to fit prompt + max_tokens up to 16k, bucketed to 1024
    needed_len = min(max(prompt_tokens + max_tokens + 64, 2048), 16384)
    seq_len = ((needed_len + 1023) // 1024) * 1024

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

        next_token = int(mx.argmax(logits[:, 0, :], axis=-1).item())
        if next_token in stop_token_ids or next_token >= 151643:
            break

        piece = tokenizer.detokenize([next_token])
        if piece and not piece.startswith("<|"):
            yield piece
            # Yield to event loop to stream chunk immediately and handle disconnects
            await asyncio.sleep(0)

        token_tensor = mx.reshape(mx.array(next_token, dtype=current_prompt.dtype), (1, 1))
        current_prompt = keras.ops.slice_update(current_prompt, [0, index], token_tensor)
        index += 1


async def warmup_model():
    """Warms up the compiled Metal graph with a short dummy prompt."""
    print("Warming up compiled graph...")
    async for _ in generate_stream(
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", max_tokens=5
    ):
        pass
    print("Server ready to accept requests!\n")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-coder",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kerashub-mlx",
            },
            {
                "id": "qwen2.5-coder-1.5b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kerashub-mlx",
            },
            {
                "id": "qwen2.5-coder-7b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "kerashub-mlx",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 1024)
    model_name = body.get("model", "qwen2.5-coder")
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    prompt = format_chatml(messages)

    if stream:
        async def event_stream():
            try:
                async for chunk in generate_stream(prompt, max_tokens=max_tokens):
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

                # Final finish chunk
                payload_done = {
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
                yield f"data: {json.dumps(payload_done)}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                # Client disconnected or cancelled request cleanly
                pass

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    chunks = []
    async for chunk in generate_stream(prompt, max_tokens=max_tokens):
        chunks.append(chunk)
    full_content = "".join(chunks)

    prompt_tok_count = len(tokenizer(prompt))
    return JSONResponse(
        {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tok_count,
                "completion_tokens": len(chunks),
                "total_tokens": prompt_tok_count + len(chunks),
            },
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KerasHub Qwen 2.5 Coder Local API Server")
    parser.add_argument(
        "--host", default=os.environ.get("HOST", "127.0.0.1"), help="Host address to bind"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 8000)),
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--preset",
        default=os.environ.get("MODEL_PRESET", "qwen2.5_coder_instruct_1.5b"),
        help="KerasHub model preset (e.g. qwen2.5_coder_instruct_1.5b, qwen2.5_coder_instruct_7b)",
    )
    args = parser.parse_args()

    os.environ["MODEL_PRESET"] = args.preset
    init_model(args.preset)
    asyncio.run(warmup_model())
    print(f"Listening on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port)

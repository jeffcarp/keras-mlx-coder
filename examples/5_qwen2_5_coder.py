"""Real-time streaming text generation with Qwen 2.5 Coder on Keras MLX Backend.

Compiles single-token decoding steps via mx.compile for Metal kernel fusion,
achieving high tokens/sec generation on Apple Silicon GPUs.
"""

import argparse
import os
import sys
import time

os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402
import keras_hub  # noqa: E402
import mlx.core as mx  # noqa: E402


def generate_stream(model, prompt, max_length=120):
    """Generates and yields text token-by-token in real time."""
    tokenizer = model.preprocessor.tokenizer
    inputs = model.preprocessor.generate_preprocess([prompt], sequence_length=max_length)
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]

    hidden_states, cache = model._build_cache(token_ids)
    prompt_length = int(keras.ops.sum(keras.ops.cast(padding_mask, "int32")).item())

    stop_token_ids = {tokenizer.end_token_id}
    if hasattr(tokenizer, "end_token2_id") and tokenizer.end_token2_id is not None:
        stop_token_ids.add(tokenizer.end_token2_id)
    for special_id in [151643, 151645]:
        stop_token_ids.add(special_id)

    current_prompt = token_ids
    current_cache = cache
    index = prompt_length

    while index < max_length:
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

        token_tensor = mx.reshape(mx.array(next_token, dtype=current_prompt.dtype), (1, 1))
        current_prompt = keras.ops.slice_update(current_prompt, [0, index], token_tensor)
        index += 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen 2.5 Coder on Keras MLX Backend with Metal acceleration."
    )
    parser.add_argument(
        "--preset",
        default=os.environ.get("MODEL_PRESET", "qwen2.5_coder_instruct_1.5b"),
        help="KerasHub model preset (e.g. qwen2.5_coder_instruct_1.5b, qwen2.5_coder_instruct_7b)",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Write a Rust http server with async threaded request handling "
            "in pure rust with no high-level http libraries."
        ),
        help="Prompt for code generation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum generation sequence length (default: 1000)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Running Qwen 2.5 Coder ({args.preset}) on Keras MLX Backend")
    print("=" * 60)
    print(f"Keras Backend: {keras.config.backend()}")
    print(f"MLX Device:    {mx.default_device()}")

    print(f"\nLoading model from preset '{args.preset}' (dtype=bfloat16)...")
    t0 = time.time()
    model = keras_hub.models.Qwen2CausalLM.from_preset(args.preset, dtype="bfloat16")
    print(f"Model loaded in {time.time() - t0:.2f}s!")
    print(f"Total Parameters: {model.count_params():,}")
    print(f"Compute Dtype:    {model.compute_dtype}")

    # Compile the single-token forward step to fuse Metal kernels
    print("\nCompiling call_with_cache with mx.compile for kernel fusion...")
    model.call_with_cache = mx.compile(model.call_with_cache)

    # Warm up the compiled graph
    print("Warming up compiled Metal graph...")
    for _ in generate_stream(model, "def foo():", max_length=10):
        pass
    print("Ready!")

    print("\nPrompt:")
    print("-" * 60)
    print(args.prompt)

    print("-" * 60)
    print("Real-Time Streaming Output:")
    print("-" * 60)
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    t_start = time.perf_counter()
    token_count = 0

    for chunk in generate_stream(model, args.prompt, max_length=args.max_length):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        token_count += 1

    elapsed = time.perf_counter() - t_start

    print("\n" + "-" * 60)
    rate = token_count / elapsed if elapsed > 0 else 0
    print(f"Streamed {token_count} tokens in {elapsed:.2f} seconds ({rate:.2f} tokens/sec)")


if __name__ == "__main__":
    main()

"""Run Gemma 3 270M causal language model using KerasHub on the MLX backend.

This script demonstrates loading a real foundation model (Gemma 3 270M)
from Hugging Face and performing autoregressive text generation accelerated
by Apple Silicon GPU via Metal and MLX.
"""

import os
import time

os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402
import keras_hub  # noqa: E402
import mlx.core as mx  # noqa: E402

print("=" * 60)
print("Running Gemma 3 (270M) with KerasHub on MLX Backend")
print("=" * 60)
print(f"Keras Backend:  {keras.config.backend()}")
print(f"MLX Device:     {mx.default_device()}")

# Load model preset from Hugging Face
preset = "hf://google/gemma-3-270m"
print(f"\nLoading model from {preset} ...")
t0 = time.time()
model = keras_hub.models.Gemma3CausalLM.from_preset(preset)
load_time = time.time() - t0
print(f"Model loaded successfully in {load_time:.2f}s!")

# Inspect model weights to verify they are MLX arrays
sample_weight = model.weights[0].value
print(f"\nModel Parameters:   {model.count_params():,}")
print(f"Weight Tensor Type: {type(sample_weight)}")
print(f"Is MLX Array:       {isinstance(sample_weight, mx.array)}")

# Run text generation
prompt = "The sky is blue because"
print(f"\nPrompt: '{prompt}'")
print("-" * 60)

max_length = 35
t_gen_start = time.perf_counter()
output = model.generate(prompt, max_length=max_length)
gen_duration = time.perf_counter() - t_gen_start

print(f"Generated Output:\n{output}")
print("-" * 60)
print(f"Generation Time: {gen_duration:.2f} seconds ({max_length} max tokens)")
print("\nKerasHub Gemma 3 generation succeeded on Apple Silicon MLX GPU!")

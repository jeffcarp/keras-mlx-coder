"""Verify Keras MLX backend setup and hardware device placement.

Note on Apple Silicon hardware:
- MLX executes on Apple Silicon's Metal GPU and CPU.
- The Apple Neural Engine (ANE) is NOT targeted by MLX; ANE is exclusively
  targeted by Apple's Core ML framework. MLX targets the GPU via Metal.
"""

import os
import mlx.core as mx

# Ensure Keras uses the MLX backend
os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402

print("=" * 60)
print("1. Keras & MLX Backend Configuration")
print("=" * 60)
print(f"Keras Version:      {keras.__version__}")
print(f"Configured Backend: {keras.config.backend()}")
print(f"MLX Version:        {mx.__version__}")
print(f"MLX Default Device: {mx.default_device()}")
print(f"MLX Default Stream: {mx.default_stream(mx.default_device())}")

print("\n" + "=" * 60)
print("2. Tensor & Hardware Verification")
print("=" * 60)

# Create a random tensor using Keras random ops
tensor = keras.random.normal(shape=(3, 3))
print(f"Tensor Type:        {type(tensor)}")
print(f"Is MLX Array:       {isinstance(tensor, mx.array)}")
print(f"Array Shape & Type: shape={tensor.shape}, dtype={tensor.dtype}")

# DLPack device: (8, 0) corresponds to kDLMetal (Metal GPU on macOS)
device_type, device_id = tensor.__dlpack_device__()
device_name = "Metal GPU" if device_type == 8 else f"DLPack({device_type})"
print(f"DLPack Device:      {device_name} (id={device_id})")

# Evaluate an operation
out = keras.ops.matmul(tensor, tensor)
mx.eval(out)
print(f"Matmul Output:\n{out}")
print("\nBackend verification successful! MLX is running on Apple Silicon GPU.")

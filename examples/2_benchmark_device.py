"""Benchmark Keras operations across GPU and CPU devices in MLX.

This script demonstrates that Keras operations are hardware-accelerated
on the Apple Silicon GPU via Metal compared to execution on the CPU.
"""

import os
import time
import mlx.core as mx

os.environ["KERAS_BACKEND"] = "mlx"
import keras  # noqa: E402

print("=" * 60)
print("Benchmarking Keras MLX on Apple Silicon GPU vs CPU")
print("=" * 60)

matrix_dim = 3000
num_iterations = 10

print(f"Matrix Dimension: {matrix_dim} x {matrix_dim}")
print(f"Iterations:       {num_iterations}")


def run_benchmark(device_type):
    mx.set_default_device(device_type)
    device_name = "Metal GPU" if device_type == mx.gpu else "CPU"

    # Allocate tensors
    a = keras.random.normal((matrix_dim, matrix_dim))
    b = keras.random.normal((matrix_dim, matrix_dim))

    # Warmup and synchronize (MLX evaluates lazily)
    c = keras.ops.matmul(a, b)
    mx.eval(c)

    # Benchmark timed loop
    start = time.perf_counter()
    for _ in range(num_iterations):
        c = keras.ops.matmul(a, b)
        mx.eval(c)
    elapsed = (time.perf_counter() - start) / num_iterations

    print(f"[{device_name}] Avg Matmul Time: {elapsed * 1000:.2f} ms")
    return elapsed


gpu_time = run_benchmark(mx.gpu)
cpu_time = run_benchmark(mx.cpu)

speedup = cpu_time / gpu_time
print("-" * 60)
print(f"GPU vs CPU Speedup: {speedup:.2f}x faster on Apple Silicon Metal GPU")
print("-" * 60)

# Reset back to GPU
mx.set_default_device(mx.gpu)

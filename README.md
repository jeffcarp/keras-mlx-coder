# keras-mlx-coder

Run State-of-the-Art (SOTA) coding models locally on Apple Silicon (M1/M2/M3/M4) using **[Keras 3](https://github.com/keras-team/keras)** and the **[Apple MLX](https://github.com/ml-explore/mlx)** pluggable backend (**[keras-mlx](https://github.com/keras-team/keras-mlx)**).

Accelerate code generation, run local OpenAI-compatible inference servers, and drive autonomous coding agents (OpenCode, Aider, Continue) with zero cloud dependencies, complete privacy, and native macOS sandboxing.

---

## ⚡ Highlights

- **Pluggable Keras Backend**: Uses Keras 3's new pluggable backend architecture targeting Apple Silicon's Metal GPU via MLX.
- **Compiled Metal Kernel Fusion**: Fuses single-token decoding steps (`model.call_with_cache`) with `mx.compile`, eliminating Python dispatch overhead and unlocking high tokens-per-second streaming.
- **SOTA Coding Models**: Out-of-the-box support for the **Qwen 2.5 Coder** series (0.5B, 1.5B, 7B, 14B, 32B) and Google **Gemma 3** via [KerasHub](https://github.com/keras-team/keras-hub).
- **OpenAI-Compatible Streaming API Server**: Built-in FastAPI server exposing `/v1/chat/completions` (Server-Sent Events streaming + non-streaming) and `/v1/models`.
- **macOS Seatbelt Sandbox**: Includes a native Apple Seatbelt (`sandbox-exec`) sandbox profile that confines coding agents to the workspace, enforces localhost-only networking, and protects sensitive keys (`~/.ssh`, `~/.aws`, `~/.gnupg`).

---

## 📊 Model Recommendations by Apple Silicon Unified Memory

Thanks to Apple Silicon's unified memory architecture, the GPU shares memory directly with system RAM without PCIe bottleneck copying:

| Model Preset | Parameters | Compute Dtype | Min Unified Memory | Recommended Mac |
| :--- | :--- | :--- | :--- | :--- |
| `qwen2.5_coder_instruct_0.5b` | 0.5B | `bfloat16` | 8 GB | Base M1/M2/M3/M4 MacBook Air |
| `qwen2.5_coder_instruct_1.5b` | 1.5B | `bfloat16` | 8 GB – 16 GB | M1/M2/M3/M4 MacBook Air / Pro |
| `qwen2.5_coder_instruct_7b` | 7.6B | `bfloat16` | 16 GB – 24 GB | M-series Pro / Max (16GB+) |
| `qwen2.5_coder_instruct_14b` | 14.7B | `bfloat16` | 32 GB – 48 GB | M-series Pro / Max (36GB+) |
| `qwen2.5_coder_instruct_32b` | 32.5B | `bfloat16` | 64 GB – 128 GB | M-series Max / Ultra (64GB+) |

---

## 🚀 Quickstart

### Prerequisites

- macOS 13.5+ (Sonoma or Sequoia recommended)
- Apple Silicon Mac (M1, M2, M3, M4 chip family)
- Python 3.10 or 3.11

### 1. Installation

Create a clean virtual environment and install dependencies:

```bash
# Clone the repository
git clone https://github.com/jeffcarp/keras-mlx-coder.git
cd keras-mlx-coder

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> **Note on Keras & MLX backend**:
> The MLX backend is activated by setting `KERAS_BACKEND=mlx`:
> ```bash
> export KERAS_BACKEND=mlx
> ```

---

## 🛠️ Step-by-Step Examples

All scripts are located in the `examples/` directory.

### 1. Verify Backend and Metal Hardware Acceleration

Check that Keras is configured with the MLX backend and verified on the Metal GPU via DLPack:

```bash
python examples/1_verify_backend.py
```

Expected output:
```text
============================================================
1. Keras & MLX Backend Configuration
============================================================
Keras Version:      3.16.0
Configured Backend: mlx
MLX Version:        0.31.2
MLX Default Device: Device(gpu, 0)
MLX Default Stream: Stream(Device(gpu, 0), 0)

============================================================
2. Tensor & Hardware Verification
============================================================
Tensor Type:        <class 'mlx.core.array'>
Is MLX Array:       True
DLPack Device:      Metal GPU (id=0)
...
Backend verification successful! MLX is running on Apple Silicon GPU.
```

### 2. Benchmark GPU vs. CPU Performance

Measure matrix multiplication speedup on Apple Silicon Metal GPU vs. CPU:

```bash
python examples/2_benchmark_device.py
```

### 3. End-to-End Model Training & Inference

Train a multi-layer perceptron with `model.fit()` on synthetic data, verifying forward pass, backpropagation, and weight updates on the GPU:

```bash
python examples/3_train_model.py
```

### 4. Autoregressive Generation with Gemma 3

Run Google's Gemma 3 (270M) foundation model via KerasHub:

```bash
python examples/4_gemma3_causal_lm.py
```

*(Note: If running gated models on Hugging Face, run `huggingface-cli login` once beforehand).*

### 5. High-Speed Real-Time Code Streaming (Qwen 2.5 Coder)

Stream code generation token-by-token with single-token forward graph compilation (`mx.compile`):

```bash
# Run default 1.5B coder model
python examples/5_qwen2_5_coder.py

# Or specify custom prompt and model preset
python examples/5_qwen2_5_coder.py \
  --preset qwen2.5_coder_instruct_7b \
  --prompt "Write a concurrent thread-pool worker in modern C++20:"
```

---

## 🌐 Local OpenAI-Compatible API Server

Serve Qwen 2.5 Coder locally through a streaming OpenAI-compatible API endpoint:

```bash
python examples/6_serve_qwen_opencode.py --port 8000
```

### Features
- `/v1/chat/completions`: Full support for Server-Sent Events (SSE) streaming (`"stream": true`) and standard JSON responses.
- `/v1/models`: Returns available model descriptors.
- ChatML formatting: Properly formats multi-turn chat messages (`<|im_start|>user...<|im_end|>`).

### Test with `curl`

In another terminal, test the streaming chat completion endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder",
    "messages": [
      {"role": "user", "content": "Write an LRU Cache in Python using OrderedDict."}
    ],
    "stream": true
  }'
```

---

## 🤖 Connecting Coding Agents

Once the server is running on `http://localhost:8000/v1`, you can connect your favorite AI coding agent:

### 1. OpenCode with macOS Seatbelt Sandbox

Run [OpenCode](https://github.com/opencode-ai/opencode) inside Apple's native kernel-enforced Seatbelt sandbox:

```bash
./examples/sandboxed-opencode.sh
```

**Security protections enabled by the sandbox**:
- **Confined Workspace**: Filesystem write operations are restricted strictly to the current project directory and `/tmp`.
- **Localhost Networking Only**: External internet (outbound WAN) is completely blocked; only connections to `http://localhost:8000` are permitted.
- **Credential Protection**: Reads to `~/.ssh`, `~/.aws`, `~/.gnupg`, and `~/.credentials` are denied at the OS kernel level.

### 2. Aider

```bash
aider \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key dummy \
  --model openai/qwen2.5-coder
```

### 3. Continue.dev (VS Code / JetBrains)

Add the local model to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Qwen 2.5 Coder (MLX)",
      "provider": "openai",
      "model": "qwen2.5-coder",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "dummy"
    }
  ]
}
```

---

## 🧪 Testing

Run unit and integration tests with pytest:

```bash
python -m pytest
```

---

## 📄 License

Apache 2.0

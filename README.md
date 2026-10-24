# keras-mlx-coder

Run State-of-the-Art (SOTA) coding models locally on Apple Silicon (M1/M2/M3/M4) using **[Keras 3](https://github.com/keras-team/keras)** and the **[Apple MLX](https://github.com/ml-explore/mlx)** pluggable backend (**[keras-mlx](https://github.com/keras-team/keras-mlx)**).

Accelerate code generation, serve an OpenAI-compatible API locally, and drive autonomous coding agents (OpenCode, Aider, Continue, Cursor) with zero cloud dependencies, full privacy, and native Metal GPU acceleration.

---

## ⚡ Highlights

- **Pluggable Keras Backend**: Leverages Keras 3's pluggable backend architecture targeting Apple Silicon's Metal GPU via MLX.
- **Unified Multi-Model Server**: Single, high-performance server supporting any [KerasHub](https://github.com/keras-team/keras-hub) causal language model (Qwen 2.5 Coder, Gemma, Llama, etc.).
- **Automatic Chat Templating**: Built-in template abstraction ([`chat_formatter.py`](file:///Users/jeffcarp/gh/jeffcarp/keras-mlx-coder/chat_formatter.py)) supporting ChatML (Qwen, DeepSeek), Gemma, and Llama 3 formatting with auto-detection.
- **Compiled Metal Kernel Fusion**: Fuses single-token decoding steps (`model.call_with_cache`) with `mx.compile`, eliminating Python dispatch overhead and unlocking high tokens-per-second streaming.
- **OpenAI-Compatible Streaming API**: FastAPI server exposing `/v1/chat/completions` (Server-Sent Events streaming + non-streaming) and `/v1/models`.

---

## 📊 Model Sizing & Unified Memory Guide

Thanks to Apple Silicon's unified memory architecture, the GPU shares high-bandwidth memory directly with system RAM:

| Model Preset | Parameters | Dtype | Min RAM | Recommended Mac |
| :--- | :--- | :--- | :--- | :--- |
| `qwen2.5_coder_instruct_0.5b` | 0.5B | `bfloat16` | 8 GB | Base M1/M2/M3/M4 MacBook Air |
| `qwen2.5_coder_instruct_1.5b` *(default)* | 1.5B | `bfloat16` | 8 GB – 16 GB | M1/M2/M3/M4 MacBook Air / Pro |
| `qwen2.5_coder_instruct_7b` | 7.6B | `bfloat16` | 16 GB – 24 GB | M-series Pro / Max (16GB+) |
| `qwen2.5_coder_instruct_14b` | 14.7B | `bfloat16` | 32 GB – 48 GB | M-series Pro / Max (36GB+) |
| `qwen2.5_coder_instruct_32b` | 32.5B | `bfloat16` | 64 GB – 128 GB | M-series Max / Ultra (64GB+) |

---

## 🚀 Quickstart

### Prerequisites

- macOS 13.5+ (Sonoma or Sequoia)
- Apple Silicon Mac (M1, M2, M3, M4)
- Python 3.10 or 3.11

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jeffcarp/keras-mlx-coder.git
cd keras-mlx-coder

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Local API Server

Start the server with the default model (Qwen 2.5 Coder 1.5B):

```bash
python serve_keras_opencode.py
```

Or specify any supported KerasHub preset or port:

```bash
# Run 7B coding model on 16GB+ Mac
python serve_keras_opencode.py --preset qwen2.5_coder_instruct_7b --port 8000

# Run Google Gemma model
python serve_keras_opencode.py --preset gemma2_instruct_2b_en --port 8000
```

#### CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--preset` | `qwen2.5_coder_instruct_1.5b` | KerasHub model preset |
| `--chat-format` | `auto` | Chat format (`chatml`, `gemma`, `llama3`, `plain`) |
| `--dtype` | `bfloat16` | Model compute dtype |
| `--max-sequence-length` | `4096` | Max sequence buffer cap (safeguards Metal allocation) |
| `--host` | `127.0.0.1` | Host address to bind |
| `--port` | `8000` | Port to listen on |

---

## 📡 API Usage

### Test Streaming with `curl`

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

### List Models

```bash
curl http://localhost:8000/v1/models
```

---

## 🤖 Connecting Coding Agents

Once the server is running on `http://localhost:8000/v1`, connect your preferred AI coding tool:

### OpenCode

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

opencode --model qwen2.5-coder
```

### Aider

```bash
aider \
  --openai-api-base http://localhost:8000/v1 \
  --openai-api-key dummy \
  --model openai/qwen2.5-coder
```

### Continue.dev (VS Code / JetBrains)

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

## 🔒 Agent Security & macOS Sandboxing

When running autonomous coding agents that write code or execute shell commands, you can enforce kernel-level isolation using macOS Seatbelt (`sandbox-exec`).

Here is a sandbox wrapper recipe that:
- Confines filesystem writes strictly to your project workspace and `/tmp`.
- Blocks outbound internet access, permitting network requests *only* to `localhost:8000`.
- Blocks read access to sensitive keys (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.credentials`).

```bash
#!/usr/bin/env bash
WORKSPACE="$(pwd)"
USER_HOME="$HOME"

PROFILE="(version 1)
(allow default)

;; 1. Network: Localhost only (talks to local Keras MLX server), blocks WAN
(deny network-outbound)
(allow network-outbound (to ip \"localhost:*\"))
(allow network-outbound (literal \"/private/var/run/mDNSResponder\"))
(allow network-outbound (regex #\"^/dev/\"))

;; 2. Filesystem Writes: Confined to workspace, /tmp, and agent caches
(deny file-write*)
(allow file-write* (subpath \"$WORKSPACE\"))
(allow file-write* (subpath \"/tmp\"))
(allow file-write* (subpath \"/private/tmp\"))
(allow file-write* (subpath \"/private/var/folders\"))
(allow file-write* (subpath \"/dev\"))
(allow file-write* (subpath \"$USER_HOME/.local/share/opencode\"))
(allow file-write* (subpath \"$USER_HOME/.cache/opencode\"))
(allow file-write* (subpath \"$USER_HOME/.config/opencode\"))

;; 3. Sensitive Reads: Block credentials
(deny file-read* (subpath \"$USER_HOME/.ssh\"))
(deny file-read* (subpath \"$USER_HOME/.aws\"))
(deny file-read* (subpath \"$USER_HOME/.gnupg\"))
(deny file-read* (subpath \"$USER_HOME/.credentials\"))
"

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"

exec /usr/bin/sandbox-exec -p "$PROFILE" opencode --model qwen2.5-coder "$@"
```

---

## 🧪 Testing

Run unit tests with pytest:

```bash
python -m pytest
```

---

## 📄 License

Apache 2.0

# ElenkhosServe

A scalable Python inference engine for Qwen models.

## Setup

From the `tiny-serve/` directory:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -q
```

You should see all tests pass.


## Structure

```text
src/elenkhos_serve/
├── config/      # Model and runtime configuration
├── model/       # Owned Qwen architecture and weight loading
├── cache/       # KV cache (simple now, paged later)
├── backends/    # Device/backend boundaries (CPU, MPS, …)
├── engine/      # Single-request forward and decode loop
├── serving/     # Request lifecycle and server (later)
├── bench/       # Benchmark metrics and harness
└── tokenizer/   # Tokenizer adapters
```

## Current progress

Week 1–2 foundation:

- `config/model_config.py` — `QwenConfig` with derived GQA sizes, validation, and Hugging Face `config.json` loading
- `model/layers.py` — `RMSNorm`, `SwiGLUMLP`, `RotaryEmbedding`, `rotate_half`, `apply_rope`
- `model/attention.py` — `QwenAttention` with QK-norm, RoPE, GQA, and eager causal attention
- `model/qwen.py` — `QwenDecoderLayer`, `QwenModel`, `QwenForCausalLM` (embedding → layers → norm → logits)
- `model/loader.py` — strict Hugging Face safetensors loading with tied-weight alias handling

## Roadmap

| Weeks | Focus |
|-------|-------|
| 1–2 | Owned Qwen model, single-request engine, benchmark baseline |
| 3–4 | Paged KV cache, int4 quantization |
| 5–6 | Continuous batching, request admission |
| 7–8 | Chunked prefill, cross-device evaluation |
| 9–11 | Measured optimization beating a baseline |
| 12 | Paper-like report and portfolio release |

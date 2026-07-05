# tiny-serve

A scalable Python inference engine for Qwen models.

## Setup

From the `tiny-serve/` directory:

```bash
# Create .venv and install the package with test dependencies
uv sync --extra dev
```

## Common commands

Run all commands from the `tiny-serve/` directory.

```bash
# Run the test suite
uv run pytest tests/ -q

# Run the Hugging Face Qwen baseline benchmark
uv run --extra bench python -m tiny_serve.bench.hf_qwen3

# Refresh uv.lock after changing dependencies in pyproject.toml
uv lock
```

Use `uv run ...` for project commands so they run inside the managed `.venv`.


## Structure

```text
src/tiny_serve/
├── config/      # Model and runtime configuration
├── model/       # Owned Qwen architecture and weight loading
├── cache/       # KV cache (simple now, paged later)
├── backends/    # Device/backend boundaries (CPU, MPS, …)
├── engine/      # Single-request forward and decode loop
├── serving/     # Request lifecycle and server (later)
├── bench/       # Benchmark metrics and harness
└── tokenizer/   # Tokenizer adapters
```

## Benchmarks

The benchmark command uses the `bench` optional dependency group and downloads
the Qwen model through Hugging Face on first run.

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

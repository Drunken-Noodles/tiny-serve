I see the full roadmap now. This changes one important thing:

Your project structure should support **all 12 weeks**, but you should only implement the Week 1–2 parts now.

The project is not only “make Qwen run.” It gradually becomes a real inference-serving system:

```text
Weeks 1–2: owned Qwen model + single-request engine + benchmark baseline
Weeks 3–4: paged KV cache + int4 quantization
Weeks 5–6: continuous batching + request admission
Weeks 7–8: chunked prefill + cross-device evaluation
Weeks 9–11: one measured optimization that beats a baseline in a specific regime
Week 12: paper-like report and portfolio release
```

So your responsibility right now is the **foundation that every later module sits on**.

## Your part, stated correctly

You are building:

```text
A scalable Python inference-engine codebase
+
a complete owned Qwen forward pass
+
clean device/backend boundaries
+
interfaces that later support KV cache, batching, quantization, and benchmarks
```

You are **not** building all later optimizations now.

The important rule is:

```text
Design for later.
Implement only what Week 1–2 needs.
```

For example, we create a `cache/` area now because paged KV cache comes in Weeks 3–4, but we do not implement PagedAttention yet.

---

# Recommended long-term project structure

Because the project is called **ElenkhosServe**, use that as the package name instead of the generic `inference_engine`.

```text
elenkhos-serve/
├── src/
│   └── elenkhos_serve/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── model_config.py
│       │   └── runtime_config.py
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── layers.py
│       │   ├── attention.py
│       │   ├── qwen.py
│       │   └── loader.py
│       │
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── kv_cache.py
│       │   └── sequence_state.py
│       │
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── torch_cpu.py
│       │   └── torch_mps.py
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── forward.py
│       │   ├── decode.py
│       │   └── sampling.py
│       │
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── request.py
│       │   ├── scheduler.py
│       │   └── server.py
│       │
│       ├── bench/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── traces.py
│       │   └── harness.py
│       │
│       └── tokenizer/
│           ├── __init__.py
│           └── hf_tokenizer.py
│
├── tests/
├── scripts/
├── benchmarks/
├── reports/
├── docs/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# Why this structure matches the 12-week syllabus

## `model/`

This is your current main responsibility.

```text
model/
= “What mathematical neural network are we running?”
```

It holds your owned Qwen architecture:

```text
RMSNorm
RoPE
QK-norm
GQA
SwiGLU
decoder layer
Qwen model
LM head
weight loader
```

This should remain mostly stable for the entire project.

Even when you add batching or PagedAttention later, the actual Qwen math should still live here.

---

## `cache/`

This is small in Week 1–2, but extremely important later.

```text
cache/
= “What past-token information does each request need to remember?”
```

At first, this can be a simple normal KV cache:

```text
layer 0 keys + values
layer 1 keys + values
...
layer 27 keys + values
```

In Weeks 3–4, this area changes from:

```text
one long tensor per request
```

to:

```text
paged blocks of KV memory
```

That is where PagedAttention fits.

So we create the boundary now, even though the first version is simple.

---

## `backends/`

```text
backends/
= “Which hardware runs the tensor operations?”
```

Week 1–2:

```text
CPU
MPS on your Mac
```

Later:

```text
Raspberry Pi CPU
possibly CUDA
```

Your model code stays the same. The backend decides device, dtype, synchronization, and later perhaps which attention method to use.

---

## `engine/`

```text
engine/
= “How does one request actually generate tokens?”
```

Week 1–2:

```text
one prompt
→ forward pass
→ choose one token
→ append token
→ repeat
```

Weeks 3–4:

```text
same decode loop
but uses a paged KV cache
```

Weeks 5–6:

```text
many requests
each takes one decode step
scheduler decides whose turn it is
```

This is why it is important that `engine/` does not contain HTTP code or benchmark code.

---

## `serving/`

```text
serving/
= “How do outside users send requests to the engine?”
```

Week 1–2, this can be minimal or mostly empty.

Later it holds:

```text
OpenAI request format
request lifecycle
cancellation
scheduler
streaming response
admission control
continuous batching
```

The server should ask the engine to generate tokens.

The server should not contain Qwen attention math.

---

## `bench/`

```text
bench/
= “How do we prove whether the engine improved?”
```

This is required throughout the full syllabus.

Every major milestone needs a measurement:

```text
MVP vs llama.cpp
paged KV cache vs MVP
int4 vs FP16 or FP32
continuous batching vs single request
chunked prefill vs ordinary prefill
final optimization vs production baseline
```

This folder becomes the reason your final project is more than “I made a toy LLM server.”

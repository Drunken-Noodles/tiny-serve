# tiny-serve: Incremental Structure and 12-Week Plan

## Project objective

`tiny-serve` is an educational LLM serving engine. It should progressively
support owned Qwen3 inference, explicit prefill and decode, paged KV storage,
direct paged attention, weight quantization, continuous batching, lifecycle
management, chunked prefill, prefix caching, reproducible benchmarking, and
one deeply evaluated advanced optimization.

The goal is not to reproduce every feature of vLLM, SGLang, llama.cpp,
TensorRT-LLM, or a distributed production serving system. The goal is a small
engine whose model mathematics, runtime state, memory ownership, scheduling
policy, and performance claims can be understood and defended.

## Progressive structure policy

The roadmap defines capabilities, not a mandatory directory tree.

- Start from the repository as it exists.
- Add structure only when the active milestone needs it.
- Do not add empty files or directories for future weeks.
- Split a module only after it has distinct responsibilities.
- Introduce an abstraction only when there is a current second implementation
  or a concrete boundary that must be enforced.
- Preserve working reference paths and tests.
- Prefer small, reversible changes.
- Correct missing earlier-week prerequisites before building on top of them.

## Current repository baseline

As of Week 3, the substantive implementation is concentrated in:

```text
src/tiny_serve/
├── config/
│   └── model_config.py       # Qwen model configuration
├── model/
│   ├── attention.py          # QK-norm, RoPE, GQA, dense causal attention
│   ├── layers.py             # RMSNorm, SwiGLU, rotary embeddings
│   ├── loader.py             # strict Hugging Face weight loading
│   └── qwen.py               # decoder layers, model, causal LM head
└── bench/
    └── hf_qwen3.py           # external Hugging Face baseline
```

The package name is `tiny_serve`. Do not rename it to `elenkhos_serve`,
`inference_engine`, or another project name.

The existing cache, engine, serving, backend, tokenizer, and runtime-config
modules are mostly placeholders. Their presence does not mean their milestones
are complete.

## Responsibility boundaries

| Area | Owns | Must not own |
|---|---|---|
| `model/` | Qwen mathematics, layers, projections, weight loading | Page lifecycle, scheduling policy, HTTP schemas |
| `cache/` | KV storage, page allocation, block tables, cache lifecycle | Request queues, sampling, API behavior |
| `engine/` | Request execution, prefill/decode orchestration, scheduling policy | HTTP translation, model mathematics |
| `generation/` if needed | Sampling, logits processing, stopping | Scheduling and cache allocation |
| `kernels/` if needed | Tensor operations with explicit tensor metadata | Requests, queues, HTTP schemas |
| `api/` or `serving/` | Protocol translation, streaming transport | Scheduling policy and model internals |
| `bench/` | Workloads, measurements, reports | Private runtime mutation or runtime behavior |
| `backends/` if needed | Device-specific selection and synchronization | Model or request policy |

Create `generation/`, `kernels/`, or `api/` only when the active implementation
needs those boundaries. Existing coherent modules may remain where they are
until a split has a concrete benefit.

Preferred dependency direction:

```text
API/serving
  -> engine
  -> model
  -> kernels

engine
  -> cache
  -> generation
  -> observability

bench
  -> public engine APIs
```

Prohibited dependency patterns:

```text
model -> engine
cache -> HTTP schemas
kernels -> request objects
API/serving -> model internals
bench -> private runtime mutation
```

## Twelve-week implementation plan

### Week 1: Owned Qwen3 correctness

Deliver:

- Qwen3 configuration and validation
- RMSNorm, RoPE, QK-norm, GQA, and SwiGLU
- decoder layer, model, and LM head
- strict Hugging Face weight loading
- deterministic layer and logit parity tests

Completion gate:

- Tiny deterministic tests pass.
- Owned model outputs match the selected Hugging Face reference within stated
  tolerances.
- No runtime scheduling abstractions are introduced.

### Week 2: Single-request engine and dense baseline

Deliver:

- Dense per-layer KV cache
- Explicit prompt prefill
- Explicit one-token decode
- Minimal internal request representation
- Sampling, stopping, and token streaming
- Owned-engine baseline for time to first token, inter-token latency,
  throughput, and memory

Completion gate:

- Cached decoding matches uncached full-sequence decoding.
- Prefill and decode are separate observable operations.
- Benchmarks invoke the public `tiny_serve` engine path.
- Paging and prefix caching are absent.

### Week 3: Paged KV storage

Deliver:

- Fixed-size physical K/V pages
- One authoritative page allocator and free list
- Per-sequence block tables
- Context length and tail-page metadata
- Logical-token-to-physical-slot mapping
- Allocation, exhaustion, release, and reuse lifecycle tests
- Reconstruction of paged K/V into dense K/V as a correctness oracle

Because the current repository is missing Week 2's owned dense-cache execution
path, first make the storage mechanism correct with tiny synthetic tensors.
Do not hide allocation inside model layers merely to force early integration.

Completion gate:

- Allocator capacity is conserved.
- Every page is either free or allocated, never both.
- Reconstruction exactly matches the original dense K/V, including a partial
  tail page and noncontiguous physical pages.
- Releasing one sequence cannot affect another.
- Existing model tests still pass.

Paged storage is not direct paged attention.

### Week 4: Direct paged attention and weight quantization

Deliver:

- Attention reads K/V pages using block-table metadata
- No required dense reconstruction in the optimized path
- Correct context-length and tail masking
- Online-softmax or a documented tiled strategy
- Parity against dense attention
- One weight-only INT4 path with an explicit scale/group format
- Memory, quality, prefill-latency, and decode-latency ablations

Completion gate:

- Direct page walking matches the dense reference across page boundaries.
- Quantized output quality stays within a stated tolerance.
- Each optimization is measured independently and together.

### Week 5: Continuous batching

Deliver:

- Explicit waiting, prefill, decode, and finished request phases
- Iteration-level scheduler
- Scheduler-produced execution plans
- Dynamic active-batch construction
- Per-request token progression and finished-request removal
- Separation between scheduling policy and model execution

Completion gate:

- Multiple requests progress independently.
- Finishing one request does not disturb others.
- Scheduler decisions can be tested without running the model.
- Model execution does not choose scheduling policy.

### Week 6: Admission, lifecycle, and API integration

Deliver:

- Token and page budgets
- Bounded waiting queue
- Admission and overload behavior
- Cancellation and deterministic cleanup
- Queueing, rejection, and cleanup metrics
- Thin API/serving integration
- Fairness or starvation measurements

Completion gate:

- Cancelled and completed requests release all noncached pages.
- Over-capacity requests are queued or rejected predictably.
- The HTTP layer translates protocols but does not schedule work.

### Week 7: Chunked prefill

Deliver:

- Prompt cursor
- Configurable prefill chunk size
- Total iteration token budget
- Mixed prefill and decode batches
- Decode reservation or another documented anti-starvation policy
- Long-prompt interference benchmark

Completion gate:

- Chunked and unchunked prefill produce equivalent outputs.
- Prompt cursors advance exactly once per processed token.
- Decode latency under long-prefill interference is measured.

### Week 8: Prefix caching

Deliver:

- Block-level prefix keys
- Full-block reuse
- Hit/miss and reuse metrics
- Reference-counted pages
- Immutable shared pages
- Copy-on-write when sequences diverge
- Bounded eviction policy
- Repeated-prefix workload

The prefix index must reuse pages from the normal allocator rather than own a
second memory pool.

Completion gate:

- Prefix reuse does not change generated outputs.
- Reference counts never become negative.
- Shared pages remain immutable.
- Eviction cannot free a referenced page.

### Week 9: Profile and select one advanced optimization

Profile target, neutral, and adversarial workloads. Select exactly one advanced
optimization based on the measured dominant bottleneck:

| Measured bottleneck | Candidate |
|---|---|
| KV memory or concurrency | KV-cache quantization |
| Repeated structured prompts | RadixAttention or cache-aware scheduling |
| Single-stream decode latency | Speculative decoding |
| Unbounded conversation length | KV eviction or streaming policy |
| Device memory capacity | KV offload |
| GPU launch overhead | CUDA graphs |

Completion gate:

- A written selection memo identifies the workload, bottleneck, evidence,
  rejected alternatives, expected benefit, and expected downside.
- No implementation begins merely because a technique is fashionable.

### Week 10: Integrate the selected optimization

Deliver:

- Correct reference comparison
- Minimal runtime integration
- Explicit fallback or disable path
- Focused correctness and lifecycle tests
- Instrumentation needed to explain behavior

Completion gate:

- The feature is independently switchable.
- Outputs or quality metrics meet a stated tolerance.
- Core ownership invariants remain intact.

### Week 11: Profile and ablate

Deliver:

- Target workload where the optimization should help
- Neutral workload where little change is expected
- Adversarial workload that exposes costs or failure modes
- End-to-end and component-level measurements
- One-feature-at-a-time ablations
- Hardware and configuration run cards

Completion gate:

- Claims are reproducible from raw data.
- Improvements are not attributed across changed model, precision, workload,
  and batching settings simultaneously.
- Regressions and limitations are reported alongside wins.

### Week 12: Release and report

Deliver:

- Architecture document
- Request-state and scheduler diagrams
- Page-table and memory-layout diagram
- Prefix-sharing and copy-on-write diagram
- Benchmark methodology and raw results
- Reproduction scripts and hardware run cards
- Known limitations
- Final report and portfolio-facing summary

Completion gate:

- A new reader can reproduce the central correctness and performance claims.
- Documentation reflects the implementation that exists, not the idealized
  long-term tree.
- Unsupported or nonportable claims are removed.

## Cross-week invariants

- Context length equals the number of valid KV tokens.
- Block-table entries reference only allocated pages.
- Allocator capacity is conserved.
- Released pages are not referenced.
- Mutable sequences never write to the same physical page.
- Shared prefix pages are immutable.
- Reference counts never become negative.
- The scheduler respects token and page budgets.
- Cancelled and completed requests release noncached pages.
- Tail masking prevents reads from unused page slots.
- Optimized paths preserve a simpler correctness reference.
- Benchmark configurations record hardware, model, precision, context length,
  output length, concurrency, warmup, sampling, and backend.

## Deferred long-term map

The following is a responsibility map, not a tree to create now:

```text
src/tiny_serve/
├── config/          # model and runtime configuration
├── model/           # Qwen mathematics and weight loading
├── kernels/         # explicit optimized tensor operations, when needed
├── cache/           # KV storage and memory management
├── quantization/    # quantized formats and transforms, when needed
├── engine/          # execution and scheduling policy
├── generation/      # sampling and stopping, if a split becomes useful
├── backends/        # hardware-specific boundaries, when needed
├── api/             # external protocol translation, when needed
├── tokenizer/       # tokenizer adapters
└── observability/   # runtime metrics and traces, when needed

benchmarks or src/tiny_serve/bench/
tests/
scripts/
configs/
docs/
reports/
```

Do not create a listed directory until an active milestone needs it. Prefer the
existing `bench/` and `serving/` locations while their responsibilities remain
coherent; rename or move them only when doing so resolves real coupling.

## Benchmark standard

Every performance claim must state:

- hardware and backend
- model and precision
- prompt and generation lengths
- concurrency and batching behavior
- warmup procedure
- sampling settings
- measured metric and units
- reference implementation
- raw observations or reproducible result files

Use tiny deterministic tests before full-model tests, correctness before
performance, and profiling before selecting an optimization.

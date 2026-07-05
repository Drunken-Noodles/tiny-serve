# hf_qwen3.py
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    output_tokens_per_second: float


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_model(
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Float16 is appropriate for MPS. Float32 is the safest CPU baseline.
    dtype = torch.float16 if device.type == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    return tokenizer, model

# ---------------------- Benchmark code ----------------------

# Throughput
@torch.inference_mode()
def generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 64,
) -> GenerationResult:
    messages = [{"role": "user", "content": prompt}]

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    input_tokens = inputs["input_ids"].shape[1]

    # Warm synchronization prevents queued device work from contaminating timing.
    synchronize(device)
    start = time.perf_counter()

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    synchronize(device)
    elapsed = time.perf_counter() - start

    generated_ids = generated[0, input_tokens:]
    output_tokens = generated_ids.numel()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        output_tokens_per_second=output_tokens / elapsed,
    )

@torch.inference_mode()
def benchmark_prefill_decode(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    decode_steps: int = 64,
) -> dict[str, float]:
    synchronize(device)
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_start = time.perf_counter()

    for _ in range(decode_steps):
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

    synchronize(device)
    decode_seconds = time.perf_counter() - decode_start

    prompt_tokens = input_ids.shape[1]

    return {
        "prompt_tokens": float(prompt_tokens),
        "decode_tokens": float(decode_steps),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": prompt_tokens / prefill_seconds,
        "decode_tokens_per_second": decode_steps / decode_seconds,
        "time_per_output_token_ms": 1000.0 * decode_seconds / decode_steps,
    }


def main() -> None:
    device = select_device()
    print(f"Device: {device}")

    tokenizer, model = load_model(device)

    # Throughput
    
    # Warm-up run
    generate(
        tokenizer,
        model,
        device,
        prompt="Write one sentence about distributed systems.",
        max_new_tokens=8,
    )

    result = generate(
        tokenizer,
        model,
        device,
        prompt="Explain what a key-value cache does in an LLM.",
        max_new_tokens=64,
    )

    print(f"\nThroughput Benchmark: ")
    print(f"Input tokens:  {result.input_tokens}")
    print(f"Output tokens: {result.output_tokens}")
    print(f"Elapsed:       {result.elapsed_seconds:.3f} s")
    print(f"Output rate:   {result.output_tokens_per_second:.2f} tok/s")
    print(result.text)
    
    
    # Prefill & Decode
    messages = [
        {
            "role": "user",
            "content": "Explain the difference between prefill and decode.",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    # Warm-up
    benchmark_prefill_decode(
        model=model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        device=device,
        decode_steps=4,
    )

    result = benchmark_prefill_decode(
        model=model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        device=device,
        decode_steps=64,
    )

    print(f"\nPrefill & Decode Benchmark: ")
    for key, value in result.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
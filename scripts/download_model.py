from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen3-0.6B"
LOCAL_DIR = Path("checkpoints/Qwen3-0.6B")


def main() -> None:
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ],
    )

    print(f"Downloaded checkpoint to: {local_path}")


if __name__ == "__main__":
    main()

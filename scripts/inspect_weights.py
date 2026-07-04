from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contains",
        type=str,
        default="",
        help="Only print tensor names containing this text.",
    )

    args = parser.parse_args()

    weights_path = args.checkpoint / "model.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find: {weights_path}"
        )

    with safe_open(
        str(weights_path),
        framework="pt",
        device="cpu",
    ) as checkpoint:
        keys = checkpoint.keys()

        for name in keys:
            if args.contains and args.contains not in name:
                continue

            tensor = checkpoint.get_tensor(name)

            print(
                f"{name:<70} "
                f"shape={tuple(tensor.shape)!s:<18} "
                f"dtype={tensor.dtype}"
            )


if __name__ == "__main__":
    main()

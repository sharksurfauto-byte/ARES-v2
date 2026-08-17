"""Sample text generation script for ARES V2 Qwen backbone supporting sharded multi-GPU execution."""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ares.backbone.qwen_loader import (
    generate_qwen_text,
    load_qwen_model,
    load_qwen_tokenizer,
    verify_qwen_backbone,
)
from ares.utils.config import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="ARES V2 - Qwen Model Text Generation")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial intelligence reliability is important because",
        help="Input text prompt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="Precision format ('bfloat16', 'float16', 'float32')",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate",
    )
    args = parser.parse_args()

    print(f"Loading model & tokenizer: '{args.model_name}' (dtype: {args.torch_dtype})...")
    cfg = ModelConfig(name_or_path=args.model_name, torch_dtype=args.torch_dtype)
    tokenizer = load_qwen_tokenizer(cfg)
    model = load_qwen_model(cfg)

    info = verify_qwen_backbone(model, tokenizer)
    print("\nModel Verification & Placement Info:")
    for k, v in info.items():
        print(f"  - {k}: {v}")

    print(f"\nPrompt: '{args.prompt}'")
    print("Generating...")

    result = generate_qwen_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    print("\nGenerated Output:")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()

"""CLI execution script for Phase 2 Foundation Adaptation training."""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mask preinstalled incompatible torchao version on Kaggle to prevent PEFT import errors
sys.modules["torchao"] = None

import torch
from ares.adaptation import (
    DomainSample,
    MultiDomainTextDataset,
    create_multi_domain_dataloader,
    setup_foundation_adapter,
    train_foundation_adapter,
)
from ares.backbone.qwen_loader import load_qwen_model, load_qwen_tokenizer
from ares.utils.config import ModelConfig, load_config
from ares.utils.environment import is_main_process, set_seed


def generate_synthetic_multi_domain_samples() -> list[DomainSample]:
    """Generate synthetic multi-domain training samples for baseline foundation adaptation verification."""
    samples = [
        # Domain 0: General
        DomainSample("gen_1", "general", "General Knowledge: The capital of France is Paris. It is known for art and culture."),
        DomainSample("gen_2", "general", "General Knowledge: Photosynthesis is the process by which green plants convert sunlight into chemical energy."),

        # Domain 1: Math
        DomainSample("math_1", "math", "Solve the math problem step-by-step: If John has 5 apples and eats 2, how many remain?", "3 apples."),
        DomainSample("math_2", "math", "Solve the math problem step-by-step: Calculate 12 multiplied by 4.", "48."),

        # Domain 2: Code
        DomainSample("code_1", "code", "Write Python code for the following task: Return the sum of a list of numbers.", "def sum_list(nums):\n    return sum(nums)"),
        DomainSample("code_2", "code", "Write Python code for the following task: Check if a string is a palindrome.", "def is_palindrome(s):\n    return s == s[::-1]"),

        # Domain 3: Science
        DomainSample("sci_1", "science", "Science Question: What state of matter is water steam?", "Gas."),
        DomainSample("sci_2", "science", "Science Question: What force pulls objects toward the center of the Earth?", "Gravity."),
    ]
    return samples * 10  # Duplicate for epoch steps execution


def main():
    parser = argparse.ArgumentParser(description="ARES V2 - Phase 2 Foundation Adaptation Training")
    parser.add_argument("--config", type=str, default="configs/experiments/smoke_test.yaml", help="Path to experiment config YAML")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", help="Hugging Face model ID")
    parser.add_argument("--torch_dtype", type=str, default="float16", help="Torch precision ('float16', 'bfloat16', 'float32')")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    set_seed(42)

    if is_main_process():
        print(f"=== ARES V2 Phase 2 Foundation Adaptation ===")
        print(f"Target Model: {args.model_name}")
        print(f"Precision Dtype: {args.torch_dtype}")

    # Load Model & Tokenizer
    model_cfg = ModelConfig(name_or_path=args.model_name, torch_dtype=args.torch_dtype)
    tokenizer = load_qwen_tokenizer(model_cfg)
    base_model = load_qwen_model(model_cfg)

    # Setup LoRA Adapter
    if is_main_process():
        print("\nWrapping model with LoRA Foundation Adapter...")
    peft_model = setup_foundation_adapter(base_model, r=16, lora_alpha=32)

    # Prepare Data
    samples = generate_synthetic_multi_domain_samples()
    dataset = MultiDomainTextDataset(samples, tokenizer, max_seq_length=256)
    dataloader, _ = create_multi_domain_dataloader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr, weight_decay=0.01)

    if is_main_process():
        print(f"\nStarting LoRA Adaptation Training ({len(dataset)} samples)...")

    results = train_foundation_adapter(
        model=peft_model,
        train_dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=2,
        output_dir="checkpoints/foundation_adapter",
    )

    if is_main_process():
        print("\n=== Phase 2 Foundation Adaptation Completed ===")
        print(f"Total Steps: {results['total_steps']}")
        print(f"Final Loss: {results['final_loss']:.4f}")


if __name__ == "__main__":
    main()

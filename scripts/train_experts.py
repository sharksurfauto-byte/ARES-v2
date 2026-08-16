"""CLI script to train domain-specialized LoRA experts for Phase 5."""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mask preinstalled incompatible torchao version on Kaggle
sys.modules["torchao"] = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ares.experts import ExpertManager, ExpertTrainer
from ares.utils.config import load_config
from ares.utils.environment import set_seed, resolve_device


class DummyCausalLM(nn.Module):
    """Dummy causal LM for local CPU testing and validation."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256):
        super().__init__()
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": hidden_dim})()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.embedding(input_ids)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        h_out = self.o_proj(q + k + v)
        logits = self.lm_head(h_out)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = type("CausalLMOutput", (), {"loss": loss, "logits": logits})()
        return output


def generate_synthetic_domain_dataset(vocab_size: int = 1000, seq_len: int = 16, samples: int = 100):
    """Generate synthetic token batch dataset."""
    input_ids = torch.randint(0, vocab_size, (samples, seq_len))
    labels = input_ids.clone()
    return TensorDataset(input_ids, labels)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 5 Domain-Specialized Experts")
    parser.add_argument("--config", type=str, default="configs/experts/expert_mixture.yaml", help="Path to expert mixture config")
    parser.add_argument("--output_dir", type=str, default="checkpoints/experts", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per expert")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device()

    print("=" * 60)
    print("ARES V2 Phase 5 — Training Domain-Specialized LoRA Experts")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Instantiate Base Model
    base_model = DummyCausalLM(vocab_size=1000, hidden_dim=256).to(device)

    # 2. Instantiate ExpertManager
    manager = ExpertManager(
        base_model=base_model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    output_dir = Path(args.output_dir)

    # 3. Train Each Domain Expert
    experts_to_train = ["E0_general", "E1_math", "E2_code", "E3_science"]
    for expert_name in experts_to_train:
        print(f"\n--- Training Domain Expert: {expert_name} ---")
        dataset = generate_synthetic_domain_dataset(vocab_size=1000, seq_len=16, samples=64)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        trainer = ExpertTrainer(
            expert_manager=manager,
            expert_name=expert_name,
            lr=2e-4,
            device=device,
        )

        history = trainer.train(dataloader, epochs=args.epochs)
        final_loss = history[-1]["train_loss"]
        print(f"Finished {expert_name} training! Final Loss: {final_loss:.4f}")

        # Save Expert Checkpoint
        saved_path = manager.save_expert_checkpoint(
            expert_name=expert_name,
            output_dir=output_dir,
            metadata={"epochs": args.epochs, "final_loss": final_loss},
        )
        print(f"Saved {expert_name} checkpoint to: {saved_path}")

    print("\nPhase 5 Domain Experts Training Complete!")


if __name__ == "__main__":
    main()

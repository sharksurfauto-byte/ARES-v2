"""CLI training script for Phase 4 Global & Local Reliability Models (GRM & LRM)."""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mask preinstalled incompatible torchao version on Kaggle
sys.modules["torchao"] = None

import torch
from torch.utils.data import DataLoader, random_split

from ares.reliability import (
    GlobalReliabilityModel,
    GRMDataset,
    GRMTrainer,
    LocalReliabilityModel,
    LRMDataset,
    LRMTrainer,
    ReliabilityManager,
)
from ares.representations import (
    RepresentationRecord,
    load_representations,
)
from ares.utils.config import load_config
from ares.utils.environment import set_seed, is_main_process


def generate_synthetic_records(n_samples: int = 400, hidden_dim: int = 3584) -> list[RepresentationRecord]:
    """Generate synthetic representation records for local training/testing."""
    import numpy as np
    records = []
    domains = ["code", "general", "math", "science"]
    layers = [-1, -6, -12, -28]

    np.random.seed(42)

    for i in range(n_samples):
        domain = domains[i % len(domains)]
        layer = layers[i % len(layers)]

        # Synthetic representation with domain offset
        offset = (domains.index(domain) + 1) * 2.0
        rep = np.random.randn(hidden_dim).astype(np.float32) + offset

        correct = (i % 5 != 0)  # 80% correct
        conf = 0.8 if correct else 0.3

        rec = RepresentationRecord(
            sample_id=f"synth_{i}",
            domain=domain,
            task="synthetic",
            layer=layer,
            representation=rep,
            logits=np.random.randn(100).astype(np.float32),
            prediction="token_5",
            correctness=correct,
            confidence=conf,
            entropy=0.5 if correct else 2.0,
            margin=0.8 if correct else 0.1,
        )
        records.append(rec)

    return records


def main():
    parser = argparse.ArgumentParser(description="Train Phase 4 Reliability Models (GRM & LRM)")
    parser.add_argument("--config", type=str, default="configs/reliability/reliability_models.yaml", help="Path to reliability config")
    parser.add_argument("--input_dir", type=str, default="datasets/representations", help="Input representation dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/reliability", help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("ARES V2 Phase 4 — Training Global & Local Reliability Models")
    print("=" * 60)

    # 1. Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(config_path)
        grm_cfg = cfg.get("grm", {})
        lrm_cfg = cfg.get("lrm", {})
        manager_cfg = cfg.get("manager", {})
    else:
        grm_cfg = {"input_dim": 3584, "bottleneck_dim": 128, "hidden_dim": 256, "num_domains": 4}
        lrm_cfg = {"input_dim": 3584, "bottleneck_dim": 64, "hidden_dim": 128}
        manager_cfg = {"aggregation_method": "weighted_sum", "weight_global": 0.5, "weight_local": 0.5}

    # 2. Load representation dataset
    input_path = Path(args.input_dir)
    records = []
    if input_path.exists():
        try:
            print(f"Loading representation dataset from {input_path}...")
            records = load_representations(input_path)
            print(f"Loaded {len(records)} records from storage.")
        except Exception as e:
            print(f"Could not load from {input_path}: {e}")

    if not records:
        print("Generating 400 synthetic representation records for training...")
        records = generate_synthetic_records(n_samples=400, hidden_dim=grm_cfg.get("input_dim", 3584))

    # Determine dimensions from data if available
    input_dim = records[0].representation.shape[0] if records else grm_cfg.get("input_dim", 3584)
    print(f"Representation dimension: {input_dim}")

    # 3. Create Datasets & DataLoaders
    grm_dataset = GRMDataset(records)
    lrm_dataset = LRMDataset(records)

    train_size = int(0.8 * len(records))
    val_size = len(records) - train_size

    grm_train_ds, grm_val_ds = random_split(grm_dataset, [train_size, val_size])
    lrm_train_ds, lrm_val_ds = random_split(lrm_dataset, [train_size, val_size])

    grm_train_loader = DataLoader(grm_train_ds, batch_size=args.batch_size, shuffle=True)
    grm_val_loader = DataLoader(grm_val_ds, batch_size=args.batch_size, shuffle=False)

    lrm_train_loader = DataLoader(lrm_train_ds, batch_size=args.batch_size, shuffle=True)
    lrm_val_loader = DataLoader(lrm_val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 4. Instantiate Models
    grm_model = GlobalReliabilityModel(
        input_dim=input_dim,
        bottleneck_dim=grm_cfg.get("bottleneck_dim", 128),
        hidden_dim=grm_cfg.get("hidden_dim", 256),
        num_domains=grm_cfg.get("num_domains", 4),
        dropout=grm_cfg.get("dropout", 0.1),
        use_layer_depth_embedding=grm_cfg.get("use_layer_depth_embedding", True),
    )

    lrm_model = LocalReliabilityModel(
        input_dim=input_dim,
        bottleneck_dim=lrm_cfg.get("bottleneck_dim", 64),
        hidden_dim=lrm_cfg.get("hidden_dim", 128),
        dropout=lrm_cfg.get("dropout", 0.1),
        use_layer_depth_embedding=lrm_cfg.get("use_layer_depth_embedding", True),
    )

    # 5. Train GRM
    print("\n--- Training Global Reliability Model (GRM) ---")
    grm_trainer = GRMTrainer(grm_model, lr=grm_cfg.get("lr", 1e-3), device=device)
    grm_history = grm_trainer.train(grm_train_loader, val_dataloader=grm_val_loader, epochs=args.epochs)
    final_grm = grm_history[-1]
    print(f"GRM Final Epoch - Train Loss: {final_grm['train_loss']:.4f}, Domain Acc: {final_grm['train_domain_accuracy']:.4f}, Val Acc: {final_grm.get('val_domain_accuracy', 0.0):.4f}")

    # 6. Train LRM
    print("\n--- Training Local Reliability Model (LRM) ---")
    lrm_trainer = LRMTrainer(lrm_model, lr=lrm_cfg.get("lr", 1e-3), device=device)
    lrm_history = lrm_trainer.train(lrm_train_loader, val_dataloader=lrm_val_loader, epochs=args.epochs)
    final_lrm = lrm_history[-1]
    print(f"LRM Final Epoch - Train Loss: {final_lrm['train_loss']:.4f}, Accuracy: {final_lrm['train_accuracy']:.4f}, Val Acc: {final_lrm.get('val_accuracy', 0.0):.4f}")

    # 7. ReliabilityManager Integration & Checkpoint Saving
    manager = ReliabilityManager(
        grm=grm_model,
        lrm=lrm_model,
        aggregation_method=manager_cfg.get("default_aggregation", "weighted_sum"),
        weight_global=manager_cfg.get("weight_global", 0.5),
        weight_local=manager_cfg.get("weight_local", 0.5),
        confidence_threshold=manager_cfg.get("confidence_threshold", 0.7),
    )

    output_dir = Path(args.output_dir)
    print(f"\nSaving trained reliability model checkpoints to {output_dir}...")
    saved_paths = manager.save_checkpoint(output_dir, metadata={"epochs": args.epochs, "input_dim": input_dim})
    print(f"Checkpoints saved: {saved_paths}")

    # 8. Run Evaluation Test
    test_rep = torch.randn(input_dim)
    res = manager.evaluate(test_rep, layer_idx=-1)
    print("\nSample Reliability Evaluation:")
    print(f"  Predicted Domain: {res.predicted_domain}")
    print(f"  Global Reliability: {res.global_reliability:.4f}")
    print(f"  Local Reliability:  {res.local_reliability:.4f}")
    print(f"  Combined R(x):      {res.combined_reliability:.4f}")
    print(f"  Is Reliable:        {res.is_reliable}")

    print("\nPhase 4 Reliability Model Training Complete!")


if __name__ == "__main__":
    main()

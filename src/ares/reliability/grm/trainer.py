"""Trainer pipeline for Global Reliability Model (GRM).

Handles multi-task training (Domain Classification + Global Feasibility Estimation)
on representation datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ares.reliability.grm.model import GlobalReliabilityModel
from ares.representations.collector import RepresentationRecord
from ares.utils.checkpoint import save_checkpoint_with_metadata, load_checkpoint_with_validation
from ares.utils.environment import is_main_process, get_rank, is_ddp_initialized


class GRMDataset(Dataset):
    """PyTorch Dataset wrapping RepresentationRecords for GRM training."""

    DOMAIN_TO_IDX = {"code": 0, "general": 1, "math": 2, "science": 3}

    def __init__(self, records: List[RepresentationRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        rep = torch.from_numpy(rec.representation).float()
        domain_idx = self.DOMAIN_TO_IDX.get(rec.domain.lower(), 1)
        domain_target = torch.tensor(domain_idx, dtype=torch.long)

        # Feasibility target (1.0 for correct/high confidence, 0.0 for incorrect)
        if rec.correctness is not None:
            feasibility_val = 1.0 if rec.correctness else 0.0
        else:
            # Fallback to confidence threshold if correctness label absent
            feasibility_val = 1.0 if rec.confidence >= 0.5 else 0.0

        feasibility_target = torch.tensor([feasibility_val], dtype=torch.float32)
        layer_idx = torch.tensor(rec.layer, dtype=torch.long)

        return {
            "representation": rep,
            "domain_target": domain_target,
            "feasibility_target": feasibility_target,
            "layer_idx": layer_idx,
        }


class GRMTrainer:
    """Trainer for Global Reliability Model (GRM)."""

    def __init__(
        self,
        model: GlobalReliabilityModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        domain_loss_weight: float = 1.0,
        feasibility_loss_weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.domain_loss_weight = domain_loss_weight
        self.feasibility_loss_weight = feasibility_loss_weight

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.domain_criterion = nn.CrossEntropyLoss()
        self.feasibility_criterion = nn.BCELoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        domain_loss_total = 0.0
        feasibility_loss_total = 0.0
        correct_domain = 0
        total_samples = 0

        for batch in dataloader:
            reps = batch["representation"].to(self.device)
            domain_targets = batch["domain_target"].to(self.device)
            feasibility_targets = batch["feasibility_target"].to(self.device)
            layer_indices = batch["layer_idx"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(reps, layer_idx=layer_indices)

            loss_domain = self.domain_criterion(outputs["domain_logits"], domain_targets)
            loss_feasibility = self.feasibility_criterion(outputs["feasibility"], feasibility_targets)

            loss = (self.domain_loss_weight * loss_domain) + (self.feasibility_loss_weight * loss_feasibility)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * reps.size(0)
            domain_loss_total += loss_domain.item() * reps.size(0)
            feasibility_loss_total += loss_feasibility.item() * reps.size(0)

            preds = outputs["domain_logits"].argmax(dim=-1)
            correct_domain += (preds == domain_targets).sum().item()
            total_samples += reps.size(0)

        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "domain_loss": domain_loss_total / n,
            "feasibility_loss": feasibility_loss_total / n,
            "domain_accuracy": correct_domain / n,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate GRM model on dataset."""
        self.model.eval()
        total_loss = 0.0
        domain_loss_total = 0.0
        feasibility_loss_total = 0.0
        correct_domain = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                reps = batch["representation"].to(self.device)
                domain_targets = batch["domain_target"].to(self.device)
                feasibility_targets = batch["feasibility_target"].to(self.device)
                layer_indices = batch["layer_idx"].to(self.device)

                outputs = self.model(reps, layer_idx=layer_indices)

                loss_domain = self.domain_criterion(outputs["domain_logits"], domain_targets)
                loss_feasibility = self.feasibility_criterion(outputs["feasibility"], feasibility_targets)

                loss = (self.domain_loss_weight * loss_domain) + (self.feasibility_loss_weight * loss_feasibility)

                total_loss += loss.item() * reps.size(0)
                domain_loss_total += loss_domain.item() * reps.size(0)
                feasibility_loss_total += loss_feasibility.item() * reps.size(0)

                preds = outputs["domain_logits"].argmax(dim=-1)
                correct_domain += (preds == domain_targets).sum().item()
                total_samples += reps.size(0)

        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "domain_loss": domain_loss_total / n,
            "feasibility_loss": feasibility_loss_total / n,
            "domain_accuracy": correct_domain / n,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> List[Dict[str, float]]:
        """Run full training loop for multiple epochs."""
        history = []
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_dataloader)
            epoch_log = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}}

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                epoch_log.update({f"val_{k}": v for k, v in val_metrics.items()})

            history.append(epoch_log)

        return history

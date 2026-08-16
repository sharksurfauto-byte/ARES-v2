"""Trainer pipeline for Local Reliability Model (LRM).

Handles token/context correctness probability learning on representation datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from ares.reliability.lrm.model import LocalReliabilityModel
from ares.representations.collector import RepresentationRecord


class LRMDataset(Dataset):
    """PyTorch Dataset wrapping RepresentationRecords for LRM correctness training."""

    def __init__(self, records: List[RepresentationRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        rep = torch.from_numpy(rec.representation).float()

        # Target correctness probability (1.0 for correct, 0.0 for incorrect)
        if rec.correctness is not None:
            target_val = 1.0 if rec.correctness else 0.0
        else:
            target_val = 1.0 if rec.confidence >= 0.5 else 0.0

        target = torch.tensor([target_val], dtype=torch.float32)
        layer_idx = torch.tensor(rec.layer, dtype=torch.long)

        return {
            "representation": rep,
            "correctness_target": target,
            "layer_idx": layer_idx,
        }


class LRMTrainer:
    """Trainer for Local Reliability Model (LRM)."""

    def __init__(
        self,
        model: LocalReliabilityModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.criterion = nn.BCELoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in dataloader:
            reps = batch["representation"].to(self.device)
            targets = batch["correctness_target"].to(self.device)
            layer_indices = batch["layer_idx"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(reps, layer_idx=layer_indices)
            loss = self.criterion(outputs["correctness_prob"], targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * reps.size(0)
            preds = (outputs["correctness_prob"] >= 0.5).float()
            correct_predictions += (preds == targets).sum().item()
            total_samples += reps.size(0)

        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "accuracy": correct_predictions / n,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate LRM model on dataset."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                reps = batch["representation"].to(self.device)
                targets = batch["correctness_target"].to(self.device)
                layer_indices = batch["layer_idx"].to(self.device)

                outputs = self.model(reps, layer_idx=layer_indices)
                loss = self.criterion(outputs["correctness_prob"], targets)

                total_loss += loss.item() * reps.size(0)
                preds = (outputs["correctness_prob"] >= 0.5).float()
                correct_predictions += (preds == targets).sum().item()
                total_samples += reps.size(0)

        n = max(total_samples, 1)
        return {
            "loss": total_loss / n,
            "accuracy": correct_predictions / n,
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

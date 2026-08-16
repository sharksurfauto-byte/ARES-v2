"""Trainer for Domain-Specialized LoRA Experts (Phase 5)."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ares.experts.manager import ExpertManager


class ExpertTrainer:
    """Trainer for domain-specialized LoRA expert adapters."""

    def __init__(
        self,
        expert_manager: ExpertManager,
        expert_name: str,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """Initialize ExpertTrainer.

        Args:
            expert_manager: Instantiated ExpertManager.
            expert_name: Name of expert to train ('E0_general', 'E1_math', etc.).
            lr: Learning rate for LoRA weights.
            weight_decay: Weight decay.
            device: Training device (CPU or CUDA).
        """
        self.manager = expert_manager
        self.expert_name = expert_name
        self.lr = lr
        self.weight_decay = weight_decay
        from ares.utils.environment import resolve_device
        self.device = device or resolve_device()

        # Set active expert for training
        self.manager.set_active_expert(self.expert_name)
        
        # Collect trainable parameters (only active LoRA adapter weights)
        trainable_params = [
            p for n, p in self.manager.base_model.named_parameters() if p.requires_grad
        ]
        if not trainable_params:
            # Fallback if model is not wrapped with PEFT
            trainable_params = list(self.manager.base_model.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch on domain dataset.

        Args:
            dataloader: DataLoader supplying batch input_ids and labels.

        Returns:
            Metrics dictionary for the epoch.
        """
        self.manager.base_model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            else:
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                attention_mask = None

            self.optimizer.zero_grad()

            outputs = self.manager.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.base_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        epoch_loss = total_loss / max(total_tokens, 1)
        elapsed = time.time() - start_time

        return {
            "loss": epoch_loss,
            "perplexity": torch.exp(torch.tensor(epoch_loss)).item() if epoch_loss < 20 else float("inf"),
            "elapsed_seconds": elapsed,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 3,
    ) -> List[Dict[str, float]]:
        """Run complete multi-epoch training loop for domain expert.

        Args:
            train_dataloader: Training dataset loader.
            val_dataloader: Optional validation dataset loader.
            epochs: Number of training epochs.

        Returns:
            List of per-epoch metric dictionaries.
        """
        history = []
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_dataloader)
            epoch_data = {"epoch": epoch + 1, "train_loss": train_metrics["loss"]}

            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                epoch_data["val_loss"] = val_loss

            history.append(epoch_data)

        return history

    def evaluate(self, val_dataloader: DataLoader) -> float:
        """Evaluate expert on validation dataset.

        Args:
            val_dataloader: Validation dataloader.

        Returns:
            Average validation loss float.
        """
        self.manager.base_model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_dataloader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch.get("labels", input_ids).to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    attention_mask = None

                outputs = self.manager.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                batch_tokens = input_ids.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        return total_loss / max(total_tokens, 1)

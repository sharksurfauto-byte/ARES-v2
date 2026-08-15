"""Multi-domain dataset loading, formatting, and DDP sampling pipeline for Phase 2 Foundation Adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

from ares.utils.environment import is_ddp_initialized


@dataclass
class DomainSample:
    """Represents a single multi-domain training/validation data point."""
    sample_id: str
    domain: str
    text: str
    target: Optional[str] = None


class MultiDomainTextDataset(Dataset):
    """PyTorch Dataset for multi-domain text samples with causal LM tokenization and loss masking."""

    def __init__(
        self,
        samples: List[DomainSample],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ):
        """
        Args:
            samples: List of DomainSample objects.
            tokenizer: PreTrainedTokenizer instance.
            max_seq_length: Maximum sequence token length.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Format full sequence text
        full_text = sample.text
        if sample.target:
            full_text = f"{sample.text}\nAnswer: {sample.target}"

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # Create Causal LM labels (mask padding tokens with -100 for PyTorch CrossEntropyLoss)
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_multi_domain_dataloader(
    dataset: MultiDomainTextDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Create PyTorch DataLoader with automatic DistributedSampler if DDP is active.

    Args:
        dataset: MultiDomainTextDataset instance.
        batch_size: Per-device batch size.
        shuffle: Whether to shuffle dataset samples.
        num_workers: Data loading worker threads count.

    Returns:
        Tuple of (DataLoader, Optional[DistributedSampler]).
    """
    sampler = None
    if is_ddp_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling when active

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, sampler

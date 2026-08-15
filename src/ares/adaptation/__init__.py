"""Adaptation module exports for ARES V2 Phase 2."""

from ares.adaptation.dataset_loader import (
    DomainSample,
    MultiDomainTextDataset,
    create_multi_domain_dataloader,
)
from ares.adaptation.trainer import (
    setup_foundation_adapter,
    train_foundation_adapter,
)

__all__ = [
    "DomainSample",
    "MultiDomainTextDataset",
    "create_multi_domain_dataloader",
    "setup_foundation_adapter",
    "train_foundation_adapter",
]

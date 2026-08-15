"""Hidden state and activation extraction utilities for ARES V2 representation monitoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def pool_hidden_states(
    hidden_states: torch.Tensor,
    method: str = "last_token",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool hidden states across the sequence dimension (dim 1).

    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim).
        method: Pooling method ('last_token', 'mean', 'max').
        attention_mask: Tensor of shape (batch_size, seq_len) with 1 for valid tokens and 0 for padding.

    Returns:
        Tensor of shape (batch_size, hidden_dim).

    Raises:
        ValueError: If hidden_states shape is invalid or unsupported pooling method is specified.
    """
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected 3D hidden_states (batch, seq, dim), got shape {hidden_states.shape}")

    batch_size, seq_len, hidden_dim = hidden_states.shape

    if method == "last_token":
        if attention_mask is not None:
            # Find index of last non-padding token for each item in batch
            last_indices = (attention_mask.sum(dim=1) - 1).clamp(min=0).long()
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            return hidden_states[batch_indices, last_indices, :]
        else:
            return hidden_states[:, -1, :]

    elif method == "mean":
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask
        else:
            return hidden_states.mean(dim=1)

    elif method == "max":
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            return masked_hidden.max(dim=1).values
        else:
            return hidden_states.max(dim=1).values

    else:
        raise ValueError(f"Unknown pooling method: '{method}'. Supported methods: 'last_token', 'mean', 'max'")


def compute_logit_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute prediction entropy across vocabulary logits: H(P) = -sum(P * log(P)).

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (batch_size, vocab_size).

    Returns:
        Entropy tensor of shape (batch_size, seq_len) or (batch_size,).
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_prediction_margin(logits: torch.Tensor) -> torch.Tensor:
    """Compute top-1 vs top-2 logit margin: margin = max_1(logits) - max_2(logits).

    Args:
        logits: Logits tensor of shape (batch_size, seq_len, vocab_size) or (batch_size, vocab_size).

    Returns:
        Margin tensor of shape (batch_size, seq_len) or (batch_size,).
    """
    top2_values = torch.topk(logits, k=2, dim=-1).values
    margin = top2_values[..., 0] - top2_values[..., 1]
    return margin


class ActivationHookManager:
    """Registers and manages PyTorch forward hooks to capture intermediate layer activations."""

    def __init__(self, model: nn.Module, target_layer_names: List[str]):
        """
        Args:
            model: Target PyTorch module.
            target_layer_names: List of submodule layer names to attach hooks to (e.g. ['model.layers.12']).
        """
        self.model = model
        self.target_layer_names = target_layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_name: str):
        def hook(module: nn.Module, input: Any, output: Any):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            self.activations[layer_name] = act.detach()
        return hook

    def register_hooks(self) -> None:
        """Attach forward hooks to all specified target submodules."""
        self.clear()
        name_to_module = dict(self.model.named_modules())
        for layer_name in self.target_layer_names:
            if layer_name in name_to_module:
                handle = name_to_module[layer_name].register_forward_hook(self._make_hook(layer_name))
                self.handles.append(handle)
            else:
                raise KeyError(f"Target layer '{layer_name}' not found in model submodules!")

    def clear(self) -> None:
        """Clear cached activations dictionary."""
        self.activations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch forward hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
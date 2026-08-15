"""Unit tests for Phase 1 Qwen backbone loading, hidden state extraction, and activation pooling."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ares.backbone import (
    ActivationHookManager,
    compute_logit_entropy,
    compute_prediction_margin,
    get_model_input_device,
    pool_hidden_states,
    resolve_device,
)


class DummySubmodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DummySubmodule()
        self.layer2 = DummySubmodule()

    def get_input_embeddings(self):
        class DummyEmbeddings:
            weight = torch.nn.Parameter(torch.randn(10, 8))
        return DummyEmbeddings()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def test_hidden_extractor_pooling():
    batch_size = 2
    seq_len = 4
    hidden_dim = 8

    # Create synthetic hidden states tensor
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)

    # Test last_token pooling
    last_pooled = pool_hidden_states(hidden_states, method="last_token", attention_mask=mask)
    assert last_pooled.shape == (batch_size, hidden_dim)
    assert torch.allclose(last_pooled[0], hidden_states[0, 2, :])
    assert torch.allclose(last_pooled[1], hidden_states[1, 1, :])

    # Test mean pooling
    mean_pooled = pool_hidden_states(hidden_states, method="mean", attention_mask=mask)
    assert mean_pooled.shape == (batch_size, hidden_dim)

    # Test max pooling
    max_pooled = pool_hidden_states(hidden_states, method="max", attention_mask=mask)
    assert max_pooled.shape == (batch_size, hidden_dim)


def test_logit_entropy_and_margin():
    # Uniform logits -> Maximum entropy
    uniform_logits = torch.ones(1, 10) * 2.0
    uniform_entropy = compute_logit_entropy(uniform_logits)
    uniform_margin = compute_prediction_margin(uniform_logits)
    assert uniform_margin.item() == pytest.approx(0.0)

    # Peaked logits -> Low entropy, high margin
    peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    peaked_entropy = compute_logit_entropy(peaked_logits)
    peaked_margin = compute_prediction_margin(peaked_logits)

    assert peaked_entropy.item() < uniform_entropy.item()
    assert peaked_margin.item() == pytest.approx(10.0)


def test_activation_hook_manager():
    model = DummyModel()
    hook_mgr = ActivationHookManager(model, target_layer_names=["layer1", "layer2"])
    hook_mgr.register_hooks()

    x = torch.randn(2, 8)
    _ = model(x)

    assert "layer1" in hook_mgr.activations
    assert "layer2" in hook_mgr.activations
    assert hook_mgr.activations["layer1"].shape == (2, 8)

    hook_mgr.remove_hooks()
    assert len(hook_mgr.handles) == 0


def test_device_helpers():
    device = resolve_device()
    assert isinstance(device, torch.device)

    dummy_model = DummyModel()
    embed_dev = get_model_input_device(dummy_model)
    assert isinstance(embed_dev, torch.device)

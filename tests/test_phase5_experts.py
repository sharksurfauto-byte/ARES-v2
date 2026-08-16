"""Unit tests for Phase 5 Specialized Experts and Adaptive Expert Router."""

import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ares.experts.manager import ExpertManager
from ares.experts.trainer import ExpertTrainer
from ares.reliability.grm.model import GlobalReliabilityModel
from ares.reliability.lrm.model import LocalReliabilityModel
from ares.reliability.manager import ReliabilityManager
from ares.routing.router import AdaptiveExpertRouter, RoutingDecision


class DummyCausalLM(nn.Module):
    """Dummy causal LM for unit testing."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64):
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

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}


@pytest.fixture
def base_model():
    return DummyCausalLM(vocab_size=100, hidden_dim=64)


@pytest.fixture
def expert_manager(base_model):
    return ExpertManager(base_model=base_model, r=8, lora_alpha=16)


@pytest.fixture
def reliability_manager():
    grm = GlobalReliabilityModel(input_dim=64, bottleneck_dim=16)
    lrm = LocalReliabilityModel(input_dim=64, bottleneck_dim=16)
    return ReliabilityManager(grm=grm, lrm=lrm, confidence_threshold=0.70)


class TestExpertManager:
    """Test ExpertManager lifecycle and adapter management."""

    def test_expert_mapping(self, expert_manager):
        assert expert_manager.get_expert_for_domain("math") == "E1_math"
        assert expert_manager.get_expert_for_domain("code") == "E2_code"
        assert expert_manager.get_expert_for_domain("science") == "E3_science"
        assert expert_manager.get_expert_for_domain("general") == "E0_general"
        assert expert_manager.get_expert_for_domain("unknown") == "E0_general"

    def test_set_active_expert(self, expert_manager):
        expert_manager.set_active_expert("E1_math")
        assert expert_manager.active_expert == "E1_math"

        expert_manager.disable_experts()
        assert expert_manager.active_expert is None

    def test_save_expert_checkpoint(self, expert_manager, tmp_path):
        out_dir = expert_manager.save_expert_checkpoint(
            expert_name="E1_math",
            output_dir=tmp_path,
            metadata={"test": True},
        )
        assert out_dir.exists()
        assert (out_dir / "checkpoint_meta.json").exists()


class TestExpertTrainer:
    """Test ExpertTrainer training loop."""

    def test_trainer_epoch(self, expert_manager):
        dataset = TensorDataset(torch.randint(0, 100, (16, 8)), torch.randint(0, 100, (16, 8)))
        loader = DataLoader(dataset, batch_size=4)

        trainer = ExpertTrainer(
            expert_manager=expert_manager,
            expert_name="E1_math",
            lr=1e-3,
        )

        history = trainer.train(loader, epochs=2)
        assert len(history) == 2
        assert "train_loss" in history[0]
        assert history[0]["train_loss"] > 0


class TestAdaptiveExpertRouter:
    """Test AdaptiveExpertRouter dual-signal routing decision rules."""

    def test_router_adaptive_high_reliability(self, reliability_manager):
        router = AdaptiveExpertRouter(
            reliability_manager=reliability_manager,
            confidence_threshold=0.70,
            domain_certainty_threshold=0.35,
        )

        # Mock high reliability evaluation
        rep = torch.randn(64)
        decision = router.route(rep, policy="adaptive")

        assert isinstance(decision, RoutingDecision)
        assert decision.policy == "adaptive"
        assert isinstance(decision.requires_intervention, bool)
        assert decision.reason != ""

    def test_router_always_base_policy(self, reliability_manager):
        router = AdaptiveExpertRouter(reliability_manager=reliability_manager)
        rep = torch.randn(64)
        decision = router.route(rep, policy="always_base")

        assert decision.selected_expert is None
        assert not decision.requires_intervention
        assert "always_base" in decision.reason

    def test_router_always_expert_policy(self, reliability_manager):
        router = AdaptiveExpertRouter(reliability_manager=reliability_manager)
        rep = torch.randn(64)
        decision = router.route(rep, policy="always_expert", override_expert="E2_code")

        assert decision.selected_expert == "E2_code"
        assert decision.requires_intervention

    def test_router_random_expert_policy(self, reliability_manager):
        router = AdaptiveExpertRouter(reliability_manager=reliability_manager)
        rep = torch.randn(64)
        decision = router.route(rep, policy="random_expert")

        assert decision.selected_expert in ["E0_general", "E1_math", "E2_code", "E3_science"]
        assert decision.requires_intervention

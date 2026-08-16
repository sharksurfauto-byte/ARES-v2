"""Unit tests for Phase 4 Global and Local Reliability Models (GRM & LRM)."""

import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ares.reliability.grm.model import GlobalReliabilityModel
from ares.reliability.grm.trainer import GRMDataset, GRMTrainer
from ares.reliability.lrm.model import LocalReliabilityModel
from ares.reliability.lrm.trainer import LRMDataset, LRMTrainer
from ares.reliability.manager import ReliabilityManager, ReliabilityResult
from ares.representations.collector import RepresentationRecord


@pytest.fixture
def sample_records():
    """Create sample records for testing."""
    records = []
    domains = ["code", "general", "math", "science"]
    layers = [-1, -6, -12, -28]

    for i in range(20):
        domain = domains[i % len(domains)]
        layer = layers[i % len(layers)]
        rep = np.random.randn(128).astype(np.float32)

        rec = RepresentationRecord(
            sample_id=f"rec_{i}",
            domain=domain,
            task="test",
            layer=layer,
            representation=rep,
            logits=np.random.randn(50).astype(np.float32),
            prediction=f"token_{i}",
            correctness=(i % 2 == 0),
            confidence=0.8 if (i % 2 == 0) else 0.3,
            entropy=0.5 if (i % 2 == 0) else 1.8,
            margin=0.7 if (i % 2 == 0) else 0.1,
        )
        records.append(rec)
    return records


class TestGlobalReliabilityModel:
    """Test GlobalReliabilityModel architecture and forward pass."""

    def test_grm_forward_shapes(self):
        model = GlobalReliabilityModel(
            input_dim=128,
            bottleneck_dim=32,
            hidden_dim=64,
            num_domains=4,
            use_layer_depth_embedding=True,
            num_layers=32,
        )

        batch_size = 8
        x = torch.randn(batch_size, 128)
        layer_idx = torch.randint(0, 32, (batch_size,))

        outputs = model(x, layer_idx=layer_idx)

        assert "domain_logits" in outputs
        assert "domain_probs" in outputs
        assert "feasibility" in outputs
        assert "bottleneck" in outputs

        assert outputs["domain_logits"].shape == (batch_size, 4)
        assert outputs["domain_probs"].shape == (batch_size, 4)
        assert outputs["feasibility"].shape == (batch_size, 1)
        assert outputs["bottleneck"].shape == (batch_size, 32)

        # Probabilities sum to 1.0
        prob_sums = outputs["domain_probs"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

        # Feasibility in [0, 1]
        assert (outputs["feasibility"] >= 0.0).all() and (outputs["feasibility"] <= 1.0).all()

    def test_grm_single_vector(self):
        model = GlobalReliabilityModel(input_dim=128, bottleneck_dim=32)
        x = torch.randn(128)
        outputs = model(x, layer_idx=-1)

        assert outputs["domain_logits"].shape == (1, 4)
        assert outputs["feasibility"].shape == (1, 1)


class TestLocalReliabilityModel:
    """Test LocalReliabilityModel architecture and forward pass."""

    def test_lrm_forward_shapes(self):
        model = LocalReliabilityModel(
            input_dim=128,
            bottleneck_dim=16,
            hidden_dim=32,
            use_layer_depth_embedding=True,
            num_layers=32,
        )

        batch_size = 8
        x = torch.randn(batch_size, 128)
        layer_idx = torch.randint(0, 32, (batch_size,))

        outputs = model(x, layer_idx=layer_idx)

        assert "correctness_prob" in outputs
        assert "failure_risk" in outputs
        assert "bottleneck" in outputs

        assert outputs["correctness_prob"].shape == (batch_size, 1)
        assert outputs["failure_risk"].shape == (batch_size, 1)
        assert outputs["bottleneck"].shape == (batch_size, 16)

        # correctness_prob + failure_risk = 1.0
        sums = outputs["correctness_prob"] + outputs["failure_risk"]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestGRMTrainer:
    """Test GRMTrainer training loop."""

    def test_grm_training(self, sample_records):
        model = GlobalReliabilityModel(input_dim=128, bottleneck_dim=32, hidden_dim=64)
        dataset = GRMDataset(sample_records)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = GRMTrainer(model, lr=1e-3)
        history = trainer.train(dataloader, epochs=3)

        assert len(history) == 3
        assert "train_loss" in history[0]
        assert "train_domain_accuracy" in history[0]
        assert history[-1]["train_loss"] <= history[0]["train_loss"] + 0.5


class TestLRMTrainer:
    """Test LRMTrainer training loop."""

    def test_lrm_training(self, sample_records):
        model = LocalReliabilityModel(input_dim=128, bottleneck_dim=16, hidden_dim=32)
        dataset = LRMDataset(sample_records)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = LRMTrainer(model, lr=1e-3)
        history = trainer.train(dataloader, epochs=3)

        assert len(history) == 3
        assert "train_loss" in history[0]
        assert "train_accuracy" in history[0]


class TestReliabilityManager:
    """Test ReliabilityManager integration."""

    def test_manager_evaluation(self):
        grm = GlobalReliabilityModel(input_dim=128, bottleneck_dim=32)
        lrm = LocalReliabilityModel(input_dim=128, bottleneck_dim=16)

        manager = ReliabilityManager(
            grm=grm,
            lrm=lrm,
            aggregation_method="weighted_sum",
            weight_global=0.6,
            weight_local=0.4,
            confidence_threshold=0.7,
        )

        rep = torch.randn(128)
        res = manager.evaluate(rep, layer_idx=-1)

        assert isinstance(res, ReliabilityResult)
        assert 0.0 <= res.global_reliability <= 1.0
        assert 0.0 <= res.local_reliability <= 1.0
        assert 0.0 <= res.combined_reliability <= 1.0
        assert res.predicted_domain in ["code", "general", "math", "science"]
        assert len(res.domain_probabilities) == 4
        assert isinstance(res.is_reliable, bool)

    def test_manager_aggregation_methods(self):
        grm = GlobalReliabilityModel(input_dim=128, bottleneck_dim=32)
        lrm = LocalReliabilityModel(input_dim=128, bottleneck_dim=16)
        rep = torch.randn(128)

        # 1. Weighted sum
        mgr1 = ReliabilityManager(grm, lrm, aggregation_method="weighted_sum")
        res1 = mgr1.evaluate(rep)

        # 2. Gated min
        mgr2 = ReliabilityManager(grm, lrm, aggregation_method="gated_min")
        res2 = mgr2.evaluate(rep)
        assert res2.combined_reliability == min(res2.global_reliability, res2.local_reliability)

        # 3. Gated product
        mgr3 = ReliabilityManager(grm, lrm, aggregation_method="gated_product")
        res3 = mgr3.evaluate(rep)
        assert abs(res3.combined_reliability - (res3.global_reliability * res3.local_reliability)) < 1e-5

    def test_manager_checkpoint_saving(self, tmp_path):
        grm = GlobalReliabilityModel(input_dim=128, bottleneck_dim=32)
        lrm = LocalReliabilityModel(input_dim=128, bottleneck_dim=16)
        manager = ReliabilityManager(grm, lrm)

        out_dir = tmp_path / "reliability_ckpt"
        saved = manager.save_checkpoint(out_dir, metadata={"test": "true"})

        assert saved["grm"].exists()
        assert saved["lrm"].exists()
        assert (out_dir / "grm_model.json").exists()
        assert (out_dir / "lrm_model.json").exists()

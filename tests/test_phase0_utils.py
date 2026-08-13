"""Unit tests for Phase 0 infrastructure, configurations, seeds, device info, and checkpoint integrity."""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ares.utils import (
    ExperimentConfig,
    compute_file_sha256,
    get_device_info,
    load_checkpoint_with_validation,
    load_config,
    save_checkpoint_with_metadata,
    set_seed,
    verify_checkpoint_integrity,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_load_config():
    config_path = Path("configs/experiments/smoke_test.yaml")
    assert config_path.exists(), "smoke_test.yaml config file should exist"
    cfg = load_config(config_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment_name == "smoke_test"
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-0.5B"
    assert cfg.reliability.grm_enabled is True
    assert cfg.training.use_ddp is True


def test_set_seed():
    set_seed(42)
    val1 = torch.randn(5)
    set_seed(42)
    val2 = torch.randn(5)
    assert torch.allclose(val1, val2), "Seed management must produce deterministic PyTorch tensors"


def test_device_info():
    info = get_device_info()
    assert "cuda_available" in info
    assert "gpu_count" in info
    assert "gpus" in info


def test_checkpoint_integrity_and_validation():
    model = DummyModel()
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save valid checkpoint
        chk_dir = save_checkpoint_with_metadata(
            save_dir=tmp_dir,
            model=model,
            component_type="TEST_MODEL",
            metadata={"test_run": True},
        )

        meta_file = Path(chk_dir) / "checkpoint_meta.json"
        assert meta_file.exists()

        # Test loading valid checkpoint
        target_model = DummyModel()
        loaded_meta = load_checkpoint_with_validation(chk_dir, target_model)
        assert loaded_meta["component_type"] == "TEST_MODEL"

        # Verify strict loading failure on missing key
        invalid_state = {"unexpected_layer.weight": torch.randn(5, 5)}
        with pytest.raises(RuntimeError) as exc_info:
            verify_checkpoint_integrity(target_model, invalid_state)
        assert "Checkpoint integrity validation failed" in str(exc_info.value)

"""Configuration utilities for ARES V2.

Provides validated dataclasses and YAML parsers for configuration management across
models, data, training, reliability, experts, and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-7B"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    device_map: Optional[str] = None
    max_position_embeddings: int = 2048
    trust_remote_code: bool = False
    use_cache: bool = True
    attn_implementation: str = "eager"


@dataclass
class DataConfig:
    dataset_name: str = "default"
    data_dir: str = "datasets"
    domains: List[str] = field(default_factory=lambda: ["general", "math", "code", "science"])
    max_seq_length: int = 1024
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    # DDP / Multi-GPU configuration
    use_ddp: bool = True
    local_rank: int = -1
    world_size: int = 1


@dataclass
class ReliabilityConfig:
    grm_enabled: bool = True
    lrm_enabled: bool = True
    probe_type: str = "mlp"  # "linear", "mlp", "gbrt"
    hidden_layers: List[int] = field(default_factory=lambda: [-1, -6, -12])
    pooling_method: str = "last_token"  # "last_token", "mean", "max"
    target_metric: str = "auroc"


@dataclass
class ExpertConfig:
    enabled: bool = True
    expert_types: List[str] = field(
        default_factory=lambda: ["E0_general", "E1_knowledge", "E2_reasoning", "E3_code"]
    )
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class ExperimentConfig:
    experiment_name: str = "default_experiment"
    description: str = "Default ARES V2 experiment configuration"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    experts: ExpertConfig = field(default_factory=ExpertConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiate ExperimentConfig from a nested dictionary."""
        model_cfg = ModelConfig(**data.get("model", {}))
        data_cfg = DataConfig(**data.get("data", {}))
        training_cfg = TrainingConfig(**data.get("training", {}))
        reliability_cfg = ReliabilityConfig(**data.get("reliability", {}))
        expert_cfg = ExpertConfig(**data.get("experts", {}))

        return cls(
            experiment_name=data.get("experiment_name", "default_experiment"),
            description=data.get("description", ""),
            seed=data.get("seed", 42),
            model=model_cfg,
            data=data_cfg,
            training=training_cfg,
            reliability=reliability_cfg,
            experts=expert_cfg,
        )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and parse experiment YAML configuration file into an ExperimentConfig instance.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ExperimentConfig instance populated with parameters from YAML.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return ExperimentConfig.from_dict(raw_dict)

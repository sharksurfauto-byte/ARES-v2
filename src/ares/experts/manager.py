"""Domain-Specialized Expert Manager for ARES V2 (Phase 5).

Manages loading, attaching, switching, and disabling PEFT LoRA expert adapters
(E0: General, E1: Math, E2: Code, E3: Science) on top of the frozen Qwen backbone.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from ares.utils.checkpoint import save_checkpoint_with_metadata, load_checkpoint_with_validation


class ExpertManager:
    """Manager for domain-specialized LoRA experts attached to frozen Qwen backbone."""

    DOMAINS = ["general", "math", "code", "science"]
    EXPERT_MAP = {
        "general": "E0_general",
        "math": "E1_math",
        "code": "E2_code",
        "science": "E3_science",
    }

    def __init__(
        self,
        base_model: nn.Module,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        """Initialize ExpertManager.

        Args:
            base_model: Pretrained Qwen causal LM model.
            r: LoRA bottleneck rank.
            lora_alpha: LoRA scaling factor alpha.
            lora_dropout: LoRA dropout probability.
            target_modules: List of target linear module names (e.g. ['q_proj', 'v_proj']).
        """
        self.base_model = base_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        self.experts: Dict[str, Any] = {}
        self.active_expert: Optional[str] = None
        self.peft_model: Optional[Any] = None

        if PEFT_AVAILABLE:
            self._init_peft_structure()

    def _init_peft_structure(self) -> None:
        """Initialize PEFT LoRA adapter configurations for all domain experts."""
        first_adapter = self.EXPERT_MAP["general"]
        config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.base_model, config, adapter_name=first_adapter)
        self.experts[first_adapter] = config

        # Add remaining domain adapters
        for domain, adapter_name in self.EXPERT_MAP.items():
            if adapter_name != first_adapter:
                adapter_config = LoraConfig(
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=self.target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.peft_model.add_adapter(adapter_name, adapter_config)
                self.experts[adapter_name] = adapter_config

        self.active_expert = first_adapter

    def set_active_expert(self, expert_name: Optional[str]) -> None:
        """Set active LoRA adapter for inference.

        Args:
            expert_name: Name of expert ('E0_general', 'E1_math', etc.), or None to disable.
        """
        if not PEFT_AVAILABLE or self.peft_model is None:
            self.active_expert = expert_name
            return

        if expert_name is None or expert_name == "base":
            self.disable_experts()
            return

        if expert_name not in self.experts:
            raise ValueError(f"Unknown expert adapter '{expert_name}'. Available: {list(self.experts.keys())}")

        self.peft_model.set_adapter(expert_name)
        self.active_expert = expert_name

    def disable_experts(self) -> None:
        """Disable all expert adapters to revert to pure base Qwen model computation."""
        if PEFT_AVAILABLE and self.peft_model is not None:
            self.peft_model.disable_adapter_layers()
        self.active_expert = None

    def enable_experts(self) -> None:
        """Re-enable active expert adapter layers."""
        if PEFT_AVAILABLE and self.peft_model is not None and self.active_expert is not None:
            self.peft_model.enable_adapter_layers()

    def get_expert_for_domain(self, domain: str) -> str:
        """Map domain string to expert adapter identifier.

        Args:
            domain: Domain name ('general', 'math', 'code', 'science').

        Returns:
            Expert adapter key string (e.g. 'E1_math').
        """
        domain_clean = domain.lower().strip()
        return self.EXPERT_MAP.get(domain_clean, "E0_general")

    def save_expert_checkpoint(
        self,
        expert_name: str,
        output_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a specific domain expert checkpoint with SHA256 sidecar.

        Args:
            expert_name: Adapter name to save.
            output_dir: Directory path.
            metadata: Custom metadata dictionary.

        Returns:
            Saved directory path.
        """
        out_dir = Path(output_dir) / expert_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if PEFT_AVAILABLE and self.peft_model is not None:
            self.peft_model.save_pretrained(out_dir, selected_adapters=[expert_name])
            save_checkpoint_with_metadata(
                model=self.peft_model,
                save_dir=out_dir,
                component_type="EXPERT",
                metadata={"expert_name": expert_name, **(metadata or {})},
                weights_filename="adapter_model.bin" if (out_dir / "adapter_model.bin").exists() else "adapter_model.safetensors",
            )
        else:
            dummy_meta = {"expert_name": expert_name, "mock": True, **(metadata or {})}
            dummy_file = out_dir / "adapter_model.bin"
            torch.save({"dummy": torch.zeros(1)}, dummy_file)
            save_checkpoint_with_metadata(
                model=nn.Linear(1, 1),
                save_dir=out_dir,
                component_type="EXPERT",
                metadata=dummy_meta,
                weights_filename="adapter_model.bin",
            )

        return out_dir

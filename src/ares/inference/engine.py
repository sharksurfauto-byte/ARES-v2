"""ARES Inference Engine - Owns the complete ARES v2 pipeline.

Integrates Qwen backbone, GRM, LRM, ReliabilityManager, ExpertManager, and
AdaptiveExpertRouter into a unified inference engine with production-mode
checkpoint validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from ares.backbone.hidden_extractor import (
    ActivationHookManager,
    pool_hidden_states,
)
from ares.backbone.qwen_loader import (
    get_model_input_device,
    load_qwen_model,
    load_qwen_tokenizer,
)
from ares.experts.manager import ExpertManager
from ares.reliability.grm.model import GlobalReliabilityModel
from ares.reliability.lrm.model import LocalReliabilityModel
from ares.reliability.manager import ReliabilityManager, ReliabilityResult
from ares.routing.router import AdaptiveExpertRouter, RoutingDecision
from ares.utils.checkpoint import load_checkpoint_with_validation
from ares.utils.config import ModelConfig
from ares.utils.environment import set_seed


@dataclass
class CheckpointPaths:
    """Paths to required checkpoints for production mode."""
    grm: Path
    lrm: Path
    expert_dir: Path
    expert_names: List[str] = None

    def __post_init__(self):
        if self.expert_names is None:
            self.expert_names = ["E0_general", "E1_math", "E2_code", "E3_science"]

    def validate_all_exist(self) -> Tuple[bool, List[str]]:
        """Validate all required checkpoint files exist."""
        missing = []

        # Check GRM
        grm_weights = self.grm / "grm_model.pt"
        if not grm_weights.exists():
            missing.append(f"GRM weights: {grm_weights}")

        # Check LRM
        lrm_weights = self.lrm / "lrm_model.pt"
        if not lrm_weights.exists():
            missing.append(f"LRM weights: {lrm_weights}")

        # Check Experts
        for expert_name in self.expert_names:
            expert_path = self.expert_dir / expert_name
            # PEFT saves adapter_model.bin or adapter_model.safetensors
            adapter_bin = expert_path / "adapter_model.bin"
            adapter_safetensors = expert_path / "adapter_model.safetensors"
            if not adapter_bin.exists() and not adapter_safetensors.exists():
                missing.append(f"Expert {expert_name}: {expert_path}/adapter_model.(bin|safetensors)")

        return len(missing) == 0, missing


class ARESInferenceEngine:
    """Unified ARES v2 Inference Engine.

    Owns and manages:
    - Qwen backbone model + tokenizer
    - GRM (Global Reliability Model)
    - LRM (Local Reliability Model)
    - ReliabilityManager (aggregates GRM + LRM)
    - ExpertManager (4 domain-specialized LoRA adapters)
    - AdaptiveExpertRouter (dual-signal routing)
    - Hidden state extraction hooks

    Production mode REQUIRES all checkpoints to exist. No silent fallbacks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        grm_checkpoint: str = "checkpoints/reliability/grm",
        lrm_checkpoint: str = "checkpoints/reliability/lrm",
        expert_checkpoint_dir: str = "checkpoints/experts",
        confidence_threshold: float = 0.7,
        domain_certainty_threshold: float = 0.35,
        target_layer: int = -1,
        use_cache: bool = False,  # CRITICAL: Disable KV cache for correctness
        attn_implementation: str = "eager",  # CRITICAL: eager attention required for output_attentions=True
        seed: int = 42,
        production_mode: bool = True,
    ):
        """
        Args:
            model_name: Hugging Face model ID
            torch_dtype: Precision format
            device_map: Device placement strategy ('auto' for multi-GPU)
            grm_checkpoint: Path to GRM checkpoint directory
            lrm_checkpoint: Path to LRM checkpoint directory
            expert_checkpoint_dir: Path to directory containing 4 expert subdirectories
            confidence_threshold: T_confidence for routing (R(x) >= threshold → BASE)
            domain_certainty_threshold: T_domain for specialized vs fallback expert
            target_layer: Layer index for hidden state extraction (-1 = last)
            use_cache: Whether to use KV cache (FALSE for dynamic expert switching)
            attn_implementation: Attention implementation ('eager' for output_attentions=True)
            seed: Random seed for reproducibility
            production_mode: If True, require all checkpoints; else allow untrained

        Raises:
            FileNotFoundError: If production_mode=True and any checkpoint missing
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.confidence_threshold = confidence_threshold
        self.domain_certainty_threshold = domain_certainty_threshold
        self.target_layer = target_layer
        self.use_cache = use_cache
        self.attn_implementation = attn_implementation
        self.seed = seed
        self.production_mode = production_mode

        # Set seed for reproducibility
        set_seed(seed, deterministic=True)

        # Validate checkpoints in production mode
        checkpoint_paths = CheckpointPaths(
            grm=Path(grm_checkpoint),
            lrm=Path(lrm_checkpoint),
            expert_dir=Path(expert_checkpoint_dir),
        )
        all_exist, missing = checkpoint_paths.validate_all_exist()

        if production_mode and not all_exist:
            error_msg = (
                "ARES PRODUCTION MODE: Missing required checkpoints:\n"
                + "\n".join(f"  - {m}" for m in missing)
                + "\n\nRun in debug mode (production_mode=False) to use untrained probes, "
                "or ensure all checkpoints are present."
            )
            raise FileNotFoundError(error_msg)

        if not all_exist:
            print("WARNING: Running in DEBUG mode with missing checkpoints:")
            for m in missing:
                print(f"  - {m}")

        # 1. Load Qwen backbone + tokenizer (handles device_map="auto" correctly)
        print(f"Loading Qwen model: {model_name} (dtype: {torch_dtype}, device_map: {device_map}, attn: {attn_implementation})...")
        model_config = ModelConfig(
            name_or_path=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
        )
        self.tokenizer = load_qwen_tokenizer(model_config)
        self.model = load_qwen_model(model_config)

        # Store input device for tensor placement (handles sharded models)
        self.input_device = get_model_input_device(self.model)

        # 2. Initialize hidden state extraction hooks
        num_layers = self.model.config.num_hidden_layers
        if target_layer < 0:
            resolved_layer = num_layers + target_layer
        else:
            resolved_layer = target_layer

        if not (0 <= resolved_layer < num_layers):
            raise ValueError(
                f"target_layer {target_layer} (resolved: {resolved_layer}) "
                f"out of range for model with {num_layers} layers"
            )

        self.target_layer_name = f"model.layers.{resolved_layer}"
        self.hook_manager = ActivationHookManager(self.model, [self.target_layer_name])
        self.hook_manager.register_hooks()

        # 3. Load GRM + LRM → ReliabilityManager
        print("Loading reliability models (GRM + LRM)...")
        probe_num_layers = max(32, num_layers)
        dtype_attr = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        self.grm = GlobalReliabilityModel(
            input_dim=self.model.config.hidden_size,
            bottleneck_dim=128,
            hidden_dim=256,
            num_domains=4,
            dropout=0.1,
            use_layer_depth_embedding=True,
            num_layers=probe_num_layers,
        ).to(device=self.input_device, dtype=dtype_attr)

        self.lrm = LocalReliabilityModel(
            input_dim=self.model.config.hidden_size,
            bottleneck_dim=64,
            hidden_dim=128,
            dropout=0.1,
            use_layer_depth_embedding=True,
            num_layers=probe_num_layers,
        ).to(device=self.input_device, dtype=dtype_attr)

        if production_mode and all_exist:
            print(f"  Loading GRM from {grm_checkpoint}...")
            load_checkpoint_with_validation(
                grm_checkpoint, self.grm, weights_filename="grm_model.pt"
            )
            print(f"  Loading LRM from {lrm_checkpoint}...")
            load_checkpoint_with_validation(
                lrm_checkpoint, self.lrm, weights_filename="lrm_model.pt"
            )
        else:
            print("  Using untrained GRM/LRM probes (DEBUG mode)")

        self.reliability_manager = ReliabilityManager(
            grm=self.grm,
            lrm=self.lrm,
            aggregation_method="weighted_sum",
            weight_global=0.5,
            weight_local=0.5,
            confidence_threshold=confidence_threshold,
        )

        # 4. Load ExpertManager with 4 LoRA adapters
        print("Loading domain experts...")
        self.expert_manager = ExpertManager(
            base_model=self.model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        if production_mode and all_exist:
            expert_names = ["E0_general", "E1_math", "E2_code", "E3_science"]
            for expert_name in expert_names:
                expert_path = Path(expert_checkpoint_dir) / expert_name
                nested_path = expert_path / expert_name
                target_path = nested_path if (nested_path / "adapter_config.json").exists() else expert_path

                print(f"  Loading {expert_name} from {target_path}...")
                self.expert_manager.peft_model.load_adapter(
                    str(target_path), adapter_name=expert_name
                )
            print("  All 4 experts loaded successfully")
        else:
            print("  Using uninitialized expert adapters (DEBUG mode)")

        # 5. Create AdaptiveExpertRouter
        self.router = AdaptiveExpertRouter(
            reliability_manager=self.reliability_manager,
            confidence_threshold=confidence_threshold,
            domain_certainty_threshold=domain_certainty_threshold,
            default_policy="adaptive",
        )

        # State tracking
        self.active_expert: Optional[str] = None
        self.expert_activation_count = 0

        print("ARESInferenceEngine initialized successfully")

    def get_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract pooled hidden state from target layer.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Pooled hidden state of shape (batch, hidden_dim)
        """
        self.hook_manager.clear()

        # Forward pass with hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=self.use_cache,
        )

        # hidden_states[0] = embeddings, hidden_states[1] = layer 0, etc.
        layer_hidden = outputs.hidden_states[self.target_layer + 1]  # +1 for embeddings

        # Pool across sequence dimension
        pooled = pool_hidden_states(
            layer_hidden,
            method="last_token",
            attention_mask=attention_mask,
        )

        return pooled

    def evaluate_reliability(
        self,
        representation: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> ReliabilityResult:
        """Evaluate reliability metrics for input representation.

        Args:
            representation: Hidden state tensor (batch, hidden_dim) or (hidden_dim,)
            layer_idx: Layer index for depth embedding

        Returns:
            ReliabilityResult with global/local/combined reliability, domain, etc.
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        return self.reliability_manager.evaluate(representation, layer_idx=layer_idx)

    def route(
        self,
        representation: torch.Tensor,
        layer_idx: Optional[int] = None,
        policy: Optional[str] = None,
    ) -> RoutingDecision:
        """Compute routing decision for input representation.

        Args:
            representation: Hidden state tensor
            layer_idx: Layer index
            policy: Routing policy override

        Returns:
            RoutingDecision with selected expert, intervention flag, reason
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        return self.router.route(
            representation,
            layer_idx=layer_idx,
            policy=policy,
        )

    def set_active_expert(self, expert_name: Optional[str]) -> None:
        """Set active expert adapter for generation.

        Args:
            expert_name: Expert adapter name ('E0_general', 'E1_math', etc.) or None for BASE
        """
        if expert_name != self.active_expert:
            self.expert_manager.set_active_expert(expert_name)
            self.active_expert = expert_name

    def forward_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Forward pass with BASE Qwen (no expert adapters active).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            output_attentions: Whether to capture self-attention tensors

        Returns:
            Logits tensor or (logits, attentions_tuple) if output_attentions is True
        """
        # Ensure no expert is active
        was_active = self.active_expert is not None
        if was_active:
            self.expert_manager.disable_experts()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=self.use_cache,
            output_attentions=output_attentions,
        )

        # Restore expert if it was active
        if was_active:
            self.expert_manager.enable_experts()

        if output_attentions:
            return outputs.logits, outputs.attentions
        return outputs.logits

    def forward_expert(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass with specified EXPERT adapter active.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            expert_name: Expert adapter name (uses active if None)

        Returns:
            Logits tensor (batch, seq_len, vocab_size)
        """
        if expert_name is not None:
            self.set_active_expert(expert_name)

        # Ensure expert is enabled
        self.expert_manager.enable_experts()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=self.use_cache,
        )

        return outputs.logits

    def forward_with_active_expert(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with currently active expert (or base if none).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Logits tensor
        """
        if self.active_expert is None:
            return self.forward_base(input_ids, attention_mask)
        else:
            return self.forward_expert(input_ids, attention_mask, self.active_expert)

    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[int, str]:
        """Sample next token from logits.

        Args:
            logits: Logits tensor (batch, seq_len, vocab_size) or (vocab_size,)
            temperature: Sampling temperature
            do_sample: Whether to sample (True) or greedy (False)

        Returns:
            Tuple of (token_id, decoded_token_string)
        """
        if logits.ndim == 3:
            next_token_logits = logits[0, -1, :]  # Last token, first batch
        else:
            next_token_logits = logits[-1, :] if logits.ndim == 2 else logits

        if do_sample:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            token_id = next_token_logits.argmax(dim=-1).item()

        token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
        return token_id, token_str

    def cleanup(self) -> None:
        """Clean up hooks and resources."""
        self.hook_manager.remove_hooks()

    def __enter__(self) -> "ARESInferenceEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def create_engine_from_config(config: Dict[str, Any]) -> ARESInferenceEngine:
    """Factory function to create engine from config dict."""
    inf_cfg = config.get("inference", config)

    return ARESInferenceEngine(
        model_name=inf_cfg.get("model", {}).get("name_or_path", "Qwen/Qwen2.5-7B"),
        torch_dtype=inf_cfg.get("model", {}).get("torch_dtype", "bfloat16"),
        device_map=inf_cfg.get("model", {}).get("device_map", "auto"),
        grm_checkpoint=inf_cfg.get("reliability", {}).get("grm_checkpoint", "checkpoints/reliability/grm"),
        lrm_checkpoint=inf_cfg.get("reliability", {}).get("lrm_checkpoint", "checkpoints/reliability/lrm"),
        expert_checkpoint_dir=inf_cfg.get("experts", {}).get("checkpoint_dir", "checkpoints/experts"),
        confidence_threshold=inf_cfg.get("router", {}).get("confidence_threshold", 0.7),
        domain_certainty_threshold=inf_cfg.get("router", {}).get("domain_certainty_threshold", 0.35),
        target_layer=inf_cfg.get("generation", {}).get("target_layer", -1),
        use_cache=inf_cfg.get("model", {}).get("use_cache", False),
        seed=inf_cfg.get("generation", {}).get("seed", 42),
        production_mode=True,
    )
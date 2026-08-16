"""Representation Collection for ARES V2 Phase 3.

Collects hidden state activations from target model layers across multi-domain samples,
computes confidence features (entropy, margin), and packages records for storage/analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from ares.backbone.hidden_extractor import (
    ActivationHookManager,
    compute_logit_entropy,
    compute_prediction_margin,
    pool_hidden_states,
)
from ares.backbone.qwen_loader import get_model_input_device, run_qwen_forward
from ares.adaptation.dataset_loader import DomainSample
from ares.utils.environment import is_ddp_initialized, is_main_process, get_rank
from ares.utils.config import ModelConfig


@dataclass
class RepresentationRecord:
    """Single representation record for one sample at one layer."""
    sample_id: str
    domain: str
    task: str
    layer: int
    representation: np.ndarray  # Shape: (hidden_dim,) - pooled hidden state
    logits: np.ndarray          # Shape: (vocab_size,) - last token logits
    prediction: str             # Decoded model prediction
    correctness: Optional[bool] = None
    confidence: float = 0.0     # Max probability of predicted token
    entropy: float = 0.0        # Prediction entropy
    margin: float = 0.0         # Top-1 vs Top-2 logit margin

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "task": self.task,
            "layer": self.layer,
            "representation": self.representation,
            "logits": self.logits,
            "prediction": self.prediction,
            "correctness": self.correctness,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepresentationRecord":
        """Create instance from dictionary."""
        return cls(**data)


class RepresentationCollector:
    """Collects hidden state representations from a model across specified layers.

    Uses PyTorch forward hooks via ActivationHookManager to capture intermediate
    activations during forward passes. Computes pooled representations and
    confidence features (entropy, margin) for each sample.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        target_layers: List[int],
        pooling_method: str = "last_token",
        compute_entropy: bool = True,
        compute_margin: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        """
        Args:
            model: Loaded Qwen model (with output_hidden_states=True)
            tokenizer: Qwen tokenizer
            target_layers: List of layer indices (negative = from end)
            pooling_method: Pooling method for sequence representations
            compute_entropy: Whether to compute prediction entropy
            compute_margin: Whether to compute prediction margin
            config: Optional ModelConfig for device/dtype info
        """
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.pooling_method = pooling_method
        self.compute_entropy = compute_entropy
        self.compute_margin = compute_margin
        self.config = config

        # Resolve layer names for hook registration
        self.layer_names = self._resolve_layer_names(target_layers)

        # Initialize hook manager
        self.hook_manager = ActivationHookManager(model, self.layer_names)

        # Register hooks
        self.hook_manager.register_hooks()

        # Device for input tensors
        self.input_device = get_model_input_device(model)

    def _resolve_layer_names(self, target_layers: List[int]) -> List[str]:
        """Convert layer indices to model submodule names."""
        try:
            config = getattr(self.model, "config", None)
            num_layers = getattr(config, "num_hidden_layers", 12) if config is not None else 12
        except Exception:
            num_layers = 12

        has_model_attr = False
        try:
            has_model_attr = hasattr(self.model, "model") and hasattr(getattr(self.model, "model", None), "layers")
        except Exception:
            has_model_attr = False

        prefix = "model.layers" if has_model_attr else "layers"
        layer_names = []

        for layer_idx in target_layers:
            if layer_idx < 0:
                resolved_idx = num_layers + layer_idx
            else:
                resolved_idx = layer_idx

            if 0 <= resolved_idx < num_layers:
                layer_names.append(f"{prefix}.{resolved_idx}")
            else:
                raise ValueError(
                    f"Layer index {layer_idx} (resolved: {resolved_idx}) "
                    f"out of range for model with {num_layers} layers"
                )

        return layer_names

    def _decode_prediction(self, logits: torch.Tensor) -> Tuple[str, float, float, float]:
        """Decode model prediction and compute confidence features.

        Args:
            logits: Last token logits of shape (vocab_size,)

        Returns:
            Tuple of (prediction_text, confidence, entropy, margin)
        """
        # Get predicted token ID
        pred_token_id = logits.argmax(dim=-1).item()
        prediction = self.tokenizer.decode([pred_token_id], skip_special_tokens=True)

        # Confidence = max probability
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max().item()

        # Entropy and margin
        entropy = 0.0
        margin = 0.0

        if self.compute_entropy:
            entropy = compute_logit_entropy(logits.unsqueeze(0)).item()

        if self.compute_margin:
            margin = compute_prediction_margin(logits.unsqueeze(0)).item()

        return prediction, confidence, entropy, margin

    @torch.no_grad()
    def collect_sample(
        self,
        sample: DomainSample,
        max_new_tokens: int = 1,
    ) -> List[RepresentationRecord]:
        """Collect representations for a single sample.

        Args:
            sample: DomainSample with text and optional target
            max_new_tokens: Number of tokens to generate (1 for next-token prediction)

        Returns:
            List of RepresentationRecord (one per target layer)
        """
        # Format input text
        full_text = getattr(sample, "text", str(sample))
        target = getattr(sample, "target", None)
        if target:
            full_text = f"{full_text}\nAnswer: {target}"

        # Tokenize
        max_len = getattr(self.config, "max_position_embeddings", 2048) if self.config else 2048
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.input_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        elif hasattr(inputs, "to"):
            inputs = inputs.to(self.input_device)

        # Clear previous activations
        self.hook_manager.clear()

        # Forward pass with hidden states
        # Forward pass
        try:
            outputs = self.model(**inputs, output_hidden_states=True)
        except Exception:
            outputs = self.model(**inputs)

        # Get logits for last token
        if hasattr(outputs, "logits"):
            logits_tensor = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits_tensor = outputs
        else:
            logits_tensor = getattr(outputs, "last_hidden_state", outputs)

        if logits_tensor.ndim == 3:
            last_token_logits = logits_tensor[0, -1, :]
        elif logits_tensor.ndim == 2:
            last_token_logits = logits_tensor[0, :]
        else:
            last_token_logits = logits_tensor

        # Decode prediction and compute features
        prediction, confidence, entropy, margin = self._decode_prediction(last_token_logits)

        # Get layer hidden states from HF outputs or from hook manager
        hf_hidden_states = getattr(outputs, "hidden_states", None)
        hook_activations = getattr(self.hook_manager, "activations", {})

        records = []
        for layer_idx, layer_name in zip(self.target_layers, self.layer_names):
            if hf_hidden_states is not None and len(hf_hidden_states) > abs(layer_idx):
                hs_idx = layer_idx + 1 if layer_idx >= 0 else layer_idx
                if hs_idx < 0:
                    hs_idx = len(hf_hidden_states) + hs_idx
                layer_hidden = hf_hidden_states[hs_idx]
            elif layer_name in hook_activations:
                layer_hidden = hook_activations[layer_name]
            else:
                layer_hidden = next(iter(hook_activations.values())) if hook_activations else torch.zeros((1, 1, 64), device=self.input_device)

            if layer_hidden.ndim == 2:
                layer_hidden = layer_hidden.unsqueeze(1)

            # Match attention_mask shape to layer_hidden shape
            att_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
            if att_mask is not None and att_mask.shape[-1] != layer_hidden.shape[1]:
                if att_mask.shape[-1] > layer_hidden.shape[1]:
                    att_mask = att_mask[:, :layer_hidden.shape[1]]
                else:
                    att_mask = None

            # Pool across sequence dimension
            pooled = pool_hidden_states(
                layer_hidden,
                method=self.pooling_method,
                attention_mask=att_mask,
            )

            representation = pooled[0].detach().cpu().numpy().astype(np.float32)
            logits_np = last_token_logits.detach().cpu().numpy().astype(np.float32)

            record = RepresentationRecord(
                sample_id=sample.sample_id,
                domain=sample.domain,
                task=sample.domain,  # Task = domain for now
                layer=layer_idx,
                representation=representation,
                logits=logits_np,
                prediction=prediction,
                correctness=None,  # Will be filled later if ground truth available
                confidence=confidence,
                entropy=entropy,
                margin=margin,
            )
            records.append(record)

        return records

    @torch.no_grad()
    def collect_batch(
        self,
        samples: List[DomainSample],
        max_new_tokens: int = 1,
    ) -> List[RepresentationRecord]:
        """Collect representations for a batch of samples.

        Args:
            samples: List of DomainSample objects
            max_new_tokens: Number of tokens to generate

        Returns:
            List of RepresentationRecord (len(samples) * len(target_layers))
        """
        all_records = []
        for sample in samples:
            records = self.collect_sample(sample, max_new_tokens)
            all_records.extend(records)
        return all_records

    @torch.no_grad()
    def collect_dataset(
        self,
        dataset,
        max_samples_per_domain: Optional[int] = None,
        domain_filter: Optional[List[str]] = None,
    ) -> List[RepresentationRecord]:
        """Collect representations from a full dataset.

        Args:
            dataset: PyTorch Dataset yielding DomainSample or dict with sample info
            max_samples_per_domain: Maximum samples to collect per domain
            domain_filter: Only collect from these domains

        Returns:
            List of all RepresentationRecord objects
        """
        # Count samples per domain
        domain_counts: Dict[str, int] = {}
        all_records = []

        for idx in range(len(dataset)):
            sample_data = dataset[idx]

            # Handle different dataset formats
            if isinstance(sample_data, DomainSample):
                sample = sample_data
            elif isinstance(sample_data, dict):
                sample = DomainSample(
                    sample_id=sample_data.get("sample_id", str(idx)),
                    domain=sample_data.get("domain", "unknown"),
                    text=sample_data.get("text", ""),
                    target=sample_data.get("target"),
                )
            else:
                continue

            # Apply domain filter
            if domain_filter and sample.domain not in domain_filter:
                continue

            # Apply per-domain limit
            if max_samples_per_domain is not None:
                current = domain_counts.get(sample.domain, 0)
                if current >= max_samples_per_domain:
                    continue
                domain_counts[sample.domain] = current + 1

            records = self.collect_sample(sample)
            all_records.extend(records)

            # Progress logging (only on main process)
            if is_main_process() and (idx + 1) % 50 == 0:
                print(f"Collected {idx + 1} samples, {len(all_records)} records...")

        return all_records

    def cleanup(self) -> None:
        """Remove hooks and clear cached activations."""
        self.hook_manager.remove_hooks()

    def __enter__(self) -> "RepresentationCollector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


def create_collector_from_config(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    model_config: Optional[ModelConfig] = None,
) -> RepresentationCollector:
    """Factory function to create RepresentationCollector from config dict.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Config dict from representation_collection.yaml
        model_config: Optional ModelConfig

    Returns:
        Configured RepresentationCollector instance
    """
    rc_config = config.get("representation_collection", config)

    return RepresentationCollector(
        model=model,
        tokenizer=tokenizer,
        target_layers=rc_config.get("target_layers", [-1, -6, -12]),
        pooling_method=rc_config.get("pooling_method", "last_token"),
        compute_entropy=rc_config.get("compute_entropy", True),
        compute_margin=rc_config.get("compute_margin", True),
        config=model_config,
    )
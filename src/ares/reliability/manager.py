"""Reliability Manager for ARES V2 Phase 4.

Integrates GRM (Global Reliability Model) and LRM (Local Reliability Model)
into a unified reliability evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from ares.reliability.grm.model import GlobalReliabilityModel
from ares.reliability.lrm.model import LocalReliabilityModel
from ares.utils.checkpoint import (
    save_checkpoint_with_metadata,
    load_checkpoint_with_validation,
)


@dataclass
class ReliabilityResult:
    """Dataclass holding outputs from ReliabilityManager."""

    global_reliability: float
    local_reliability: float
    combined_reliability: float
    predicted_domain: str
    domain_probabilities: Dict[str, float]
    failure_risk: float
    confidence_threshold: float
    is_reliable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_reliability": self.global_reliability,
            "local_reliability": self.local_reliability,
            "combined_reliability": self.combined_reliability,
            "predicted_domain": self.predicted_domain,
            "domain_probabilities": self.domain_probabilities,
            "failure_risk": self.failure_risk,
            "confidence_threshold": self.confidence_threshold,
            "is_reliable": self.is_reliable,
        }


class ReliabilityManager(nn.Module):
    """Unified Reliability Evaluation Manager.

    Integrates GRM and LRM probes and provides flexible, evidence-based
    aggregation algorithms.
    """

    DOMAINS = ["code", "general", "math", "science"]

    def __init__(
        self,
        grm: GlobalReliabilityModel,
        lrm: LocalReliabilityModel,
        aggregation_method: str = "weighted_sum",
        weight_global: float = 0.5,
        weight_local: float = 0.5,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()
        self.grm = grm
        self.lrm = lrm
        self.aggregation_method = aggregation_method
        self.weight_global = weight_global
        self.weight_local = weight_local
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def evaluate(
        self,
        representation: torch.Tensor,
        layer_idx: Optional[Union[int, torch.Tensor]] = None,
    ) -> ReliabilityResult:
        """Evaluate reliability metrics for input representation.

        Args:
            representation: Input vector of shape (input_dim,) or (batch_size, input_dim)
            layer_idx: Layer index or tensor

        Returns:
            ReliabilityResult instance
        """
        self.grm.eval()
        self.lrm.eval()

        device = next(self.grm.parameters()).device
        if representation.device != device:
            representation = representation.to(device)

        if representation.ndim == 1:
            representation = representation.unsqueeze(0)

        # 1. Run GRM
        grm_out = self.grm(representation, layer_idx=layer_idx)
        r_global = float(grm_out["feasibility"][0, 0].item())
        domain_probs_raw = grm_out["domain_probs"][0].float().cpu().numpy()

        domain_probs = {
            domain: float(domain_probs_raw[i])
            for i, domain in enumerate(self.DOMAINS[:len(domain_probs_raw)])
        }
        pred_domain_idx = int(grm_out["domain_logits"][0].argmax(dim=-1).item())
        pred_domain = self.DOMAINS[pred_domain_idx] if pred_domain_idx < len(self.DOMAINS) else "general"

        # 2. Run LRM
        lrm_out = self.lrm(representation, layer_idx=layer_idx)
        r_local = lrm_out["correctness_prob"][0, 0].item()
        failure_risk = lrm_out["failure_risk"][0, 0].item()

        # 3. Aggregate reliability scores
        if self.aggregation_method == "weighted_sum":
            w_sum = self.weight_global + self.weight_local
            w_g = self.weight_global / max(w_sum, 1e-6)
            w_l = self.weight_local / max(w_sum, 1e-6)
            r_combined = w_g * r_global + w_l * r_local
        elif self.aggregation_method == "gated_min":
            r_combined = min(r_global, r_local)
        elif self.aggregation_method == "gated_product":
            r_combined = r_global * r_local
        else:
            r_combined = 0.5 * r_global + 0.5 * r_local

        is_reliable = bool(r_combined >= self.confidence_threshold)

        return ReliabilityResult(
            global_reliability=r_global,
            local_reliability=r_local,
            combined_reliability=r_combined,
            predicted_domain=pred_domain,
            domain_probabilities=domain_probs,
            failure_risk=failure_risk,
            confidence_threshold=self.confidence_threshold,
            is_reliable=is_reliable,
        )

    def save_checkpoint(
        self,
        output_dir: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Save GRM and LRM weights with SHA256 sidecars."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        grm_dir = out_dir / "grm"
        lrm_dir = out_dir / "lrm"

        save_checkpoint_with_metadata(
            model=self.grm,
            save_dir=grm_dir,
            component_type="RELIABILITY_MODEL",
            metadata={"model_type": "GRM", **(metadata or {})},
            weights_filename="grm_model.pt",
        )
        save_checkpoint_with_metadata(
            model=self.lrm,
            save_dir=lrm_dir,
            component_type="RELIABILITY_MODEL",
            metadata={"model_type": "LRM", **(metadata or {})},
            weights_filename="lrm_model.pt",
        )

        return {"grm": grm_dir / "grm_model.pt", "lrm": lrm_dir / "lrm_model.pt"}

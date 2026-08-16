"""Adaptive Expert Router for ARES V2 (Phase 5).

Implements dual-signal routing decision logic based on GRM domain probabilities
and LRM local reliability scores to selectively activate specialized computation.
"""

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Union
import torch

from ares.reliability.manager import ReliabilityManager, ReliabilityResult


@dataclass
class RoutingDecision:
    """Detailed trace object representing an adaptive routing decision."""

    policy: str
    selected_expert: Optional[str]
    requires_intervention: bool
    reliability_score: float
    predicted_domain: str
    domain_confidence: float
    is_domain_certain: bool
    reason: str


class AdaptiveExpertRouter:
    """Dual-signal Adaptive Expert Router."""

    DOMAINS = ["general", "math", "code", "science"]
    EXPERT_MAP = {
        "general": "E0_general",
        "math": "E1_math",
        "code": "E2_code",
        "science": "E3_science",
    }

    def __init__(
        self,
        reliability_manager: ReliabilityManager,
        confidence_threshold: float = 0.70,
        domain_certainty_threshold: float = 0.35,
        default_policy: str = "adaptive",
    ):
        """Initialize AdaptiveExpertRouter.

        Args:
            reliability_manager: Instantiated ReliabilityManager.
            confidence_threshold: T_confidence threshold. R(x) >= threshold routes to Base model.
            domain_certainty_threshold: T_domain threshold for specialized expert vs fallback E0.
            default_policy: Routing policy ('adaptive', 'always_base', 'always_expert', 'random_expert').
        """
        self.reliability_manager = reliability_manager
        self.confidence_threshold = confidence_threshold
        self.domain_certainty_threshold = domain_certainty_threshold
        self.default_policy = default_policy

    def route(
        self,
        representation: torch.Tensor,
        layer_idx: Optional[Union[int, torch.Tensor]] = -1,
        policy: Optional[str] = None,
        override_expert: Optional[str] = None,
    ) -> RoutingDecision:
        """Compute routing decision for an input hidden state representation.

        Args:
            representation: Input tensor of shape (hidden_dim,) or (batch, hidden_dim).
            layer_idx: Layer index for depth embedding.
            policy: Optional policy override ('adaptive', 'always_base', 'always_expert', 'random_expert').
            override_expert: Explicit expert override.

        Returns:
            RoutingDecision instance containing full decision trace.
        """
        pol = policy or self.default_policy

        # Evaluate reliability metrics
        rel_result = self.reliability_manager.evaluate(representation, layer_idx=layer_idx)
        r_x = rel_result.combined_reliability
        pred_domain = rel_result.predicted_domain
        domain_probs = rel_result.domain_probabilities
        p_max = max(domain_probs.values()) if domain_probs else 0.25

        is_certain = p_max >= self.domain_certainty_threshold
        target_expert = self.EXPERT_MAP.get(pred_domain, "E0_general")

        # 1. Policy: always_base
        if pol == "always_base":
            return RoutingDecision(
                policy=pol,
                selected_expert=None,
                requires_intervention=False,
                reliability_score=r_x,
                predicted_domain=pred_domain,
                domain_confidence=p_max,
                is_domain_certain=is_certain,
                reason="Policy 'always_base': Routed to Base Qwen (0 extra compute).",
            )

        # 2. Policy: random_expert
        if pol == "random_expert":
            rand_expert = random.choice(list(self.EXPERT_MAP.values()))
            return RoutingDecision(
                policy=pol,
                selected_expert=rand_expert,
                requires_intervention=True,
                reliability_score=r_x,
                predicted_domain=pred_domain,
                domain_confidence=p_max,
                is_domain_certain=is_certain,
                reason=f"Policy 'random_expert': Randomly selected '{rand_expert}'.",
            )

        # 3. Policy: always_expert
        if pol == "always_expert":
            chosen = override_expert or target_expert
            return RoutingDecision(
                policy=pol,
                selected_expert=chosen,
                requires_intervention=True,
                reliability_score=r_x,
                predicted_domain=pred_domain,
                domain_confidence=p_max,
                is_domain_certain=is_certain,
                reason=f"Policy 'always_expert': Activated expert '{chosen}'.",
            )

        # 4. Policy: adaptive (Dual-Signal Decision Tree)
        if r_x >= self.confidence_threshold:
            # High reliability -> Base Qwen
            return RoutingDecision(
                policy=pol,
                selected_expert=None,
                requires_intervention=False,
                reliability_score=r_x,
                predicted_domain=pred_domain,
                domain_confidence=p_max,
                is_domain_certain=is_certain,
                reason=f"High Reliability (R(x)={r_x:.3f} >= {self.confidence_threshold}): Routed to Base Qwen (0 extra compute).",
            )
        else:
            # Low reliability -> Expert Intervention
            if is_certain:
                chosen_expert = target_expert
                reason_str = (
                    f"Low Reliability (R(x)={r_x:.3f} < {self.confidence_threshold}) + "
                    f"Confident Domain (P({pred_domain})={p_max:.3f} >= {self.domain_certainty_threshold}): "
                    f"Activated specialized expert '{chosen_expert}'."
                )
            else:
                chosen_expert = "E0_general"
                reason_str = (
                    f"Low Reliability (R(x)={r_x:.3f} < {self.confidence_threshold}) + "
                    f"Ambiguous Domain (P_max={p_max:.3f} < {self.domain_certainty_threshold}): "
                    f"Activated fallback expert 'E0_general'."
                )

            return RoutingDecision(
                policy=pol,
                selected_expert=chosen_expert,
                requires_intervention=True,
                reliability_score=r_x,
                predicted_domain=pred_domain,
                domain_confidence=p_max,
                is_domain_certain=is_certain,
                reason=reason_str,
            )

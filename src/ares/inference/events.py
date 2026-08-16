"""Inference Event Definitions for ARES V2 Phase 6.

Structured events yielded per generated token for UI consumption, telemetry logging,
and Phase 6 evaluation. Every field corresponds to a real computation in the ARES pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class InferenceEvent:
    """Single token generation event with full ARES pipeline trace.

    This event is yielded by `generate_stream()` for each token produced.
    All fields represent actual computed values from the ARES pipeline:
    Qwen → Hidden State → GRM/LRM → ReliabilityManager → AdaptiveExpertRouter
    → ExpertManager → Next Token.

    Attributes:
        # Token identification
        token: The decoded token string (may be partial for streaming)
        token_id: The integer token ID from the vocabulary
        token_index: 0-based index within the generated sequence (excludes prompt)
        sequence_position: Absolute position in full sequence (prompt + generation)
        is_prompt_token: True if this token is from the input prompt

        # Reliability metrics (from ReliabilityManager)
        global_reliability: GRM global feasibility score R_global ∈ [0,1]
        local_reliability: LRM local correctness probability R_local ∈ [0,1]
        combined_reliability: Aggregated R(x) ∈ [0,1] (weighted sum / gated)
        failure_risk: LRM failure risk = 1 - local_reliability
        is_reliable: True if combined_reliability >= confidence_threshold

        # Domain classification (from GRM)
        predicted_domain: Domain with highest probability ('general', 'math', 'code', 'science')
        domain_probabilities: Full probability distribution over 4 domains
        domain_confidence: Max domain probability (P_max)
        is_domain_certain: True if domain_confidence >= domain_certainty_threshold

        # Routing decision (from AdaptiveExpertRouter)
        policy: Routing policy used ('adaptive', 'always_base', 'always_expert', 'random_expert')
        selected_expert: Expert adapter name ('E0_general', 'E1_math', 'E2_code', 'E3_science') or None for BASE_QWEN
        requires_intervention: True if expert was activated (selected_expert is not None)
        routing_reason: Human-readable explanation of the routing decision

        # Latency breakdown (CRITICAL for demo honesty)
        routing_latency_ms: Time for base forward + GRM/LRM + Router (ms)
        expert_latency_ms: Time for expert forward pass (0.0 if BASE)
        total_latency_ms: Sum of routing_latency_ms + expert_latency_ms

        # Logits provenance (which model produced the token)
        base_logits_used: True if token was sampled from base Qwen logits
        expert_logits_used: True if token was sampled from expert logits

        # Cumulative expert tracking
        expert_activation_count: Cumulative count of expert activations so far

        # Metadata
        timestamp: Wall-clock timestamp (time.time())
        layer_idx: Layer index used for hidden state extraction (e.g., -1)
    """

    # Token identification
    token: str
    token_id: int
    token_index: int
    sequence_position: int
    is_prompt_token: bool

    # Reliability metrics
    global_reliability: float
    local_reliability: float
    combined_reliability: float
    failure_risk: float
    is_reliable: bool

    # Domain classification
    predicted_domain: str
    domain_probabilities: Dict[str, float]
    domain_confidence: float
    is_domain_certain: bool

    # Routing decision
    policy: str
    selected_expert: Optional[str]
    requires_intervention: bool
    routing_reason: str

    # Latency breakdown
    routing_latency_ms: float
    expert_latency_ms: float
    total_latency_ms: float

    # Logits provenance
    base_logits_used: bool
    expert_logits_used: bool

    # Cumulative tracking
    expert_activation_count: int

    # Metadata
    timestamp: float
    layer_idx: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (JSON/CSV)."""
        return {
            "token": self.token,
            "token_id": self.token_id,
            "token_index": self.token_index,
            "sequence_position": self.sequence_position,
            "is_prompt_token": self.is_prompt_token,
            "global_reliability": self.global_reliability,
            "local_reliability": self.local_reliability,
            "combined_reliability": self.combined_reliability,
            "failure_risk": self.failure_risk,
            "is_reliable": self.is_reliable,
            "predicted_domain": self.predicted_domain,
            "domain_probabilities": self.domain_probabilities,
            "domain_confidence": self.domain_confidence,
            "is_domain_certain": self.is_domain_certain,
            "policy": self.policy,
            "selected_expert": self.selected_expert,
            "requires_intervention": self.requires_intervention,
            "routing_reason": self.routing_reason,
            "routing_latency_ms": self.routing_latency_ms,
            "expert_latency_ms": self.expert_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "base_logits_used": self.base_logits_used,
            "expert_logits_used": self.expert_logits_used,
            "expert_activation_count": self.expert_activation_count,
            "timestamp": self.timestamp,
            "layer_idx": self.layer_idx,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceEvent":
        """Create instance from dictionary."""
        return cls(**data)

    def get_expert_display_name(self) -> str:
        """Get display name for expert (for UI)."""
        if self.selected_expert is None:
            return "BASE_QWEN"
        return self.selected_expert

    def get_route_display(self) -> str:
        """Get short route display for timeline (BASE/EXPERT)."""
        return "EXPERT" if self.requires_intervention else "BASE"

    def get_latency_breakdown_str(self) -> str:
        """Get formatted latency string for UI."""
        if self.expert_latency_ms > 0:
            return f"routing={self.routing_latency_ms:.1f}ms + expert={self.expert_latency_ms:.1f}ms = {self.total_latency_ms:.1f}ms"
        return f"base={self.total_latency_ms:.1f}ms"


@dataclass
class GenerationConfig:
    """Configuration for token generation (fixed for reproducibility)."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    do_sample: bool = True
    seed: int = 42
    target_layer: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "seed": self.seed,
            "target_layer": self.target_layer,
        }


@dataclass
class RoutingPolicyConfig:
    """Configuration for routing policies."""

    confidence_threshold: float = 0.70
    domain_certainty_threshold: float = 0.35
    default_policy: str = "adaptive"
    available_policies: List[str] = field(default_factory=lambda: ["adaptive", "always_base", "always_expert", "random_expert"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "domain_certainty_threshold": self.domain_certainty_threshold,
            "default_policy": self.default_policy,
            "available_policies": self.available_policies,
        }


def create_prompt_events(
    prompt: str,
    tokenizer,
    layer_idx: int = -1,
) -> List[InferenceEvent]:
    """Create InferenceEvent objects for prompt tokens (for display purposes).

    These represent the input prompt tokens with default/placeholder values
    since no ARES routing occurs for prompt tokens.
    """
    from ares.backbone.qwen_loader import get_model_input_device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_tokens = input_ids[0].tolist()
    events = []

    for i, token_id in enumerate(prompt_tokens):
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        event = InferenceEvent(
            token=decoded,
            token_id=token_id,
            token_index=i,
            sequence_position=i,
            is_prompt_token=True,
            global_reliability=0.0,
            local_reliability=0.0,
            combined_reliability=0.0,
            failure_risk=0.0,
            is_reliable=True,
            predicted_domain="general",
            domain_probabilities={"general": 1.0, "math": 0.0, "code": 0.0, "science": 0.0},
            domain_confidence=1.0,
            is_domain_certain=True,
            policy="prompt",
            selected_expert=None,
            requires_intervention=False,
            routing_reason="Prompt token (no routing)",
            routing_latency_ms=0.0,
            expert_latency_ms=0.0,
            total_latency_ms=0.0,
            base_logits_used=True,
            expert_logits_used=False,
            expert_activation_count=0,
            timestamp=time.time(),
            layer_idx=layer_idx,
        )
        events.append(event)

    return events
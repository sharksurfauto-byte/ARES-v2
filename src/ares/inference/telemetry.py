"""Telemetry collection and aggregation for ARES V2 inference prototype.

Collects InferenceEvent data from generate_stream() and produces:
- Per-run metrics snapshots (tokens, expert %, reliability, latency)
- JSONL/CSV export for Phase 6 evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import json
import csv
import numpy as np


@dataclass
class TelemetrySnapshot:
    """Aggregated metrics snapshot from InferenceEvent stream."""
    tokens_generated: int
    prompt_tokens: int
    expert_activations: int
    base_activations: int
    expert_compute_percentage: float
    expert_activation_reduction_vs_always_on: float
    average_reliability: float
    average_routing_latency_ms: float
    average_expert_latency_ms: float
    average_total_latency_ms: float
    tokens_per_second: float
    domain_distribution: Dict[str, int]
    expert_distribution: Dict[str, int]
    intervention_timeline: List[Dict]
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "expert_activations": self.expert_activations,
            "base_activations": self.base_activations,
            "expert_compute_percentage": self.expert_compute_percentage,
            "expert_activation_reduction_vs_always_on": self.expert_activation_reduction_vs_always_on,
            "average_reliability": self.average_reliability,
            "average_routing_latency_ms": self.average_routing_latency_ms,
            "average_expert_latency_ms": self.average_expert_latency_ms,
            "average_total_latency_ms": self.average_total_latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "domain_distribution": self.domain_distribution,
            "expert_distribution": self.expert_distribution,
            "intervention_timeline": self.intervention_timeline,
            "seed": self.seed,
        }


@dataclass
class TelemetryCollector:
    """Collects InferenceEvent objects from generate_stream and aggregates metrics."""

    events: List[InferenceEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    seed: int = 42

    def add_event(self, event: InferenceEvent) -> None:
        """Add a single InferenceEvent to the collector."""
        self.events.append(event)

    def add_events(self, events: List[InferenceEvent]) -> None:
        """Add multiple events at once."""
        self.events.extend(events)

    def get_snapshot(self) -> TelemetrySnapshot:
        """Aggregate all collected events into a TelemetrySnapshot."""
        if not self.events:
            return TelemetrySnapshot(
                tokens_generated=0,
                prompt_tokens=0,
                expert_activations=0,
                base_activations=0,
                expert_compute_percentage=0.0,
                expert_activation_reduction_vs_always_on=0.0,
                average_reliability=0.0,
                average_routing_latency_ms=0.0,
                average_expert_latency_ms=0.0,
                average_total_latency_ms=0.0,
                tokens_per_second=0.0,
                domain_distribution={},
                expert_distribution={},
                intervention_timeline=[],
                seed=self.seed,
            )

        n_events = len(self.events)

        # Token counts
        tokens_generated = sum(1 for e in self.events if not e.is_prompt_token)
        prompt_tokens = sum(1 for e in self.events if e.is_prompt_token)
        prompt_tokens = min(prompt_tokens, n_events)

        # Expert activations
        expert_activations = sum(1 for e in self.events if e.requires_intervention)
        base_activations = n_events - expert_activations - sum(1 for e in self.events if e.is_prompt_token)

        # Compute percentages
        expert_compute_percentage = (expert_activations / n_events * 100) if n_events > 0 else 0.0

        # Always-on baseline = 100% (reference)
        expert_activation_reduction_vs_always_on = (
            1.0 - (expert_activations / n_events)) if n_events > 0 else 1.0

        # Reliability metrics
        reliabilities = [e.combined_reliability for e in self.events if not e.is_prompt_token]
        avg_reliability = float(np.mean(reliabilities)) if reliabilities else 0.0

        # Latency metrics (only non-prompt events)
        routing_latencies = [e.routing_latency_ms for e in self.events if not e.is_prompt_token]
        expert_latencies = [e.expert_latency_ms for e in self.events if not e.is_prompt_token and e.expert_latency_ms > 0]
        total_latencies = [e.total_latency_ms for e in self.events if not e.is_prompt_token]

        avg_routing_latency = float(np.mean(routing_latencies)) if routing_latencies else 0.0
        avg_expert_latency = float(np.mean(expert_latencies)) if expert_latencies else 0.0
        avg_total_latency = float(np.mean(total_latencies)) if total_latencies else 0.0

        # Timing
        elapsed = time.time() - self.start_time
        tokens_per_sec = (tokens_generated / elapsed) if elapsed > 0 else 0.0

        # Domain distribution
        domain_dist: Dict[str, int] = {}
        for e in self.events:
            if not e.is_prompt_token:
                d = e.predicted_domain
                domain_dist[d] = domain_dist.get(d, 0) + 1

        # Expert distribution
        expert_dist: Dict[str, int] = {}
        for e in self.events:
            if e.requires_intervention:
                expert_name = e.selected_expert or "UNKNOWN"
                expert_dist[expert_name] = expert_dist.get(expert_name, 0) + 1

        # Intervention timeline (per token)
        intervention_timeline = []
        for e in self.events:
            if e.requires_intervention:
                intervention_timeline.append({
                    "token": e.token,
                    "position": e.sequence_position,
                    "expert": e.selected_expert,
                    "domain": e.predicted_domain,
                    "reliability": e.combined_reliability,
                    "routing_reason": e.routing_reason,
                })

        return TelemetrySnapshot(
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            expert_activations=expert_activations,
            base_activations=base_activations,
            expert_compute_percentage=round(expert_compute_percentage, 2),
            expert_activation_reduction_vs_always_on=round(expert_activation_reduction_vs_always_on, 2),
            average_reliability=round(avg_reliability, 4),
            average_routing_latency_ms=round(avg_routing_latency, 2),
            average_expert_latency_ms=round(avg_expert_latency, 2),
            average_total_latency_ms=round(avg_total_latency, 2),
            tokens_per_second=round(tokens_per_sec, 2),
            domain_distribution=domain_dist,
            expert_distribution=expert_dist,
            intervention_timeline=intervention_timeline,
            seed=self.seed,
        )

    def export_jsonl(self, path: Path) -> None:
        """Export events as JSONL (one JSON per line)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(event.to_dict().__str__().replace("'", '"') + "\n")

    def export_csv(self, path: Path) -> None:
        """Export events as CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "token", "token_id", "token_index", "sequence_position",
                "is_prompt_token", "global_reliability", "local_reliability",
                "combined_reliability", "failure_risk", "is_reliable",
                "predicted_domain", "domain_confidence", "is_domain_certain",
                "policy", "selected_expert", "requires_intervention",
                "routing_reason", "routing_latency_ms", "expert_latency_ms",
                "total_latency_ms", "base_logits_used", "expert_logits_used",
                "expert_activation_count", "timestamp", "layer_idx"
            ])
            for event in self.events:
                row = [
                    event.token, event.token_id, event.token_index, event.sequence_position,
                    event.is_prompt_token, event.global_reliability, event.local_reliability,
                    event.combined_reliability, event.failure_risk, event.is_reliable,
                    event.predicted_domain, event.domain_confidence, event.is_domain_certain,
                    event.policy, event.selected_expert, event.requires_intervention,
                    event.routing_reason, event.routing_latency_ms, event.expert_latency_ms,
                    event.total_latency_ms, event.base_logits_used, event.expert_logits_used,
                    event.expert_activation_count, event.timestamp, event.layer_idx
                ]
                writer.writerow(row)
"""ARES V2 Inference Module - Phase 6.

Dedicated inference abstraction that integrates all existing Phases 1-5:
Qwen backbone, GRM/LRM reliability models, ExpertManager LoRA adapters,
AdaptiveExpertRouter, and streaming generation with telemetry.

Recommended usage:

    from ares.inference import ARESInferenceEngine, generate_stream
    engine = ARESInferenceEngine()
    for event in generate_stream(engine, "Prompt text", max_new_tokens=64):
        # event contains full ARES pipeline trace per token
        print(f"{event.token} | R={event.combined_reliability:.2f} | "
              f"Expert={event.selected_expert} | latency={event.total_latency_ms:.1f}ms")
"""

from ares.inference.events import (
    InferenceEvent,
    GenerationConfig,
    RoutingPolicyConfig,
    create_prompt_events,
)
from ares.inference.engine import ARESInferenceEngine, CheckpointPaths
from ares.inference.generation import generate_stream, generate_batch
from ares.inference.telemetry import TelemetryCollector, TelemetrySnapshot

__all__ = [
    "ARESInferenceEngine",
    "InferenceEvent",
    "GenerationConfig",
    "RoutingPolicyConfig",
    "create_prompt_events",
    "generate_stream",
    "generate_batch",
    "TelemetryCollector",
    "TelemetrySnapshot",
]
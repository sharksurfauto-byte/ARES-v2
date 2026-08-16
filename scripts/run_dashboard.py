"""Streamlit dashboard for ARES v2 inference prototype.

Displays live ARES pipeline: token stream, reliability gauges, domain probabilities,
routing timeline, and cumulative statistics. Uses ARESInferenceEngine and
TelemetryCollector from the Phase 1-5 integrated codebase.
"""

import streamlit as st
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path

from ares.inference import ARESInferenceEngine, generate_stream, TelemetryCollector
from ares.inference.events import InferenceEvent

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="ARES v2 - Adaptive Reliability & Expert System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Initialization ─────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = None
if "events" not in st.session_state:
    st.session_state.events = []
if "telemetry" not in st.session_state:
    st.session_state.telemetry = TelemetryCollector()
if "generated" not in st.session_state:
    st.session_state.generated = False
if "policy" not in st.session_state:
    st.session_state.policy = "adaptive"
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# ─── Configuration ────────────────────────────────────────────────────
# Checkpoint paths - production mode requires all checkpoints
DEVICE_MAP = "auto"
USE_CACHE = False  # Critical: disable KV cache for dynamic expert switching

POLICY_OPTIONS = ["adaptive", "always_base", "always_expert", "random_expert"]

# ─── Helper Functions ─────────────────────────────────────────────────


def init_engine():
    """Initialize ARESInferenceEngine with production checkpoints."""
    from ares.utils.checkpoint import load_checkpoint_with_validation

    try:
        engine = ARESInferenceEngine(
            model_name="Qwen/Qwen2.5-7B",
            torch_dtype="bfloat16",
            device_map=DEVICE_MAP,
            use_cache=USE_CACHE,
            production_mode=True,  # Will error if checkpoints missing
            confidence_threshold=0.7,
            domain_certainty_threshold=0.35,
            target_layer=-1,
            seed=42,
        )
        st.session_state.engine = engine
        st.session_state.telemetry = TelemetryCollector()
        return True
    except FileNotFoundError as e:
        st.error(f"Missing required checkpoints for production mode:\n{e}")
        # Fall back to debug mode (untrained probes) - NOT recommended for demo
        try:
            engine = ARESInferenceEngine(
                model_name="Qwen/Qwen2.5-7B",
                torch_dtype="bfloat16",
                device_map=DEVICE_MAP,
                use_cache=USE_CACHE,
                production_mode=False,
                confidence_threshold=0.7,
                domain_certainty_threshold=0.35,
                target_layer=-1,
                seed=42,
            )
            st.session_state.engine = engine
            st.session_state.telemetry = TelemetryCollector()
            st.sidebar.warning("Running in DEBUG mode (untrained probes).")
            return True
        except Exception as e2:
            st.error(f"Failed to initialize engine even in debug mode: {e2}")
            return False


def run_generation(engine, prompt, max_new_tokens, temperature, do_sample, policy):
    """Run generation and collect events."""
    from ares.inference.generation import generate_stream

    events = list(generate_stream(
        engine=engine,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        policy=policy,
        layer_idx=-1,
    ))

    st.session_state.events = events
    st.session_state.generated = True

    # Collect telemetry
    from ares.inference.telemetry import TelemetryCollector
    # We need to re-create or update the telemetry collector

    # Update session state with final metrics
    from ares.inference.telemetry import TelemetryCollector
    tc = TelemetryCollector(events)
    snapshot = tc.get_snapshot()
    st.session_state.telemetry = tc
    return snapshot


def render_gauge(label: str, value: float, max_val: float = 1.0, title_suffix: str = "") -> None:
    """Render a simple gauge using st.progress."""
    pct = min(max_val, max(0, value)) / max_val if max_val > 0 else 0
    st.write(f"**{label}**: {value:.3f}")
    st.progress(pct, f"{label}: {value:.3f} / {max_val:.3f}")


def render_domain_bars(domains: Dict[str, int], total: int) -> None:
    """Render domain probability bars."""
    st.write("**Domain Probabilities**")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)  # Rough visual bar
        st.write(f"{domain:8s}: {pct:5.1f}% {bar}")


def render_routing_timeline(events: List[InferenceEvent]) -> None:
    """Render the routing timeline table."""
    st.write("**Routing Timeline**")

    # Table header
    cols = ["Token", "R(x)", "Domain", "Route"]
    cols_display = st.columns(len(cols))
    for i, col in enumerate(cols):
        cols_display[i].write(f"**{col}**")

    # Render each token
    for e in events:
        if e.is_prompt_token:
            continue
        row_cols = st.columns(len(cols))
        route_str = "EXPERT" if e.requires_intervention else "BASE"

        # Highlight interventions
        if e.requires_intervention:
            route_str = f"🔴 {route_str}"

        row_cols[0].write(e.token[:8] + ("..." if len(e.token) > 8 else ""))
        row_cols[1].write(f"{e.combined_reliability:.2f}")
        row_cols[2].write(e.predicted_domain)
        row_cols[3].write(route_str)


# ─── Main Application ─────────────────────────────────────────────────
def main():
    st.title("🧠 ARES v2 - Adaptive Reliability & Expert System")
    st.caption("Real-time inference dashboard - Qwen 7B + GRM/LRM + Expert Routing")

    # ─── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Policy selection
        st.session_state.policy = st.radio(
            "Routing Policy",
            POLICY_OPTIONS,
            index=0,
            help="Adaptive: Qwen→GRM/LRM→Router→BASE/EXPERT; Always-Expert: 100% expert; Base-Only: 0%"
        )

        st.divider()

        # Generation params
        max_new = st.slider("Max new tokens", 16, 256, 128)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        do_sample = st.checkbox("Do sample", value=True)

        st.divider()

        # Initialize / Generate buttons
        if st.button("🔧 Initialize Engine", type="primary"):
            with st.spinner("Initializing ARES engine..."):
                success = init_engine()
                if success:
                    st.success("Engine ready!")

        st.divider()

        if st.button("▶️ Generate", type="secondary") and st.session_state.engine:
            if not st.session_state.prompt.strip():
                st.warning("Please enter a prompt first.")
            else:
                with st.spinner("Generating with ARES pipeline..."):
                    snapshot = run_generation(
                        engine=st.session_state.engine,
                        prompt=st.session_state.prompt,
                        max_new_tokens=max_new,
                        temperature=temperature,
                        do_sample=do_sample,
                        policy=st.session_state.policy,
                    )
                # Render summary stats
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tokens generated", snapshot.tokens_generated)
                c2.metric("Expert activations%", f"{snapshot.expert_compute_percentage}%")
                c3.metric("Reduction vs Always-On", f"{snapshot.expert_activation_reduction_vs_always_on*100:.1f}%")
                c4.metric("Avg reliability", f"{snapshot.average_reliability:.2f}")

                # Render domain bars
                render_domain_bars(snapshot.domain_distribution, snapshot.tokens_generated)

                # Render reliability gauges
                render_gauge("Global R", snapshot.average_reliability, 1.0)
                render_gauge("Routing latency", snapshot.average_routing_latency_ms, 50.0)
                render_gauge("Expert latency", snapshot.average_expert_latency_ms, 50.0)

                # Render routing timeline
                render_routing_timeline(st.session_state.events)

        st.divider()

        # Prompt input
        user_prompt = st.text_area(
            "💬 Enter prompt:",
            value=st.session_state.prompt,
            height=100,
            placeholder="e.g., Solve: 2x + 5 = 15"
        )
        st.session_state.prompt = user_prompt

    # ─── Main Area - Token Stream ─────────────────────────────────────
    if st.session_state.generated and st.session_state.events:
        st.divider()
        st.subheader("📜 Token Stream")

        # Token stream with typewriter effect
        placeholder = st.container()
        with placeholder:
            for e in st.session_state.events:
                if e.is_prompt_token:
                    continue
                # Typewriter-style display
                st.write(f"**{e.token}**", end=" ")
                # Show routing info in a small caption
                route_marker = "🔴" if e.requires_intervention else ""
                st.caption(f"R={e.combined_reliability:.2f} | {e.get_route_display()} | {e.get_latency_breakdown_str()}")

    elif not st.session_state.generated:
        # Welcome message when nothing generated yet
        st.info("👈 Initialize the engine and enter a prompt to get started.")
        st.markdown(
            """
            **What this dashboard shows:**
            - **Token Stream**: Live generation with typewriter effect
            - **Routing Timeline**: Per-token BASE/EXPERT decisions
            - **Domain Probabilities**: GRM domain distribution
            - **Reliability Gauges**: Global R, Local R, Combined R(x)
            - **Expert Activations**: % of tokens using expert adapters
            - **Routing Policies**: Adaptive, Base-Only, Always-Expert, Random
            """
        )


if __name__ == "__main__":
    main()
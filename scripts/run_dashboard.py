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
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
if "telemetry" not in st.session_state:
    st.session_state.telemetry = TelemetryCollector()
if "generated" not in st.session_state:
    st.session_state.generated = False
if "policy" not in st.session_state:
    st.session_state.policy = "adaptive"
if "prompt" not in st.session_state:
    st.session_state.prompt = "Solve step-by-step: If 3x + 7 = 22, what is x?"

# ─── Configuration ────────────────────────────────────────────────────
DEVICE_MAP = "auto"
USE_CACHE = False  # Critical: disable KV cache for dynamic expert switching

POLICY_OPTIONS = ["adaptive", "always_base", "always_expert", "random_expert"]

# ─── Helper Functions ─────────────────────────────────────────────────


def init_engine():
    """Initialize ARESInferenceEngine with production checkpoints."""
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
            st.error(f"Failed to initialize engine: {e2}")
            return False


def run_generation(engine, prompt, max_new_tokens, temperature, do_sample, policy):
    """Run generation and collect events."""
    from ares.inference.generation import generate_stream
    from ares.inference.telemetry import TelemetryCollector

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

    tc = TelemetryCollector(events)
    snapshot = tc.get_snapshot()
    st.session_state.telemetry = tc
    st.session_state.snapshot = snapshot
    return snapshot


def render_gauge(label: str, value: float, max_val: float = 1.0) -> None:
    """Render a simple gauge using st.progress."""
    pct = min(1.0, max(0.0, float(value / max_val))) if max_val > 0 else 0.0
    st.write(f"**{label}**: {value:.3f}")
    st.progress(pct, text=f"{label}: {value:.3f} / {max_val:.3f}")


def render_domain_bars(domains: Dict[str, int], total: int) -> None:
    """Render domain probability bars."""
    st.write("**GRM Domain Probability Distribution**")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        st.write(f"`{domain:8s}`: {pct:5.1f}% {bar}")


def render_routing_timeline(events: List[InferenceEvent]) -> None:
    """Render the routing timeline table."""
    st.write("**Detailed Per-Token Routing Log**")
    cols = ["Token", "R(x)", "Domain", "Route", "Latency"]
    cols_display = st.columns(len(cols))
    for i, col in enumerate(cols):
        cols_display[i].write(f"**{col}**")

    for e in events:
        if e.is_prompt_token:
            continue
        row_cols = st.columns(len(cols))
        route_str = f"🔴 EXPERT ({e.get_expert_display_name()})" if e.requires_intervention else "🟢 BASE"

        row_cols[0].write(f"`{e.token}`")
        row_cols[1].write(f"{e.combined_reliability:.3f}")
        row_cols[2].write(e.predicted_domain)
        row_cols[3].write(route_str)
        row_cols[4].write(f"{e.total_latency_ms:.1f} ms")


# ─── Main Application ─────────────────────────────────────────────────
def main():
    st.title("🧠 ARES v2 - Adaptive Reliability & Expert System")
    st.caption("Real-time inference prototype — Qwen 7B + GRM/LRM Probes + Multi-Domain LoRA Experts")

    # Auto-initialize engine if not ready
    if st.session_state.engine is None:
        with st.spinner("Initializing ARES Engine & loading checkpoints..."):
            success = init_engine()
            if success:
                st.success("ARES Engine initialized cleanly!")

    # ─── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ System Control & Policy")

        if st.session_state.engine is not None:
            st.success("🟢 Engine Active & Loaded")
        else:
            st.error("🔴 Engine Not Initialized")
            if st.button("🔧 Force Re-Initialize"):
                init_engine()

        st.divider()

        # Policy selection
        st.session_state.policy = st.radio(
            "Routing Policy",
            POLICY_OPTIONS,
            index=0,
            help="Adaptive: Dynamic GRM/LRM routing; Always-Expert: 100% expert; Always-Base: 0%"
        )

        st.divider()

        # Generation params
        max_new = st.slider("Max new tokens", 16, 256, 128)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
        do_sample = st.checkbox("Do sample", value=True)

    # ─── Main Area - Prompt & Generation ──────────────────────────────
    user_prompt = st.text_area(
        "💬 Enter Input Prompt:",
        value=st.session_state.prompt,
        height=100,
        placeholder="e.g., Solve: 2x + 5 = 15 or Write a python function for quicksort"
    )
    st.session_state.prompt = user_prompt

    if st.button("▶️ Generate Response with ARES Routing", type="primary", use_container_width=True):
        if not st.session_state.engine:
            st.error("Engine not ready. Please click Initialize in sidebar.")
        elif not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Generating tokens with ARES two-pass routing..."):
                run_generation(
                    engine=st.session_state.engine,
                    prompt=user_prompt,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    do_sample=do_sample,
                    policy=st.session_state.policy,
                )

    # ─── Results Dashboard (Main Area) ────────────────────────────────
    if st.session_state.generated and st.session_state.events and st.session_state.snapshot:
        snapshot = st.session_state.snapshot

        st.divider()
        st.header("📊 ARES Execution Telemetry")

        # Key Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tokens Generated", snapshot.tokens_generated)
        c2.metric("Expert Activations", f"{snapshot.expert_compute_percentage:.1f}%", f"{snapshot.expert_activations} tokens")
        c3.metric("Compute Savings", f"{snapshot.expert_activation_reduction_vs_always_on*100:.1f}%", "vs Always-On")
        c4.metric("Mean Reliability R(x)", f"{snapshot.average_reliability:.3f}")

        st.divider()
        st.header("📜 Model Output Response")
        full_text = "".join(e.token for e in st.session_state.events if not e.is_prompt_token)
        st.write(full_text)

        st.divider()
        st.header("🔍 Per-Token ARES Pipeline Annotations")
        for e in st.session_state.events:
            if e.is_prompt_token:
                continue
            route_marker = f"🔴 EXPERT ({e.get_expert_display_name()})" if e.requires_intervention else "🟢 BASE"
            st.markdown(
                f"**`{e.token}`** — `{route_marker}` | **R(x)**={e.combined_reliability:.3f} | "
                f"Domain={e.predicted_domain} | {e.get_latency_breakdown_str()}"
            )

        st.divider()
        st.header("📈 Reliability & Latency Breakdown")
        g1, g2, g3 = st.columns(3)
        with g1:
            render_gauge("Mean Reliability", snapshot.average_reliability, 1.0)
        with g2:
            render_gauge("Routing Latency", snapshot.average_routing_latency_ms, 50.0)
        with g3:
            render_gauge("Expert Latency", snapshot.average_expert_latency_ms, 50.0)

        render_domain_bars(snapshot.domain_distribution, snapshot.tokens_generated)

        st.divider()
        render_routing_timeline(st.session_state.events)

    elif not st.session_state.generated:
        st.info("👈 Enter a prompt above and click **Generate Response with ARES Routing** to view live telemetry.")


if __name__ == "__main__":
    main()
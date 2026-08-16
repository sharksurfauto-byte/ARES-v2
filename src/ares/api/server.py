"""FastAPI Server for ARES V2 Visualizer Backend.

Provides real-time Server-Sent Events (SSE) streaming of token-by-token
inference events (Qwen 7B -> GRM/LRM -> Router -> Base/Expert -> Next Token)
for the Vite/React Transformer Explainer-style visualizer frontend.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ares.inference.engine import ARESInferenceEngine
from ares.inference.generation import generate_stream
from ares.inference.telemetry import TelemetryCollector

# Global engine singleton instance
_GLOBAL_ENGINE: Optional[ARESInferenceEngine] = None


def set_global_engine(engine: ARESInferenceEngine) -> None:
    """Set global engine singleton instance."""
    global _GLOBAL_ENGINE
    _GLOBAL_ENGINE = engine


def get_global_engine() -> Optional[ARESInferenceEngine]:
    """Get global engine singleton instance."""
    return _GLOBAL_ENGINE


class GenerationRequest(BaseModel):
    """Pydantic model for generation request payload."""

    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: int = Field(default=128, ge=1, le=512, description="Max new tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    do_sample: bool = Field(default=True, description="Whether to sample or greedy decode")
    policy: str = Field(default="adaptive", description="Routing policy ('adaptive', 'always_base', 'always_expert', 'random_expert')")
    collect_attentions: bool = Field(default=False, description="Whether to collect true Qwen self-attention weights")
    attn_layer: int = Field(default=12, ge=0, le=27, description="Target layer index for self-attention (0 to 27)")
    attn_head: int = Field(default=7, ge=-1, le=27, description="Target head index (-1 for average across heads, 0 to 27)")


def create_app(engine: Optional[ARESInferenceEngine] = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        engine: Optional pre-initialized ARESInferenceEngine instance.

    Returns:
        Configured FastAPI application instance.
    """
    if engine is not None:
        set_global_engine(engine)

    app = FastAPI(
        title="ARES V2 Visualizer API",
        description="Real-time telemetry and adaptive expert routing API for ARES V2",
        version="2.0.0",
    )

    # Enable CORS for local Vite dev server and external frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        eng = get_global_engine()
        return {
            "status": "healthy" if eng is not None else "degraded",
            "engine_initialized": eng is not None,
            "device": str(eng.input_device) if eng is not None else "none",
            "model_name": eng.model_name if eng is not None else "none",
            "production_mode": eng.production_mode if eng is not None else False,
        }

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        """Get current inference router & engine config."""
        eng = get_global_engine()
        if eng is None:
            raise HTTPException(status_code=503, detail="ARES Engine not initialized")

        return {
            "model_name": eng.model_name,
            "confidence_threshold": eng.confidence_threshold,
            "domain_certainty_threshold": eng.domain_certainty_threshold,
            "target_layer": eng.target_layer,
            "active_expert": eng.active_expert,
            "available_experts": list(eng.expert_manager.EXPERT_MAP.values()),
            "routing_policies": ["adaptive", "always_base", "always_expert", "random_expert"],
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
        }

    @app.post("/api/generate_stream")
    async def generate_stream_endpoint(request: GenerationRequest) -> StreamingResponse:
        """Stream token-by-token generation events using Server-Sent Events (SSE)."""
        eng = get_global_engine()
        if eng is None:
            raise HTTPException(status_code=503, detail="ARES Engine not initialized")

        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        async def sse_event_generator():
            collector = TelemetryCollector()
            for event in generate_stream(
                engine=eng,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
                policy=request.policy,
                collect_attentions=request.collect_attentions,
                attn_layer=request.attn_layer,
                attn_head=request.attn_head,
            ):
                collector.add_event(event)
                payload = event.to_dict()
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.005)  # Yield control to event loop

            # Send final snapshot event
            snapshot = collector.get_snapshot()
            final_payload = {"type": "snapshot", "data": snapshot.to_dict()}
            yield f"data: {json.dumps(final_payload)}\n\n"

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Mount static frontend directory if dist exists
    dist_dir = Path(__file__).parent.parent.parent.parent / "visualizer" / "dist"
    if dist_dir.exists() and dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    return app

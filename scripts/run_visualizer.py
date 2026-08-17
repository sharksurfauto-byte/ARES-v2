"""CLI Launcher for ARES V2 Live Interactive Visualizer Server.

Initializes ARESInferenceEngine with trained GRM/LRM probes and LoRA experts,
then launches the FastAPI Uvicorn backend server on port 8501 (hosting both
the SSE API endpoints and the Vite/React interactive visualizer frontend).
"""

import argparse
import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from ares.api.server import create_app, set_global_engine
from ares.inference.engine import ARESInferenceEngine
from ares.utils.environment import resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="ARES V2 Live Visualizer Server Launcher")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Base model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8501, help="Port number")
    parser.add_argument("--production", action="store_true", default=True, help="Enforce production checkpoint validation")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (untrained probes fallback)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Confidence threshold R_th")
    parser.add_argument("--domain_certainty_threshold", type=float, default=0.35, help="Domain certainty threshold D_th")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit NF4 quantization (VRAM optimization)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("🧠 ARES V2 — Live Transformer Explainer Visualizer Server")
    print(f"Host: {args.host}:{args.port}")
    print("=" * 60)

    production_mode = not args.debug

    # Initialize Engine
    print("Initializing ARES Inference Engine...")
    try:
        engine = ARESInferenceEngine(
            model_name=args.model,
            torch_dtype="bfloat16",
            device_map="auto",
            load_in_4bit=args.load_in_4bit,
            production_mode=production_mode,
            confidence_threshold=args.confidence_threshold,
            domain_certainty_threshold=args.domain_certainty_threshold,
            target_layer=-1,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Warning: Checkpoints missing for production mode ({e}). Falling back to debug mode...")
        engine = ARESInferenceEngine(
            model_name=args.model,
            torch_dtype="bfloat16",
            device_map="auto",
            load_in_4bit=args.load_in_4bit,
            production_mode=False,
            confidence_threshold=args.confidence_threshold,
            domain_certainty_threshold=args.domain_certainty_threshold,
            target_layer=-1,
            seed=args.seed,
        )

    # Set global engine singleton
    set_global_engine(engine)

    # Create FastAPI application
    app = create_app(engine)

    print(f"\nStarting Uvicorn Server on http://{args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

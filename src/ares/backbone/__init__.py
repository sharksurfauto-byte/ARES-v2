"""Backbone module exports for ARES V2."""

from ares.backbone.hidden_extractor import (
    ActivationHookManager,
    compute_logit_entropy,
    compute_prediction_margin,
    pool_hidden_states,
)
from ares.backbone.qwen_loader import (
    generate_qwen_text,
    get_model_input_device,
    get_torch_dtype,
    load_qwen_model,
    load_qwen_tokenizer,
    resolve_device,
    run_qwen_forward,
    verify_qwen_backbone,
)

__all__ = [
    "get_torch_dtype",
    "resolve_device",
    "get_model_input_device",
    "load_qwen_tokenizer",
    "load_qwen_model",
    "verify_qwen_backbone",
    "run_qwen_forward",
    "generate_qwen_text",
    "pool_hidden_states",
    "compute_logit_entropy",
    "compute_prediction_margin",
    "ActivationHookManager",
]

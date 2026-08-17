"""Qwen Backbone Model & Tokenizer Loader for ARES V2.

Provides authoritative loading, sharded multi-GPU device_map support (Accelerate),
strict parameter/config verification, forward pass execution, and deterministic text generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ares.utils.config import ModelConfig
from ares.utils.environment import get_rank, is_ddp_initialized


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map string representation ('bfloat16', 'float16', 'float32') to torch.dtype.

    Note: Qwen 2.5 models suffer from numerical overflow/NaNs when executed in float16 on GPU.
    If 'float16' is requested on CUDA devices supporting bfloat16, bfloat16 is automatically used.
    """
    dtype_str = dtype_str.lower()
    if dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif dtype_str in ("float16", "fp16"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif dtype_str in ("float32", "fp32"):
        return torch.float32
    else:
        raise ValueError(f"Unsupported torch_dtype string: {dtype_str}")


def resolve_device() -> torch.device:
    """Determine the appropriate PyTorch device for model execution considering DDP rank & CUDA capability."""
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 7:
                if is_ddp_initialized():
                    local_rank = get_rank()
                    return torch.device(f"cuda:{local_rank}")
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def get_model_input_device(model: PreTrainedModel) -> torch.device:
    """Retrieve the exact device of the model's input embedding layer.

    Essential for sharded multi-GPU models (e.g. device_map='auto') where
    parameters are distributed across multiple GPUs.
    """
    return model.get_input_embeddings().weight.device


def load_qwen_tokenizer(
    model_config: Optional[ModelConfig] = None,
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    """Load Qwen tokenizer with official HF interface and verified special tokens."""
    config = model_config or ModelConfig(name_or_path="Qwen/Qwen2.5-7B")

    tokenizer = AutoTokenizer.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        padding_side=padding_side,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    return tokenizer


def load_qwen_model(
    model_config: Optional[ModelConfig] = None,
    device_map: Optional[Union[str, Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
) -> PreTrainedModel:
    """Load pretrained Qwen backbone model using Hugging Face AutoModelForCausalLM.

    Supports automatic multi-GPU model sharding via Accelerate (`device_map="auto"`).
    When `device_map` is specified or auto-detected for multi-GPU systems, explicit `.to()` calls
    are bypassed to allow layer sharding across available VRAM (e.g. dual T4s on Kaggle).

    Args:
        model_config: ModelConfig instance (defaults to Qwen/Qwen2.5-7B).
        device_map: Optional device placement strategy ('auto', 'balanced', or custom dict).
                    If None and multiple GPUs are present, defaults to 'auto'.
        device: Specific PyTorch device fallback when device_map is not used.

    Returns:
        PreTrainedModel instance loaded with strict parameters.
    """
    config = model_config or ModelConfig(name_or_path="Qwen/Qwen2.5-7B")
    torch_dtype = get_torch_dtype(config.torch_dtype)

    # Empty PyTorch CUDA cache before loading to reclaim VRAM from previous runs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determine sharding strategy
    resolved_device_map = device_map
    if resolved_device_map is None and config.device_map:
        resolved_device_map = config.device_map

    # Auto-select device_map="auto" if multiple GPUs are available and DDP is not active
    if resolved_device_map is None and torch.cuda.is_available() and torch.cuda.device_count() > 1 and not is_ddp_initialized():
        resolved_device_map = "auto"

    kwargs = {
        "revision": config.revision,
        "dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
        "use_cache": config.use_cache,
        "output_hidden_states": True,  # Mandatory for ARES representation collection
        "attn_implementation": getattr(config, "attn_implementation", "eager"),
    }

    if getattr(config, "load_in_4bit", False):
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            if resolved_device_map is None:
                kwargs["device_map"] = "auto"
        except ImportError:
            raise ImportError("bitsandbytes package is required for load_in_4bit=True. Run `pip install bitsandbytes`.")
    elif resolved_device_map is not None:
        kwargs["device_map"] = resolved_device_map

    model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path,
        **kwargs,
    )

    # Only place model explicitly if device_map was NOT specified
    if resolved_device_map is None:
        target_device = device or resolve_device()
        model.to(target_device)

    model.eval()  # Backbone is frozen in Phase 1 & 2
    return model


def verify_qwen_backbone(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    """Verify structural integrity and inspect parameter distribution across devices.

    Args:
        model: Loaded Qwen model.
        tokenizer: Loaded Qwen tokenizer.

    Returns:
        Dictionary containing architecture specs and device distribution metadata.
    """
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    vocab_size = getattr(config, "vocab_size", None)

    if hidden_size is None or num_hidden_layers is None or vocab_size is None:
        raise RuntimeError("Model config missing required architecture parameters!")

    embed_weights = model.get_input_embeddings().weight
    if embed_weights.shape[0] < len(tokenizer):
        raise RuntimeError(
            f"Embedding matrix size ({embed_weights.shape[0]}) smaller than tokenizer vocab size ({len(tokenizer)})!"
        )

    # Inspect per-device parameter distribution for sharded models
    device_distribution: Dict[str, int] = {}
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        dev_str = str(param.device)
        device_distribution[dev_str] = device_distribution.get(dev_str, 0) + numel

    # Extract device_map summary if model was sharded
    hf_device_map = getattr(model, "hf_device_map", None)

    return {
        "model_type": getattr(config, "model_type", "qwen"),
        "model_name_or_path": getattr(config, "_name_or_path", "Qwen/Qwen2.5-7B"),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "input_device": str(get_model_input_device(model)),
        "device_distribution": device_distribution,
        "hf_device_map_summary": list(set(hf_device_map.values())) if isinstance(hf_device_map, dict) else hf_device_map,
        "dtype": str(next(model.parameters()).dtype),
    }


def run_qwen_forward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str] | str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Execute a deterministic forward pass over text prompt(s) and extract logits & hidden states.

    Target device is automatically resolved to model.get_input_embeddings().weight.device
    to ensure input tensor alignment in multi-GPU sharded setups.
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # Resolve input placement to model input embedding device
    target_device = device or get_model_input_device(model)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(target_device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.logits, outputs.hidden_states


def generate_qwen_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    do_sample: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    """Execute deterministic text generation using Qwen backbone.

    Target device is automatically resolved to model.get_input_embeddings().weight.device
    to ensure input tensor alignment in multi-GPU sharded setups.
    """
    target_device = device or get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
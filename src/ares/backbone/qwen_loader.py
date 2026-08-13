from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ares.utils.config import ModelConfig
from ares.utils.environment import get_rank, is_ddp_initialized

def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map string representation ('bfloat16', 'float16', 'float32') to torch.dtype."""
    dtype_str = dtype_str.lower()
    if dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif dtype_str in ("float16", "fp16"):
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

def load_qwen_tokenizer(
        model_config:Optional[ModelConfig]=None,
        padding_side:str='right'
)->PreTrainedTokenizer:
    config=model_config or ModelConfig(name_or_path="Qwen/Qwen2.5-7B")

    tokenizer=AutoTokenizer.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        padding_side=padding_side
    )

    #ensure pad token is properly defined
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token":"<|endoftext|>"})
    return tokenizer

def load_qwen_model(
    model_config: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None,
)->PreTrainedModel:
    config = model_config or ModelConfig(name_or_path="Qwen/Qwen2.5-7B")
    target_device = device or resolve_device()
    torch_dtype = get_torch_dtype(config.torch_dtype)

    model=AutoModelForCausalLM.from_pretrained(
        config.name_or_path,
        revision=config.revision,
        dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
        use_cache=config.use_cache,
        output_hidden_states=True,  # Mandatory for ARES representation collection
    )

    model.to(target_device)
    model.eval()
    return model

def verify_qwen_backbone(
        model:PreTrainedModel,
        tokenizer:PreTrainedTokenizer
)->Dict[str,Any]:
    config = model.config
    hidden_size = getattr(config, "hidden_size", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    vocab_size = getattr(config, "vocab_size", None)

    if hidden_size is None or num_hidden_layers is None or vocab_size is None:
        raise RuntimeError("Model config missing required architecture parameters!")

    # Verify embedding layer matches tokenizer vocab size
    embed_weights = model.get_input_embeddings().weight
    if embed_weights.shape[0] < len(tokenizer):
        raise RuntimeError(
            f"Embedding matrix size ({embed_weights.shape[0]}) smaller than tokenizer vocab size ({len(tokenizer)})!"
        )

    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_type": getattr(config, "model_type", "qwen"),
        "model_name_or_path": getattr(config, "_name_or_path", "Qwen/Qwen2.5-1.5B"),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "total_parameters": param_count,
        "trainable_parameters": trainable_param_count,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }


def run_qwen_forward(
        model:PreTrainedModel,
        tokenizer:PreTrainedTokenizer,
        prompts:List[str] | str,
        device: Optional[torch.device] = None,
)->Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if isinstance(prompts, str):
        prompts=[prompts]

    target_device=device or next(model.parameters()).device

    inputs=tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(target_device)

    with torch.no_grad():
        outputs=model(**inputs, output_hidden_states=True)

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
    target_device=device or next(model.parameters()).device
    inputs=tokenizer(prompt, return_tensors='pt').to(target_device)

    with torch.no_grad():
        output_ids=model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
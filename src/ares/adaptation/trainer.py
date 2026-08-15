"""LoRA Parameter-Efficient Foundation Adaptation Trainer for Phase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

# Safely handle preinstalled incompatible torchao versions on environments like Kaggle
try:
    import peft.import_utils
    if hasattr(peft.import_utils, "is_torchao_available"):
        _orig_torchao_check = peft.import_utils.is_torchao_available
        def _safe_torchao_check() -> bool:
            try:
                return _orig_torchao_check()
            except ImportError:
                return False
        peft.import_utils.is_torchao_available = _safe_torchao_check
except Exception:
    pass

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = PeftModel = TaskType = get_peft_model = None

from ares.adaptation.dataset_loader import MultiDomainTextDataset, create_multi_domain_dataloader
from ares.utils.checkpoint import save_checkpoint_with_metadata
from ares.backbone.qwen_loader import get_model_input_device
from ares.utils.environment import is_main_process


def setup_foundation_adapter(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> PeftModel:
    """Wrap pretrained backbone model with Hugging Face PEFT LoRA adapter modules.

    Args:
        model: PreTrainedModel backbone instance.
        r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        target_modules: List of linear projection names to adapt.

    Returns:
        PeftModel instance with trainable LoRA adapter layers.
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "Hugging Face 'peft' library is required for foundation adaptation. "
            "Please install via 'pip install peft'."
        )

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)

    if is_main_process():
        peft_model.print_trainable_parameters()

    return peft_model


def train_foundation_adapter(
    model: PeftModel,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1,
    max_steps: int = -1,
    gradient_accumulation_steps: int = 1,
    output_dir: str | Path = "checkpoints/foundation_adapter",
) -> Dict[str, Any]:
    """Execute training loop for LoRA foundation adaptation.

    Args:
        model: PeftModel instance.
        train_dataloader: PyTorch DataLoader for training.
        optimizer: PyTorch optimizer.
        num_epochs: Number of training epochs.
        max_steps: Maximum training steps (-1 for full epoch execution).
        gradient_accumulation_steps: Gradient accumulation steps multiplier.
        output_dir: Path to save trained adapter weights.

    Returns:
        Dictionary containing training statistics (loss history, steps completed).
    """
    model.train()
    total_steps = 0
    step_loss_history: List[float] = []

    for epoch in range(num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            # Resolve batch input placement to model embedding device
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Route to model device
            target_device = next(model.parameters()).device
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)
            labels = labels.to(target_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1
                step_loss_history.append(accumulated_loss)

                if is_main_process() and total_steps % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}] Step [{total_steps}] Loss: {accumulated_loss:.4f}")

                accumulated_loss = 0.0

            if max_steps > 0 and total_steps >= max_steps:
                break

        if max_steps > 0 and total_steps >= max_steps:
            break

    # Save trained adapter on main process
    if is_main_process():
        save_path = Path(output_dir)
        model.save_pretrained(save_path)
        print(f"LoRA Foundation Adapter successfully saved to: {save_path.resolve()}")

    return {
        "total_steps": total_steps,
        "final_loss": step_loss_history[-1] if step_loss_history else 0.0,
        "loss_history": step_loss_history,
    }

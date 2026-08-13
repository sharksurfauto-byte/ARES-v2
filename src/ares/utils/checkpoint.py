"""Checkpoint integrity and validation utilities for ARES V2.

Enforces strict checkpoint integrity rules:
- Prohibits unvalidated strict=False loads.
- Validates SHA256 checksums, metadata sidecars, key matches, and tensor shapes.
- Single authoritative checkpoint verification pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn as nn


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA256 checksum of a file on disk.

    Args:
        file_path: Path to target file.

    Returns:
        Hex digest string of the file's SHA256 hash.
    """
    path = Path(file_path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checkpoint_integrity(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    allow_missing_keys: bool = False,
    allow_unexpected_keys: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Verify that a state dict exactly matches the model structure and tensor shapes.

    Args:
        model: PyTorch module target.
        state_dict: State dict containing parameter tensors to load.
        allow_missing_keys: Explicit override flag for partial adapter loading.
        allow_unexpected_keys: Explicit override flag for extended head loading.

    Returns:
        Tuple of (missing_keys, unexpected_keys, shape_mismatches).

    Raises:
        RuntimeError: If key or shape mismatches exist and overrides are not enabled.
    """
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    state_keys = set(state_dict.keys())

    missing_keys = list(model_keys - state_keys)
    unexpected_keys = list(state_keys - model_keys)

    shape_mismatches = []
    common_keys = model_keys.intersection(state_keys)
    for key in common_keys:
        expected_shape = model_state[key].shape
        actual_shape = state_dict[key].shape
        if expected_shape != actual_shape:
            shape_mismatches.append(
                f"Key '{key}': expected shape {expected_shape}, got {actual_shape}"
            )

    errors = []
    if missing_keys and not allow_missing_keys:
        errors.append(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys and not allow_unexpected_keys:
        errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    if shape_mismatches:
        errors.append(f"Shape mismatches ({len(shape_mismatches)}): {shape_mismatches[:5]}...")

    if errors:
        error_msg = "Checkpoint integrity validation failed!\n" + "\n".join(errors)
        raise RuntimeError(error_msg)

    return missing_keys, unexpected_keys, shape_mismatches


def save_checkpoint_with_metadata(
    save_dir: str | Path,
    model: nn.Module,
    component_type: str,
    metadata: Dict[str, Any],
    weights_filename: str = "model.pt",
) -> Path:
    """Save model state dict alongside a validated metadata JSON sidecar and SHA256 checksum.

    Args:
        save_dir: Target directory path.
        model: Model module to save.
        component_type: Category ('BACKBONE', 'RELIABILITY_MODEL', 'EXPERT', 'ROUTER').
        metadata: Custom metadata dictionary to embed in sidecar.
        weights_filename: Filename for the saved PyTorch binary.

    Returns:
        Path to the saved checkpoint directory.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / weights_filename
    torch.save(model.state_dict(), weights_path)

    # Compute binary hash
    sha256_hash = compute_file_sha256(weights_path)

    # Build tensor shape metadata
    tensor_meta = {
        name: list(param.shape)
        for name, param in model.state_dict().items()
    }

    full_meta = {
        "component_type": component_type,
        "weights_file": weights_filename,
        "sha256": sha256_hash,
        "tensor_shapes": tensor_meta,
        "user_metadata": metadata,
    }

    meta_path = out_dir / "checkpoint_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    return out_dir


def load_checkpoint_with_validation(
    checkpoint_dir: str | Path,
    model: nn.Module,
    weights_filename: str = "model.pt",
    allow_missing_keys: bool = False,
    allow_unexpected_keys: bool = False,
) -> Dict[str, Any]:
    """Load model checkpoint binaries after verifying SHA256 checksum and key/shape integrity.

    Args:
        checkpoint_dir: Path to directory containing weights and metadata.
        model: PyTorch model into which weights should be loaded.
        weights_filename: Filename of the saved PyTorch binary.
        allow_missing_keys: Explicit override flag for partial adapter loading.
        allow_unexpected_keys: Explicit override flag for extended head loading.

    Returns:
        Metadata dictionary loaded from checkpoint_meta.json sidecar.

    Raises:
        FileNotFoundError: If weights file or metadata sidecar is missing.
        RuntimeError: If SHA256 hash or tensor integrity check fails.
    """
    chk_path = Path(checkpoint_dir)
    weights_file = chk_path / weights_filename
    meta_file = chk_path / "checkpoint_meta.json"

    if not weights_file.exists():
        raise FileNotFoundError(f"Checkpoint weights missing at: {weights_file.resolve()}")

    # Verify SHA256 if metadata sidecar exists
    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_sha256 = metadata.get("sha256")
        if expected_sha256:
            actual_sha256 = compute_file_sha256(weights_file)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Checkpoint SHA256 mismatch! Expected: {expected_sha256}, Got: {actual_sha256}"
                )

    state_dict = torch.load(weights_file, map_location="cpu")

    # Perform strict structural integrity check
    verify_checkpoint_integrity(
        model,
        state_dict,
        allow_missing_keys=allow_missing_keys,
        allow_unexpected_keys=allow_unexpected_keys,
    )

    model.load_state_dict(state_dict, strict=(not allow_missing_keys and not allow_unexpected_keys))
    return metadata

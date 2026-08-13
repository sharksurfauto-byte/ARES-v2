"""Environment, seed management, and Multi-GPU (DDP) detection utilities for ARES V2."""

import os
import random
from typing import Any, Dict
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: Random seed integer.
        deterministic: If True, configures PyTorch cuDNN backend for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed hardware information about available CPU and GPUs.

    Returns:
        Dictionary containing platform, CUDA availability, GPU count, device names, and VRAM.
    """
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpus = []

    if cuda_available:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "major": props.major,
                "minor": props.minor,
            })

    return {
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpus": gpus,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
    }


def is_ddp_initialized() -> bool:
    """Check whether PyTorch distributed process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get global rank of the current process in DDP mode, or 0 if single-process."""
    if is_ddp_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes in DDP mode, or 1 if single-process."""
    if is_ddp_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True if running on the primary process (Rank 0), False otherwise."""
    return get_rank() == 0


def setup_ddp(backend: str = "nccl") -> int:
    """Initialize DistributedDataParallel (DDP) environment for Multi-GPU training.

    Automatically resolves LOCAL_RANK, RANK, and WORLD_SIZE environment variables
    injected by torchrun or accelerate.

    Args:
        backend: PyTorch DDP backend ('nccl' for CUDA/Kaggle, 'gloo' for CPU/Windows).

    Returns:
        int: Local rank of the process on the local machine.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0  # Not running under torchrun / DDP multi-process manager

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Adjust backend if CUDA is unavailable
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not is_ddp_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )

    return local_rank


def cleanup_ddp() -> None:
    """Clean up and destroy PyTorch distributed process group if initialized."""
    if is_ddp_initialized():
        dist.destroy_process_group()

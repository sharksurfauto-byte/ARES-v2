"""GPU VRAM Memory Cleaning Utility for ARES V2."""

import gc
import torch

def clear_gpu_memory():
    print("Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"CUDA Memory Cleared! Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available.")

if __name__ == "__main__":
    clear_gpu_memory()

"""
Hardware acceleration utilities for WhisperX
Optimizations for GPU acceleration and mixed precision training
"""

import torch
import warnings
from typing import Optional, Union


def setup_mixed_precision(device: Union[str, torch.device]) -> tuple[bool, Optional[torch.dtype]]:
    """
    Setup mixed precision training if supported by hardware

    Args:
        device: Target device

    Returns:
        Tuple of (use_mixed_precision, dtype)
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        # Check for Tensor Core support (Turing+ architecture)
        try:
            major, minor = torch.cuda.get_device_capability(device.index if device.index is not None else 0)
            has_tensor_cores = major >= 7 or (major == 6 and minor >= 1)

            if has_tensor_cores:
                # Use bfloat16 if available (Ampere+), otherwise float16
                if major >= 8:  # Ampere and newer
                    return True, torch.bfloat16
                else:  # Volta, Turing
                    return True, torch.float16
        except Exception:
            pass

    return False, None


def optimize_torch_settings(device: Union[str, torch.device], threads: Optional[int] = None):
    """
    Optimize PyTorch settings for performance

    Args:
        device: Target device
        threads: Number of threads (auto-detect if None)
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Set optimal thread count
    if threads is None:
        import os
        cpu_count = os.cpu_count() or 1
        if device.type == "cuda":
            # Use fewer threads on GPU to avoid CPU bottleneck
            threads = min(4, cpu_count)
        else:
            # Use more threads on CPU
            threads = cpu_count

    torch.set_num_threads(threads)

    # Enable optimizations
    if hasattr(torch, "set_float32_matmul_precision"):
        # Use TensorFloat-32 (TF32) on Ampere+ GPUs
        torch.set_float32_matmul_precision("medium")

    # Enable JIT optimizations
    torch.jit.set_bailout_depth(20)

    # CUDA optimizations
    if device.type == "cuda" and torch.cuda.is_available():
        # Enable cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Enable memory optimization
        try:
            torch.cuda.empty_cache()
            # Use memory efficient attention if available
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass


def get_optimal_batch_size(device: Union[str, torch.device], model_size: str = "base") -> int:
    """
    Calculate optimal batch size based on available GPU memory

    Args:
        device: Target device
        model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)

    Returns:
        Recommended batch size
    """
    if isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda" or not torch.cuda.is_available():
        return 1

    try:
        # Get available GPU memory in GB
        total_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated(device)
        memory_gb = available_memory / (1024**3)

        # Rough memory requirements per model (in GB per sample)
        memory_per_sample = {
            "tiny": 0.5,
            "base": 0.7,
            "small": 1.0,
            "medium": 2.0,
            "large": 3.0,
            "large-v2": 3.0,
            "large-v3": 3.5,
        }

        base_memory = memory_per_sample.get(model_size, 2.0)

        # Calculate batch size with safety margin
        max_batch_size = max(1, int((memory_gb * 0.8) / base_memory))

        # Cap at reasonable limits
        return min(max_batch_size, 32)

    except Exception:
        return 4  # Safe default


def enable_compilation_optimizations(model: torch.nn.Module) -> torch.nn.Module:
    """
    Enable PyTorch 2.0 compilation optimizations if available

    Args:
        model: PyTorch model to optimize

    Returns:
        Optimized model
    """
    try:
        if hasattr(torch, "compile") and torch.cuda.is_available():
            # Use compile for faster inference
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Enabled PyTorch 2.0 compilation optimizations")
    except Exception as e:
        warnings.warn(f"Could not enable compilation optimizations: {e}")

    return model


class MemoryOptimizer:
    """Context manager for memory optimization during inference"""

    def __init__(self, device: Union[str, torch.device]):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.original_settings = {}

    def __enter__(self):
        if self.device.type == "cuda":
            # Store original settings
            self.original_settings["empty_cache"] = True

            # Clear cache
            torch.cuda.empty_cache()

            # Enable memory efficient operations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == "cuda":
            # Clean up
            torch.cuda.empty_cache()


def warmup_model(model, device: Union[str, torch.device], input_shape: tuple = (1, 80, 3000)):
    """
    Warmup model to optimize first inference

    Args:
        model: Model to warmup
        device: Target device
        input_shape: Input tensor shape for warmup
    """
    if isinstance(device, str):
        device = torch.device(device)

    try:
        with torch.no_grad():
            dummy_input = torch.randn(input_shape, device=device)

            # Run a few warmup iterations
            for _ in range(3):
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

        print("✓ Model warmup completed")

    except Exception as e:
        warnings.warn(f"Model warmup failed: {e}")
"""Util to check if gpu is available"""
import torch


def is_gpu_available():
    """
    Check if a GPU is available for PyTorch operations.

    This function checks for the availability of CUDA-compatible GPUs and
    Apple's Metal Performance Shaders (MPS) for use with PyTorch.

    Returns:
        bool: True if either CUDA or MPS is available, False otherwise.
    """
    return torch.cuda.is_available() or torch.backends.mps.is_available()

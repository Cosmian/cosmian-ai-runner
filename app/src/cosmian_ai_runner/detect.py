def is_gpu_available():
    import torch

    return torch.cuda.is_available()

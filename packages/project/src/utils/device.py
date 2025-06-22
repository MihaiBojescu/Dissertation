import torch


def get_device() -> torch.device:
    """
    Pick an accelerated backend. Fallback to CPU is none is available.
    """
    if torch.backends.cudnn.is_available() and torch.cuda.is_available():
        print("Running on a CUDA/ROCm device")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("Running on a Metal device")
        return torch.device("mps")

    try:
        import intel_extension_for_pytorch as ipex

        if torch.xpu.device_count() > 0:
            print("Running on an Intel device")
            return torch.device("xpu")
    except:
        pass

    print("Running on CPU")
    return torch.device("cpu")

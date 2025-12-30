# utils/gpu_doctor.py

from __future__ import annotations
import torch, platform, os

def summarize(prefix: str = "GPU Doctor"):
    py = platform.python_version()
    tv = torch.__version__
    cuda = torch.cuda.is_available()
    dev  = torch.cuda.get_device_name(0) if cuda else "N/A"
    idx  = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"[{prefix}] Python={py}  torch={tv}  cuda={cuda}  CUDA_VISIBLE_DEVICES={idx}  device0={dev}")
    return cuda

if __name__ == "__main__":
    summarize()
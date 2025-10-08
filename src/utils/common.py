import os
import json
import random
import platform
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def gpu_available() -> bool:
    import torch
    return torch.cuda.is_available()


def log_device_info():
    import torch, platform
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print(f"[GPU] CUDA available: {n} device(s) -> {names}, torch.version.cuda={torch.version.cuda}")
        print(f"[GPU] cudnn.enabled={torch.backends.cudnn.enabled} "
              f"benchmark={torch.backends.cudnn.benchmark} "
              f"deterministic={torch.backends.cudnn.deterministic}")
    else:
        print("[GPU] CUDA not available; falling back to CPU.")

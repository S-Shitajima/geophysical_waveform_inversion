from typing import Union

import numpy as np
import torch

"""
refenrece: https://github.com/lanl/OpenFWI/blob/main/transforms.py
"""

def log_transform_np(x: Union[float, np.ndarray], k: float = 1.0, c: float = 0.0) -> np.ndarray:
    return (np.log1p(np.abs(k * x) + c)) * np.sign(x)


def log_transform_torch(x: Union[float, torch.tensor], k: float = 1.0, c: float = 0.0) -> torch.tensor:
    return (torch.log1p(torch.abs(k * x) + c)) * torch.sign(x)
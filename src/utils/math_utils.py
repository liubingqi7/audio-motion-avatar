import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x / (1 - x))
    else:
        return np.log(x / (1 - x))
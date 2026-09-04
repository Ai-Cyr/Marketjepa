import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixe les graines Python, NumPy et PyTorch pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

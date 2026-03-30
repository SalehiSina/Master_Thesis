import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim as optim


def set_random_seed(seed: int) -> None:
    """Reset seed for Numpy and PyTorch

    :param seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

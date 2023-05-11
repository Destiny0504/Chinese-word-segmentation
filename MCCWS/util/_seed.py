import torch
import numpy as np
import random


def set_seed(SEED: int = None):
  
  assert SEED != None

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)

  if torch.cuda.is_available():
    # Disable cuDNN benchmark for deterministic selection on algorithm.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
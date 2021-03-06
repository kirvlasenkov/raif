import random
import numpy as np
import os
import config

try:
    import torch
except:
    pass


def fix_all_seeds(seed=config.SEED):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.manual_seed(seed)
    except:
        pass
    try:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

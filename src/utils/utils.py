import os
import random
import datetime as dt
import logging
from logging import StreamHandler, FileHandler, Formatter

import numpy as np
import torch
from torch import nn

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def weights_init_he(m, activation='leaky_relu'):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.1, nonlinearity=activation)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def create_logger(log_filename, log_dir='./logs'):
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter("%(message)s"))

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    now = dt.datetime.now()
    log_filename = now.strftime('%m%d%H%M') + '_' + log_filename
    if os.path.splitext(log_filename)[-1] != '.log':
        log_filename += '.log'
    file_handler = FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(Formatter('%(asctime)s %(message)s'))

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logger = logging.getLogger(__name__)   
    return logger

def add_gaussian_noise(x, noise_factor=0.5):
    noise = torch.randn_like(x) * noise_factor
    x_noisy = x + noise
    return x_noisy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
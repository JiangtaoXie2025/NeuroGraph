# utils/helpers.py
#
# Contains generic helper functions for logging, device management, etc.

import torch
import logging
import os

def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def setup_logger(log_dir="logs", log_name="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, map_location=None):
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

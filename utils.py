# -*- coding: utf-8 -*-
"""
@author: Van Thiep <thiepne24u@gmail.com>
"""
import torch
import config

# Load saved weights
def load_checkpoint(checkpoint_file, model, device=config.device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if checkpoint["state_dict"]:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint['conv3.bias'])

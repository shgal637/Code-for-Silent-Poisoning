import sys
import os
import argparse
import datetime
import time
import random
# import logging
import numpy as np
import torch

def gradient_masking(tensor):
    g_norm = torch.reshape(torch.norm(tensor, p=2, dim=1, keepdim=True), [-1, 1])
    max_norm = torch.max(g_norm)
    stds = torch.sqrt(torch.clamp(max_norm ** 2 / (g_norm ** 2 + 1e-32) - 1.0, min=0.0))
    standard_gaussian_noise = torch.normal(size=(tensor.shape[0], 1), mean=0.0, std=1.0)
    if 'cuda' in str(tensor.device):
        standard_gaussian_noise = standard_gaussian_noise.cuda()
    gaussian_noise = standard_gaussian_noise * stds
    tensor = tensor * (1 + gaussian_noise)
    return tensor


def new_gradient_masking(tensor):
    result = []
    for i in range(tensor.shape[0]):
        result.append(torch.norm(tensor[i], p=2))
    result = torch.stack(result, dim=0)  # [bs]
    g_norm = torch.reshape(result, [-1, 1])  # [bs,1]
    max_norm = torch.max(g_norm)
    stds = torch.sqrt(torch.clamp(max_norm ** 2 / (g_norm ** 2 + 1e-32) - 1.0, min=0.0))  # [bs,1]
    standard_gaussian_noise = torch.normal(size=(tensor.shape[0], 1), mean=0.0, std=1.0)  # [bs,1]
    if 'cuda' in str(tensor.device):
        standard_gaussian_noise = standard_gaussian_noise.cuda()
    gaussian_noise = standard_gaussian_noise * stds  # [bs,1]
    if len(gaussian_noise.shape) != len(tensor.shape):
        for i in range(len(tensor.shape) - len(gaussian_noise.shape)):
            gaussian_noise = torch.unsqueeze(gaussian_noise, -1)
        gaussian_noise = gaussian_noise.cpu().repeat(1, tensor.shape[1], tensor.shape[2], tensor.shape[3]).detach()
        if 'cuda' in str(tensor.device):
            gaussian_noise = gaussian_noise.cuda()
    tensor = tensor * (1 + gaussian_noise)
    return tensor

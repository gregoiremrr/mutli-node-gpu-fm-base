"""
Monitoring utilities for W&B integration.
Add to training/monitoring.py
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch_utils import misc


# Sample generation for image datasets

def generate_sample_grid(model, encoder, n_samples=16, n_steps=100, sigma_data=0.5,
                         device=torch.device('cuda')):
    """
    Generate a grid of samples using Heun's method.

    Args:
        model:       EMA model (already in eval mode)
        encoder:   for decoding latents -> RGB (or StandardRGBEncoder)
        n_samples: must be a perfect square for the grid (e.g., 16 = 4x4)
        n_steps:   integration steps
        sigma_data: noise scaling

    Returns:
        grid: [C, H_grid, W_grid] uint8 tensor
    """
    w = h = int(np.sqrt(n_samples))
    assert w * h == n_samples
    assert n_samples >= 10, "n_samples must be at least 10 to cover all CIFAR-10 classes."

    fixed_classes = torch.arange(10, device=device)
    random_classes = torch.randint(0, 10, (n_samples - 10,), device=device)
    class_indices = torch.cat([fixed_classes, random_classes])
    class_labels = F.one_hot(class_indices, num_classes=10).float()

    x = model.sample(class_labels, n_samples, n_steps, device)

    # Decode to RGB uint8
    images = encoder.decode(x)  # [N, 3, H, W] uint8
    grid = misc.tile_images(images, w=w, h=h)  # [3, H*h, W*w]
    return grid


# Main logging function called from training loop

def log_to_wandb(wandb, state, step_stats, ema, encoder, device, log_samples=True, n_samples=16, n_steps=50):
    """
    Log everything to W&B. Call this from the status reporting block.

    Args:
        wandb:       the wandb module (already imported and initialized)
        state:       training state EasyDict (cur_nimg, etc.)
        step_stats:  training state for the step EasyDict (cur_nimg, etc.)
        ema:         EMA object (to get the EMA model for generation)
        encoder:     for decoding samples
        device:      torch device
        log_samples: whether to generate and log sample images
        n_samples:   number of samples to generate
        n_steps:     Heun steps for generation
    """
    log_dict = {}

    # Per-bin scalar losses
    for key, value in step_stats.items():
        log_dict[f"metrics/{key}"] = step_stats[key]

    # Sample generation
    if log_samples and ema is not None:
        ema_model = ema.get()
        # ema.get() returns either the model or a list of (model, suffix)
        if isinstance(ema_model, list):
            ema_model = ema_model[0][0]  # take the first EMA profile
        ema_model.eval()
        grid = generate_sample_grid(
            ema_model, encoder, n_samples=n_samples,
            n_steps=n_steps, device=device
        )
        # grid is [C, H, W] uint8 -> numpy HWC for wandb
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        log_dict["samples"] = wandb.Image(grid_np, caption=f"step {state.cur_nimg // 1000}k")
        ema_model.train()

    wandb.log(log_dict, step=state.cur_nimg)

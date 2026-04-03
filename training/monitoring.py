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


# --------------------------------------------------------------------------- #
# Sample generation for image datasets                                         #
# --------------------------------------------------------------------------- #

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

    dt = 1.0 / n_steps
    x = torch.randn(n_samples, model.img_channels, model.img_resolution, model.img_resolution,
                     device=device) * sigma_data

    fixed_classes = torch.arange(10, device=device)
    random_classes = torch.randint(0, 10, (n_samples - 10,), device=device)
    class_indices = torch.cat([fixed_classes, random_classes])
    class_labels = F.one_hot(class_indices, num_classes=10).float()

    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full([n_samples], i * dt, device=device)
            k1 = model(x, t, class_labels=class_labels)
            x_pred = x + dt * k1
            t_next = torch.full([n_samples], (i + 1) * dt, device=device)
            k2 = model(x_pred, t_next, class_labels=class_labels)
            x = x + 0.5 * dt * (k1 + k2)

    # Decode to RGB uint8
    images = encoder.decode(x)  # [N, 3, H, W] uint8
    grid = misc.tile_images(images, w=w, h=h)  # [3, H*h, W*w]
    return grid


# --------------------------------------------------------------------------- #
# Task stats figures (adapted from your notebook)                              #
# --------------------------------------------------------------------------- #

def fig_correlation_matrix(mat, title="C"):
    """Heatmap with diverging colormap and value annotations."""
    mat_np = mat.detach().cpu().numpy() if torch.is_tensor(mat) else np.array(mat)
    K = mat_np.shape[0]
    vabs = max(np.abs(mat_np).max(), 1e-8)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    ax.imshow(mat_np, cmap=plt.cm.RdBu, norm=norm, aspect="equal")

    for i in range(K):
        for j in range(K):
            val = mat_np[i, j]
            color = "white" if abs(val) > 0.6 * vabs else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_title(title)
    plt.tight_layout()
    return fig


def fig_weights_bar(w, title="Combination weights"):
    """Bar plot of current task weights."""
    w_np = w.detach().cpu().numpy() if torch.is_tensor(w) else np.array(w)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    colors = ["royalblue" if v >= 0 else "indianred" for v in w_np]
    ax.bar(range(len(w_np)), w_np, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_xticks(range(len(w_np)))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def fig_loss_per_bin(loss_dict, n_tasks):
    """Line plot of per-bin training loss over steps."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    cmap = plt.cm.tab10
    for k in range(n_tasks):
        if k in loss_dict and len(loss_dict[k]) > 0:
            ax.plot(loss_dict[k], label=f"bin {k}", color=cmap(k % 10), linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train loss per time-bin")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Main logging function called from training loop                              #
# --------------------------------------------------------------------------- #

def log_to_wandb(wandb, state, ema, encoder, device, log_samples=True, n_samples=16, n_steps=50):
    """
    Log everything to W&B. Call this from the status reporting block.

    Args:
        wandb:       the wandb module (already imported and initialized)
        state:       training state EasyDict (cur_nimg, etc.)
        ema:         EMA object (to get the EMA model for generation)
        encoder:     for decoding samples
        device:      torch device
        log_samples: whether to generate and log sample images
        n_samples:   number of samples to generate
        n_steps:     Heun steps for generation
    """
    log_dict = {}

    # Per-bin scalar losses
    log_dict[f"task/loss_bin"] = state.losses[-1].item()
    log_dict[f"task/grad_norm_bin"] = state.grad_norms[-1].item()

    # Figures (matplotlib -> wandb.Image)
    fig_C = fig_correlation_matrix(step_stats["C"], title="C (gradient correlation)")
    fig_Ccos = fig_correlation_matrix(step_stats["C_cos"], title="C_cos")
    fig_w = fig_weights_bar(step_stats["w"])
    log_dict["plots/C"] = wandb.Image(fig_C)
    log_dict["plots/C_cos"] = wandb.Image(fig_Ccos)
    log_dict["plots/weights"] = wandb.Image(fig_w)
    plt.close(fig_C)
    plt.close(fig_Ccos)
    plt.close(fig_w)

    # Per-bin loss curves (accumulated over training)
    if history is not None and len(history) > 0:
        fig_loss = fig_loss_per_bin(history, n_tasks)
        log_dict["plots/loss_per_bin"] = wandb.Image(fig_loss)
        plt.close(fig_loss)

    # Sample generation
    if log_samples and ema is not None:
        ema_model = ema.get()
        # ema.get() returns either the model or a list of (model, suffix)
        if isinstance(ema_model, list):
            ema_model = ema_model[0][0]  # take the first EMA profile
        ema_model.eval()
        grid = generate_sample_grid(
            ema_model, encoder, n_samples=n_samples,
            n_steps=n_steps, device=device,
        )
        # grid is [C, H, W] uint8 -> numpy HWC for wandb
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        log_dict["samples"] = wandb.Image(grid_np, caption=f"step {state.cur_nimg // 1000}k")
        ema_model.train()

    wandb.log(log_dict, step=state.cur_nimg)

# Flow Matching Training

A multi-GPU training pipeline for flow-matching generative models on images
(pixels or VAE latents), with mixed-precision training, post-hoc EMA, and
W&B logging.
The infrastructure (`dnnlib`, `torch_utils`, the U-Net architectures in
`training/networks.py`, post-hoc EMA, `dataset_tool.py`, FID/FD-DINOv2
metrics, and persistence-based pickling) is borrowed from NVIDIA's
[EDM2](https://github.com/NVlabs/edm2). The flow-matching model, loss, samplers, and configuration layout sit on top.

## Layout

```
.
├── train.py                   # Build a config and launch training.
├── generate_images.py         # Sample from a saved snapshot pickle.
├── calculate_metrics.py       # FID and FD-DINOv2 against a reference dataset.
├── reconstruct_phema.py       # Post-hoc EMA reconstruction (EDM2).
├── dataset_tool.py            # Pack a folder of images into a zip dataset.
├── training/
│   ├── training_loop.py       # The actual training loop.
│   ├── model.py               # FlowMatchingModel + sample().
│   ├── loss.py                # FlowMatchingLoss.
│   ├── networks.py            # SongUNet / DhariwalUNet (from EDM).
│   ├── encoders.py            # StandardRGB and Stability VAE encoders.
│   ├── schedulers.py          # LR schedules.
│   ├── phema.py               # Power-function and traditional EMA.
│   ├── monitoring.py          # W&B logging helpers.
│   └── dataset.py             # Streaming image dataset (zip or folder).
├── torch_utils/               # Distributed, persistence, training stats (EDM).
├── dnnlib/                    # EasyDict, class/func construction by name (EDM).
├── scripts/                   # Shell scripts: env setup, training, metrics.
├── datasets/                  # Place your packed datasets here.
├── training-runs/             # Output runs (one timestamped subdir per launch).
├── fid-refs/                  # Reference statistics for FID/FD-DINOv2.
└── out/                       # Generated images.
```

## Setup

```bash
# Install the Python environment (CUDA 12.4 wheel; adjust as needed).
bash scripts/module.sh
```

A `Dockerfile` is also provided.

## End-to-end workflow

### 1. Pack the dataset

```bash
python dataset_tool.py convert \
    --source=raw_cifar/ \
    --dest=datasets/cifar10.zip \
    --resolution=32x32
```

For VAE-latent training, use `dataset_tool.py encode` to pre-encode images.

### 2. Train

```bash
bash scripts/training/script-cifar10.sh
```

Or directly:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10 \
    --data=datasets/cifar10.zip \
    --preset=fm-cifar10
```

Each launch creates a timestamped subdirectory inside `--outdir` (e.g.
`training-runs/cifar10/260423_160712_fm-cifar10`). Pointing `--outdir` at
an existing run that contains a `training-state-*.pt` resumes from the
latest checkpoint.

### 3. Reconstruct post-hoc EMA snapshots (optional)

```bash
python reconstruct_phema.py \
    --indir=training-runs/cifar10/<run-dir> \
    --outdir=training-runs/cifar10/<run-dir> \
    --outstd=0.100
```

### 4. Compute reference statistics for the dataset

```bash
bash scripts/metrics/ref50k.sh
```

### 5. Generate images

```bash
bash scripts/metrics/gen50k.sh
```

### 6. Compute FID / FD-DINOv2

```bash
bash scripts/metrics/fid50k.sh
```

## Adding a new training run

The configuration is split into two preset dictionaries at the top of
`train.py`:

- `**dataset_presets**` holds everything intrinsic to the data:
`sigma_data`, the network architecture (`net_kwargs`), the sampler used
during monitoring (`sampler_kwargs`), and the LR scheduler family
(`lr_scheduler_kwargs`).
- `**config_presets**` describes a particular training run on top of a
dataset: which dataset to use, conditional vs unconditional, total
`nimg`, batch size, prediction target (`x` or `v`), classifier-free
guidance dropout, channel width, dropout, base LR, and gradient clipping.
Adding a new run is mostly a matter of editing those two dictionaries and
pointing `--preset` / `--data` at the new entry. Per-run overrides are
exposed as CLI flags on `train.py`. The two dictionaries are required to
have disjoint keys (asserted at startup) so it's always clear which preset
a knob lives in.

## Monitoring

Loss, learning rate, gradient norm, gradient-clip coefficient, and timing
counters are pushed to W&B at every `--status` interval, alongside a grid
of samples generated from the EMA model with the dataset's configured
sampler. Scalar metrics are duplicated against three x-axes (training
step, images seen, wall-clock time); plots are tied to the training step
axis.

## Credits

Built on top of NVIDIA's [EDM](https://github.com/NVlabs/edm) and
[EDM2](https://github.com/NVlabs/edm2).
import os
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop
import datetime

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'fm-cifar10': dnnlib.EasyDict(
        total_nimg=200_000 * 64,   # 200k steps * batch_size = total_nimg
        batch_size=256,
        pred="v",
        p_uncond_labels=0.13,
        channels=128,
        dropout=0.0,
        lr=1e-3,
        max_clip_norm=1
    ),
    'fm-cifar10-xpred': dnnlib.EasyDict(
        total_nimg=200_000 * 64,   # 200k steps * batch_size = total_nimg
        batch_size=256,
        pred="x",
        p_uncond_labels=0.13,
        channels=128,
        dropout=0.0,
        lr=1e-3,
        max_clip_norm=1
    )
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='fm-cifar10', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset and Dataloader.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    c.data_loader_kwargs = dict(
        class_name='torch.utils.data.DataLoader',
        pin_memory=opts.pin_memory,
        num_workers=opts.num_workers,
        prefetch_factor=opts.prefetch_factor
    )

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    # Hyperparameters.
    c.total_nimg=opts.total_nimg
    c.batch_size=opts.batch_size
    c.model_kwargs = dnnlib.EasyDict(
        class_name='training.model.FlowMatchingModel',
        pred=opts.pred,
        sigma_data=0.5,
        net_kwargs=dnnlib.EasyDict(
            class_name='training.networks.SongUNet',
            embedding_type='positional',
            encoder_type='standard',
            decoder_type='standard',
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=opts.channels,
            channel_mult=[2, 2, 2],
            dropout=opts.dropout
        ),
        use_fp16 = opts.fp16
    )
    c.ema_kwargs = dict(class_name='training.phema.PowerFunctionEMA')
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.FlowMatchingLoss', p_uncond=opts.p_uncond_labels)
    c.optimizer_kwargs = dict(class_name='torch.optim.AdamW', weight_decay=1e-3, betas=(0.9, 0.99))
    c.lr_kwargs = dnnlib.EasyDict(
        func_name='training.schedulers.cosine_lr',
        base_lr=opts.lr,
        total_nimg=opts.total_nimg
    )
    c.max_clip_norm = opts.max_clip_norm

    # Performance-related options.
    c.max_batch_gpu = opts.max_batch_gpu or None
    c.loss_scaling = opts.ls
    c.cudnn_benchmark = opts.bench

    # I/O-related options.
    c.status_nimg = opts.status or None
    c.snapshot_nimg = opts.snapshot or None
    c.checkpoint_nimg = opts.checkpoint or None
    c.seed = opts.seed
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, pretrained_pkl, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Pretrained model:        {pretrained_pkl}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.model_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, pretrained_pkl, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, pretrained_pkl=pretrained_pkl, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Output directory (resumed if exists with checkpoints)', metavar='DIR', type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--pretrained-pkl',   help='Pretrained snapshot path', metavar='DIR', type=str,   default=None)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='fm-cifar10', show_default=True)

# Hyperparameters. (should be None by default and be configured in the presets)
@click.option('--total_nimg',       help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch-size',       help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--pred',             help='Quantity predicted by the network', metavar='x/v',    type=str, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--max_clip_norm',    help='Max gradient norm for clipping', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)

# Performance-related options.
@click.option('--max-batch-gpu',    help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=None, show_default=True)
@click.option('--pin-memory',       help='Enable mixed-precision training', metavar='BOOL',     default=True, show_default=True)
@click.option('--num-workers',      help='Number of workers in the dataloader', metavar='INT',  type=int, default=2, show_default=True)
@click.option('--prefetch_factor',  help='Number of batches for each worker', metavar='INT',    type=int, default=2, show_default=True)
@click.option('--fp16/--no-fp16',   help='Enable mixed-precision training', metavar='BOOL',     default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='128Mi', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)


def cmdline(outdir, pretrained_pkl, dry_run, **opts):
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)

    # If outdir has no timestamp yet, add one.
    # If it already exists (user is resuming), use as-is.
    if os.path.isdir(outdir) and any(f.startswith('training-state-') for f in os.listdir(outdir)):
        run_dir = outdir
        dist.print0(f'Resuming from {run_dir}')

        if pretrained_pkl:
            raise click.ClickException('Cannot use --pretrained when resuming from an existing run')
    else:
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        if dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                name_bytes = now.encode('utf-8')
            else:
                name_bytes = b'\x00' * 20
            name_tensor = torch.ByteTensor(list(name_bytes)).cuda()
            torch.distributed.broadcast(name_tensor, src=0)
            now = bytes(name_tensor.tolist()).decode('utf-8').rstrip('\x00')
        preset_name = opts.get('preset', 'run')
        run_dir = os.path.join(outdir, f'{now}_{preset_name}')

    print_training_config(run_dir=run_dir, pretrained_pkl=pretrained_pkl, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=run_dir, pretrained_pkl=pretrained_pkl, c=c)
    torch.distributed.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------

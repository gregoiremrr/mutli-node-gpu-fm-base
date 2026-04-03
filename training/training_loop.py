import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import wandb
from training.monitoring import log_to_wandb

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs,
    encoder_kwargs,
    data_loader_kwargs,
    model_kwargs,
    loss_kwargs,
    optimizer_kwargs,
    lr_kwargs,
    ema_kwargs,
    max_clip_norm,

    run_dir,                # Output directory.
    seed,                   # Global random seed.
    batch_size,             # Total batch size for one training iteration.
    max_batch_gpu,          # Limit batch size per GPU. None = no limit.
    total_nimg,             # Train for a total of N training images.
    status_nimg,            # Report status every N training images. None = disable.
    snapshot_nimg,          # Save network snapshot every N training images. None = disable.
    checkpoint_nimg,        # Save state checkpoint every N training images. None = disable.

    loss_scaling,           # Loss scaling factor for reducing FP16 under/overflows.
    cudnn_benchmark,        # Enable torch.backends.cudnn.benchmark?
    force_finite = True,    # Get rid of NaN/Inf gradients before feeding them to the optimizer.
):
    # Device.
    device = torch.device('cuda')

    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if max_batch_gpu is None or max_batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    else:
        batch_gpu = max_batch_gpu

    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or snapshot_nimg % batch_size == 0
    assert checkpoint_nimg is None or checkpoint_nimg % batch_size == 0

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing model...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    model = dnnlib.util.construct_class_by_name(**model_kwargs, **interface_kwargs)
    model.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(model, [
            torch.zeros([batch_gpu, model.img_channels, model.img_resolution, model.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, model.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=model.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(model=model, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, model=model, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    assert total_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {total_nimg // 1000} kimg:')
    dist.print0()

    # Setup WandB
    if dist.get_rank() == 0:
        if not state.get('wandb_run_id', None):
            state.wandb_run_id = wandb.util.generate_id()
        wandb_run = wandb.init(
            project="flow-matching",
            name=os.path.basename(run_dir),
            dir=run_dir,
            id=state.wandb_run_id,
            resume="allow"
        )

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    step_stats = dnnlib.EasyDict()
    while True:
        done = (state.cur_nimg >= total_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # W&B logging.
            if wandb_run is not None:
                log_to_wandb(
                    wandb=wandb,
                    state=state,
                    step_stats=step_stats,
                    ema=ema,
                    encoder=encoder,
                    device=device,
                    log_samples=True,
                    n_samples=16,
                    n_steps=50,
                )

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, total_nimg // 1000)
            if state.cur_nimg == total_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(model) if hasattr(optimizer, 'get_ema') else model
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_model, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_model).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(model)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        step_stats.loss = 0
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))

                raw_loss = loss_fn(model=ddp, images=images, labels=labels.to(device))

                training_stats.report('Loss/loss', raw_loss)
                step_stats.loss += raw_loss.item() / num_accumulation_rounds

                loss = raw_loss * (loss_scaling / num_accumulation_rounds)
                loss.backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Unscale the gradients
        inv_scale = 1 / loss_scaling
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(inv_scale) # In-place multiplication
                
                if force_finite:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)

        # Clip
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)

        optimizer.step()

        step_stats.grad_norm = grad_norm.item()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

    if wandb_run is not None:
        wandb.finish()

#----------------------------------------------------------------------------

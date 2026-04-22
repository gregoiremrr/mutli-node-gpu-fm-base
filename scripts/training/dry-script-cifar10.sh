torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10 \
    --data=datasets/cifar10.zip \
    --preset=fm-cifar10 \
    --max-batch-gpu=32 \
    --fp16 \
    --status=1Ki \
    --snapshot=64Ki \
    --checkpoint=128Ki \
    --dry-run

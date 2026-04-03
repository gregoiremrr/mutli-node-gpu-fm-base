torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10_test \
    --data=datasets/cifar10.zip \
    --preset=fm-cifar10 \
    --max-batch-gpu=32 \
    --fp16 \
    --status=128Ki \
    --snapshot=8Mi \
    --checkpoint=128Mi \
    --dry-run

torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs/cifar10/260407_092329_fm-cifar10 \
    --data=datasets/cifar10.zip \
    --preset=fm-cifar10 \
    --max-batch-gpu=512 \
    --no-fp16 \
    --status=20Ki \
    --snapshot=256Ki \
    --checkpoint=512Ki

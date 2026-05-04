export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

torchrun --standalone --nproc_per_node=2 train.py \
    --outdir=training-runs/cifar10/260425_151340_fm-cifar10-trig \
    --data=datasets/cifar10.zip \
    --preset=fm-cifar10-trig \
    --max-batch-gpu=512 \
    --no-fp16 \
    --status=20Ki \
    --snapshot=256Ki \
    --checkpoint=512Ki

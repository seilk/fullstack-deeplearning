python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env main.py \
        --model resnet50 \
        --batch-size 64 \
        --epochs 2 \
        --data-path /data/imagenet

debugpy-run -m torch.distributed.launch -- \
            --nproc_per_node=2 \
            --use_env main.py \
            --model resnet50 \
            --batch-size 64 \
            --epochs 2 \
            --data-path /data/imagenet
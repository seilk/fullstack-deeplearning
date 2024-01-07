python main.py \
        --args1 1 \
        --args1 2 \
        --args1 3 \
        --args1 4 

debugpy-run main.py \
            --args1 1 \
            --args1 2 \
            --args1 3 \
            --args1 4 \

debugpy-run -m torch.distributed.launch -- \
            --nproc_per_node=4 \
            --args1 1 \
            --args1 2 \
            --args1 3 \
            --args1 4 \

            
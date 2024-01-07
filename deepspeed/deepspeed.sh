deepspeed --num_gpus=4 main.py \
--args
--deepspeed_config config.json \

# when using specific gpus
deepspeed —inlcude localhost:0, 1 main.py \
--args \
config.json

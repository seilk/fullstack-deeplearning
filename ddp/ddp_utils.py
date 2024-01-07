import os
import sys
import random 

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

def throughOnlyOnce(rank, func):
    def wrapper(*args, **kwargs):
        if rank == 0:
            func(*args, **kwargs)
    return wrapper

def set_seed_ddp(seed: int, rank: int):
    torch.manual_seed(seed+rank) # PyTorch를 위한 Seed 설정
    torch.cuda.manual_seed(seed+rank) # CUDA를 위한 Seed 설정 (GPU 사용 시)
    torch.cuda.manual_seed_all(seed+rank) # 멀티 GPU 사용 시 모든 CUDA 장치를 위한 Seed 설정
    torch.backends.cudnn.deterministic = True # CUDNN 연산자의 결정론적 옵션 활성화
    torch.backends.cudnn.benchmark = False # CUDNN 벤치마크 모드 비활성화
    np.random.seed(seed+rank) # Numpy를 위한 Seed 설정
    random.seed(seed+rank) # Python 내장 random 모듈을 위한 Seed 설정

def wrappingModelwithDDP(models, local_gpu_id):
    wrapped_models = []
    for model in models:
        has_trainable_params = any(p.requires_grad for p in model.parameters())
        if not has_trainable_params:
            print(f"The model {model.__class__.__name__} does not have any trainable parameters.")
            wrapped_models.append(model.cuda(local_gpu_id))
            continue
        wrapped_models.append(DDP(module=model, device_ids=[local_gpu_id], find_unused_parameters=False))
    return wrapped_models if len(wrapped_models) != 1 else wrapped_models[0]
    
def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_environment_variable(var_name, default=None):
    """Helper function to get an environment variable or return a default."""
    return os.environ.get(var_name, default)


def initialize_distributed_backend(args):
    """Initializes the distributed backend based on the environment."""
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url + get_environment_variable('MASTER_PORT', '29500'),
        world_size=args.world_size,
        rank=args.rank,
    )


def set_device_for_distributed(args):
    """Sets the device for the current distributed process."""
    torch.cuda.set_device(args.gpu)
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)


def is_master_process(args):
    """Checks if the current process is the master process."""
    return args.rank == 0


def init_distributed_mode(args):
    """Initializes distributed mode based on the environment."""
    if 'WORLD_SIZE' in os.environ:
        args.rank = int(get_environment_variable("RANK", 0))
        args.world_size = int(get_environment_variable('WORLD_SIZE', 1))
        args.gpu = int(get_environment_variable('LOCAL_RANK', 0))
        print(f"RANK: {args.rank}, WORLD_SIZE: {args.world_size}, GPU: {args.gpu}")
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(get_environment_variable('SLURM_PROCID', 0))
        args.gpu = args.rank % torch.cuda.device_count()
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank = args.gpu = 0
        args.world_size = 1
    else:
        print('Does not support training without GPU.', force=True)
        sys.exit(1)

    initialize_distributed_backend(args)
    set_device_for_distributed(args)
    setup_for_distributed(is_master_process(args))
    dist.barrier()

import torch.distributed as dist
import torch.multiprocessing as mp
import torch


def f(rank):
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=4, rank=rank)
    t = torch.rand(1)
    gather_t = [torch.ones_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, t)
    print(rank, t, gather_t)


if __name__ == '__main__':
    mp.spawn(f, nprocs=4, args=())
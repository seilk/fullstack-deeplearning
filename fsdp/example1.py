import functools

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.t5.modeling_t5 import T5Block


def train(rank, world_size, batch_size, epochs=100):
    global_rank = rank
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:33445",
        rank=global_rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=rank)

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for data in tqdm(dataloader):
            data = {k: data[k].to(rank) for k in data}

            optimizer.zero_grad()

            output = model(**data)
            output.loss.backward()
            optimizer.step()
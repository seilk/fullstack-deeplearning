import torch
from .ddp_utils import init_distributed_training, wrappingModelwithDDP, set_seed_ddp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from train import Trainer

def main(args, config):
	
	if args.ddp == True:
		init_distributed_training(args)
		set_seed_ddp(config['seed'], args.rank)
		local_gpu_id = args.rank
		num_processes = args.world_size

	train_dataset = Dataset(...)

	if args.ddp == True :
		total_num_workers = config['train']['num_workers'] * num_processes
		total_batch_size = config['train']['batch'] * num_processes
		train_dataset_sampler = DistributedSampler(train_dataset, shuffle=True)
		train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, sampler=train_dataset_sampler, num_workers=total_num_workers)
	else:
		train_dataloader = DataLoader(train_dataset, batch_size=config['train']['batch'], shuffle=True, num_workers=config['train']['num_workers'])
	
	if torch.cuda.is_available():
		if args.ddp == True:
			device = torch.device(f'cuda:{local_gpu_id}')
		else :
			device = torch.device(f'cuda')
	else:
		device = torch.device('cpu')

	...

	if args.ddp == True:
		model = wrappingModelwithDDP([model], local_gpu_id=local_gpu_id)
	
	trainer = Trainer(args, config, device, model, train_dataloader, ... )
	trainer()

if __name__ == "__main__":
    args = ...
    config = ...
    main(args, config)

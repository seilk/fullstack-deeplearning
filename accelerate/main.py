import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from train import Trainer

def main(args, config):
	global accelerator
	accelerator = Accelerator()
	train_dataset = Dataset(...)	
	device = accelerator.device

	...

	model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, 
                                                                        optimizer, 
                                                                        train_dataloader, 
                                                                        scheduler)
	trainer = Trainer(args, config, device, model, train_dataloader, accelerator, ... )
	trainer()

if __name__ == "__main__":
    args = ...
    config = ...
    main(args, config)
    
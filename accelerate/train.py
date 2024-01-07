import os
from attrs import define, field
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict
from utils import accThroughOnlyOnce

@define
class Trainer:
    args: argparse.Namespace
    config: dict
    device: torch.device = torch.device("cpu")
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    epochs: int = 10
    accelerator: field(default=None)

    def __call__(self):
        """
        Implement the training logic here.
        """
        for epoch in range(self.epochs):
            self.train_epoch(self.model, self.train_loader, self.optimizer, self.accelerator)
            # If you want to push the model to the validation module is in where different from the training module,
            # You have to push `model.module` instead of `model`
            self.validate_epoch(self.model, self.val_loader)
            

    def train_epoch(self, model, train_loader, optimizer):
        """
        Implement the training logic for one epoch.
        """
        loss = torch.Tensor([0.0]).to(self.device)
        self.model.train()
        for batch in train_loader:
            # Training logic for each batch
            ...
            # loss.backward()
            self.accelerator.backward(loss)
            optimizer.step()
            self.save_model_state_dict(model, self.accelerator, 'model_state.pth')
            dist.barrier()  # Wait for all processes to save the model
    
    @accThroughOnlyOnce
    def validate_epoch(self):
        """
        Implement the validation logic for one epoch.
        """
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                # Validation logic for each batch
                ...

    @accThroughOnlyOnce
    def save_model_state_dict(self, model, accelerator, filename):
        model = accelerator.unwrap_model(model)
        accelerator.save_state(output_dir="my_checkpoint")
    
# Example usage
# trainer = Trainer(model, train_loader, val_loader, optimizer)
# trainer()

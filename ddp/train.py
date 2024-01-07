from attrs import define, field
import torch
import argparse
from collections import OrderedDict

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

    def __call__(self):
        """
        Implement the training logic here.
        """
        for epoch in range(self.epochs):
            self.train_epoch(self.model, self.train_loader, self.optimizer, self.epochs)
            self.validate_epoch()

    def train_epoch(self, model, train_loader, optimizer, epochs):
        """
        Implement the training logic for one epoch.
        """
        
        self.model.train()
        for batch in self.train_loader:
            # Training logic for each batch
            ...
            self.save_model_state_dict(model, 'model_state.pth')
            

    def validate_epoch(self):
        """
        Implement the validation logic for one epoch.
        """
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                # Validation logic for each batch
                ...

    def save_model_state_dict(self, model, filename):
        try:
            # Try to access the underlying model in DDP
            state_dict = model.module.state_dict()
        except AttributeError:
            # If model is not wrapped in DDP, save normally
            state_dict = model.state_dict()

        # Optionally, you can remove 'module.' prefix from state_dict keys
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            clean_state_dict[name] = v

        torch.save(clean_state_dict, filename)


# Example usage
# trainer = Trainer(model, train_loader, val_loader, optimizer)
# trainer()

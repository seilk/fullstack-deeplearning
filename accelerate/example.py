import torch
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator # add

accelerator = Accelerator() # add
# - device = 'cpu'
device = accelerator.device # add

model = torch.nn.Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters())
lr_scheduler = ...

dataset = load_dataset('my_dataset')
data = torch.utils.data.DataLoader(dataset, shuffle=True)

model, optimizer, data = accelerator.prepare(model, optimizer, lr_scheduler, data) # add

model.train()
for epoch in range(10):
	for source, targets in data:
		# source = source.to(device)
		# targets = targets.to(device)

		optimizer.zero_grad()

		output = model(source)
		loss = F.cross_entropy(output, targets)

		# -   loss.backward()
		accelerator.backward(loss) # add
		optimizer.step()
		lr_scheduler.step()
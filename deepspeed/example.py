import torch
import torch.nn.functional as F
from datasets import load_dataset
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

model = torch.nn.Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters())
lr_scheduler = ...

dataset = load_dataset('my_dataset')
data = torch.utils.data.DataLoader(dataset, shuffle=True)

model, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                model=model,
                                                model_parameters=model.parameters())

deepspeed.init_distributed()
model.train()

for epoch in range(10):
	for source, targets in data:
		# source = source.to(device)
		# targets = targets.to(device)

		optimizer.zero_grad()

		output = model(source)
		loss = F.cross_entropy(output, targets)

		model.backward(loss) # add
		model.step()
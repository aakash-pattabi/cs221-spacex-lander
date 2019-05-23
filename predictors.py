import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class LinearPredictor(nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.hidden = nn.Linear(state_size + action_size, 1)

	def forward(self, x):
		return self.hidden(x)

# Default ReLU activation
class NNPredictor(nn.Module):
	def __init__(self, hidden_layers, input_size, output_size):
		super().__init__()
		self.n_layers = len(hidden_layers)
		hidden_layers = [input_size] + hidden_layers + [output_size]
		self.layers = nn.ModuleList()
		for i in range(len(hidden_layers) - 1):
			self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

	def forward(self, x):
		for i in range(self.n_layers):
			x = self.layers[i](x)
			x = F.relu(x)
		x = self.layers[-1](x)
		return x

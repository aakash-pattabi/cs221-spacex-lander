import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from collections import OrderedDict

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

'''
Let [action_buckets] be an ordered dictionary specifying for each of the plausible 
elements in the action vector, how many buckets over which to discretize its values. The 
values in the dictionary may be vectors of the discretized "pivot" points, to make indexing / 
action-creation easier...

Different actions may be discretized differently, which is where the challenge comes from. 
Finally, make sure that [action_buckets] has _torch_ tensors as its values, not _numpy_ ones
'''
class DiscQLPredictor(NNPredictor):
	def __init__(self, hidden_layers, input_size, action_buckets):
		self.action_length = len(action_buckets)
		self.output_size = sum([len(v) for k, v in action_buckets.items()])
		super().__init__(hidden_layers, input_size, self.output_size)
		num_buckets_per_action = [0] + [len(v) for k, v in action_buckets.items()]
		self.act_idx = torch.cumsum(torch.Tensor(num_buckets_per_action), dim = 0).int()
		self.keys = [k for k in action_buckets.keys()]
		self.action_buckets = action_buckets

	# [x] is the output of a forward pass
	def get_actions(self, x):
		x = x.view(-1, self.output_size)
		actions = torch.zeros((len(x), self.action_length))
		q_vals = torch.zeros(len(x))
		for i in range(len(self.act_idx) - 1):
			maxs, indices = x[:,self.act_idx[i]:self.act_idx[i+1]].max(dim = 1)
			actions[:,i] = self.action_buckets[self.keys[i]][indices]
			q_vals += maxs
		return actions, q_vals

	# [x] is the output of a forward pass
	def get_q_val_for_actions(self, x, a):
		a = a.view(-1, self.action_length)
		b = torch.zeros_like(a)
		q_vals = torch.zeros(len(x))
		i = 0
		for k, v in self.action_buckets.items():
			tmp = torch.abs(a[:,i].view(-1, 1) - v)
			__, indices = tmp.min(dim = 1)
			indices += self.act_idx[i]
			b[:,i] = indices
			i += 1
		b = b.long()
		return x.gather(1, b).sum(dim = 1)

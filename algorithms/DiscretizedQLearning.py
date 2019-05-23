import numpy as np 
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from rocket_lander.constants import * 
from algorithms.TemporalDifference import TDAgent
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import random
from tensorboardX import SummaryWriter

# Cartesian product of three np.arrays
def three_cartesian_product(a, b, c): 
	out = np.array(np.meshgrid(a, b, c)).T.reshape((-1, 3))
	return out

'''
Class: ReplayBuffer

Stores (s, a, r, sp) tuples to feed into a DQN as below. Prunes the replay 
buffer by removing [drop] percent of the samples after evory network update to 
prevent old data from skewing the training data distribution. 
'''
class ReplayBuffer(object):
	def __init__(self, warmup, minibatch_size, device):
		self.warmup = warmup
		self.minibatch_size = minibatch_size
		self.device = device	# Used for tensor-create operations
		self.buffer = []
		self.size = 0

	def push(self, s, a, r, sp):
		self.buffer.append([s, a, r, sp])
		self.size += 1

	def sample(self):
		tuples = random.sample(self.buffer, self.minibatch_size)
		s = torch.Tensor([tuples[i][0] for i in range(self.minibatch_size)], device = self.device)
		a = torch.Tensor([tuples[i][1] for i in range(self.minibatch_size)], device = self.device)
		r = torch.Tensor([tuples[i][2] for i in range(self.minibatch_size)], device = self.device)
		sp = torch.Tensor([tuples[i][3] for i in range(self.minibatch_size)], device = self.device)
		return s, a, r, sp

	def clean(self, drop):
		drop = int(drop*len(self.buffer))
		self.buffer = self.buffer[drop:]
		self.size -= drop

'''
Class: PrioritizedReplayBuffer
'''
class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, warmup, minibatch_size, device):
		super().__init__(warmup, minibatch_size, device)
		self.priorities = []

	def push(self, s, a, r, sp):
		self.buffer.append([s, a, r, sp])
		self.priorities.append(r)
		self.size += 1

	def sample(self):
		p = np.array(self.priorities)
		p += np.max(np.abs(p))
		p /= np.sum(p)
		indices = np.random.choice(self.size, size = self.minibatch_size, p = p)
		tuples = [self.buffer[i] for i in indices]
		s = torch.Tensor([tuples[i][0] for i in range(self.minibatch_size)], device = self.device)
		a = torch.Tensor([tuples[i][1] for i in range(self.minibatch_size)], device = self.device)
		r = torch.Tensor([tuples[i][2] for i in range(self.minibatch_size)], device = self.device)
		sp = torch.Tensor([tuples[i][3] for i in range(self.minibatch_size)], device = self.device)
		return s, a, r, sp

	def clean(self, drop):
		drop = int(drop*len(self.buffer))
		self.buffer = self.buffer[drop:]
		self.priorities = self.priorities[drop:]
		self.size -= drop

'''
Class: DiscretizedQAgent

Implements a DQN with fixed Q-targets using neural network function approximators. 
'''
class DiscretizedQAgent(TDAgent):
	def __init__(self, env, lr, discount, \
			epsilon_start, epsilon_min, epsilon_decay, print_debug, \
			disc_buckets, predictor, tau, optimizer, minibatch_size, device):
		super().__init__(env, lr, discount, \
				epsilon_start, epsilon_min, epsilon_decay, print_debug)

		self.device = device
		# Let the discretization buckets be tuples of values in sorted order that
		# span the range (lb_val, ub_val) for each element val in an action 3-tuple. 
		# Think of each element in a disc bucket tuple as the mean of its bucket
		self.main_thrust_buckets = np.array(disc_buckets["main_thrust"])
		self.side_thrust_buckets = np.array(disc_buckets["side_thrust"])
		self.nozzle_buckets = np.array(disc_buckets["nozzle"])

		# Function appoximators
		self.action_network = predictor.to(self.device)
		self.target_network = predictor.to(self.device)
		self.copy_network_parameters()

		# Hyperparameters and optimizer; [tau] is the number of _environment steps_ until
		# parameter copies b/w the target and actor networks (~100 steps/episode in SpaceX rocket env)
		self.tau = tau
		self.minibatch_size = minibatch_size
		self.optimizer = optimizer(self.action_network.parameters(), lr = self.lr)

		# Interpretable mapping of action indices to 3-tuple actions
		self.actions_mat = None
		self._init_actions()

		# Data and counting variables
		self.n_steps = 0
		self.buffer = ReplayBuffer(1000, minibatch_size, self.device)

		# Logger for TensorboardX
		self.writer = SummaryWriter("runs/test")

	def _init_actions(self):
		self.actions_mat = three_cartesian_product(self.main_thrust_buckets, \
			self.side_thrust_buckets, self.nozzle_buckets)

	def next_action(self, s):
		with torch.no_grad():
			q_vals = self.action_network.forward(torch.from_numpy(s).float())
		idx = np.argmax(q_vals.numpy())
		return self.actions_mat[idx,:]

	def get_action_index(self, a):
		diff = self.actions_mat - a
		s = np.sum(diff, axis = 1)
		return np.argmin(s)

	def copy_network_parameters(self):
		self.target_network.load_state_dict(self.action_network.state_dict())

	def update(self, s, a, r, sp):
		# Update counter
		self.n_steps += 1

		# Transform action triple into index in actios_mat and push to buffer
		a = self.get_action_index(a)
		self.buffer.push(s, a, r, sp)

		# Proceed only if sufficient samples in the buffer
		if self.buffer.size < self.buffer.warmup:
			return

		# Sample from ER buffer
		s, a, r, sp = self.buffer.sample()

		# Forward pass to compute next state optimal Q-values
		with torch.no_grad():
			q_vals = self.target_network.forward(sp)
		opt_q_vals, __ = torch.max(q_vals, dim = 1)

		# Forward pass to compute Q-values of current states
		cur_q_vals = self.action_network.forward(s)
		cur_q_vals = cur_q_vals.gather(1, a.view(-1, 1).long())

		# Compute TD target and loss
		td_target = (r + self.discount_rate*opt_q_vals)
		loss = F.mse_loss(td_target, cur_q_vals)

		# Update training network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if (self.n_steps % self.tau == 0):
			# Log model loss, avg. Q-value, and magnitude of gradient updates to Tensorboard
			loss = loss.detach().numpy()
			grad_mean = np.mean([torch.mean(torch.abs(x.grad.data)).numpy() for x in self.action_network.parameters()])
			q_mean = torch.mean(cur_q_vals).detach().numpy()
			self.writer.add_scalar("actor_loss", loss, self.n_steps)
			self.writer.add_scalar("avg_gradient_update_mag", grad_mean, self.n_steps)
			self.writer.add_scalar("avg_current_q_value", q_mean, self.n_steps)

			# Print debugging information if requested
			if self.print_debug:
				print("Network loss: {n:.{d}}".format(n = loss, d = 2))
				print("Avg. magnitude of gradient updates: {n:.{d}}".format(n = grad_mean, d = 2))
				print("Avg. (current) q-value: {n:.{d}}".format(n = q_mean, d = 2))

			# Update target network, anneal epsilon, and prune minibatch of old samples. 
			# Arbitrarily we drop 10% of the buffer -- should tune this, but whatever...
			self.copy_network_parameters()
			self.anneal_epsilon()
			self.buffer.clean(0.1)

	def get_pickleable(self):
		output = {
			"action_network" : self.action_network.state_dict(), 
			"target_network" : self.target_network.state_dict(), 
			"actions_mat" : self.actions_mat,
			"n_steps" : self.n_steps, 
			"lr" : self.lr, 
			"discount_rate" : self.discount_rate, 
			"eps" : self.epsilon,
			"eps_min" : self.epsilon_min, 
			"eps_decay" : self.epsilon_decay, 
			"tau" : self.tau
		}

		# Assume that a model will always be saved before (forced) exit, meaning we 
		# may close the object's log writer before pickling...
		self.writer.close()

		return output

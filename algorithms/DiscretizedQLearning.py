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

# Given matrix a, return a matrix consisting of the values 
# in a mapped to their closest values in a
def get_closest(a, b):
	c = np.abs(a[:, None] - b[None, :])
	mins = np.argmin(c, axis = 1)
	out = np.take(b, mins)
	return out

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
	def __init__(self, warmup, minibatch_size):
		self.warmup = warmup
		self.minibatch_size = minibatch_size
		self.buffer = []
		self.size = 0

	def push(self, s, a, r, sp):
		self.buffer.append([s, a, r, sp])
		self.size += 1

	def sample(self):
		tuples = random.sample(self.buffer, self.minibatch_size)
		s = np.stack([tuples[i][0] for i in range(self.minibatch_size)], axis = 0)
		a = np.stack([tuples[i][1] for i in range(self.minibatch_size)], axis = 0)
		r = np.array([tuples[i][2] for i in range(self.minibatch_size)])
		sp = np.stack([tuples[i][3] for i in range(self.minibatch_size)], axis = 0)
		return s, a, r, sp

	def clean(self, drop):
		drop = int(drop*len(self.buffer))
		self.buffer = self.buffer[drop:]
		self.size -= drop

'''
Class: PrioritizedReplayBuffer
'''
class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, warmup, minibatch_size):
		super().__init__(warmup, minibatch_size)
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
		s = np.stack([tuples[i][0] for i in range(self.minibatch_size)], axis = 0)
		a = np.stack([tuples[i][1] for i in range(self.minibatch_size)], axis = 0)
		r = np.array([tuples[i][2] for i in range(self.minibatch_size)])
		sp = np.stack([tuples[i][3] for i in range(self.minibatch_size)], axis = 0)
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
		self.action_network = predictor.to(self.device)
		self.target_network = predictor.to(self.device)
		self.copy_network_parameters()
		self.tau = tau
		self.optimizer = optimizer(self.action_network.parameters(), lr = self.lr)
		self.minibatch_size = minibatch_size
		self.actions_mat = None
		self.__init_actions()
		self.n_steps = 0
		self.buffer = PrioritizedReplayBuffer(1000, minibatch_size)

	def __init_actions(self):
		self.actions_mat = three_cartesian_product(self.main_thrust_buckets, \
			self.side_thrust_buckets, self.nozzle_buckets)

	def compute_bucket_assignments(self, a):
		main_thrust = get_closest(a[:,0], self.main_thrust_buckets)
		side_thrust = get_closest(a[:,1], self.side_thrust_buckets)
		nozzle = get_closest(a[:,2], self.nozzle_buckets)
		out = np.zeros((a.shape[0], 3))
		out[:,0] = main_thrust; out[:,1] = side_thrust; out[:,2] = nozzle
		return out

	def get_action_indices(self, a):
		a = self.compute_bucket_assignments(a)
		idx = [np.where(np.all(self.actions_mat == act, axis = 1))[0][0] \
			for act in a]
		return np.array(idx)

	def next_action(self, s):
		with torch.no_grad():
			q_vals = self.action_network.forward(torch.from_numpy(s).float())
		idx = np.argmax(q_vals.numpy())
		return self.actions_mat[idx,:]

	def copy_network_parameters(self):
		self.target_network.load_state_dict(self.action_network.state_dict())

	def update(self, s, a, r, sp):
		# Update counter and push tuple to buffer
		self.n_steps += 1
		self.buffer.push(s, a, r, sp)

		# Proceed only if sufficient samples in the buffer
		if self.buffer.size < self.buffer.warmup:
			return

		# Sample from ER buffer
		s, a, r, sp = self.buffer.sample()
		a_idx = torch.from_numpy(self.get_action_indices(a))

		# Forward pass to compute next state optimal Q-values
		q_vals = self.target_network.forward(torch.from_numpy(sp).float())
		opt_q_vals, opt_actions = torch.max(q_vals, dim = 1)

		# Forward pass to compute Q-values of current states
		cur_q_vals = self.action_network.forward(torch.from_numpy(s).float())
		cur_q_vals = cur_q_vals.gather(1, a_idx.view(-1, 1))

		# Update training network
		td_target = (torch.from_numpy(r).float() + self.discount_rate*opt_q_vals)
		loss = F.smooth_l1_loss(td_target, cur_q_vals)

		self.optimizer.zero_grad()
		loss.backward()
		for param in self.action_network.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		if (self.n_steps % self.tau == 0):
			# Print debugging information if requested
			if self.print_debug:
				print("Network loss: {n:.{d}}".format(n = loss, d = 2))
				grads = [torch.mean(x.grad.data).numpy() for x in self.action_network.parameters()]
				print("Magnitude of gradient updates: {n:.{d}}".format(n = np.mean(grads), d = 2))

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
		return output

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

# Cartesian product of three np.arrays
def three_cartesian_product(a, b, c): 
	out = np.array(np.meshgrid(a, b, c)).T.reshape((-1, 3))
	return out

'''
Class: DiscretizedQAgent

Implements a DQN with fixed Q-targets using neural network function approximators. 
TODO: Add an experience replay buffer as well. 
'''
class DiscretizedQAgent(TDAgent):
	def __init__(self, env, lr, discount, \
			epsilon_start, epsilon_min, epsilon_decay, \
			disc_buckets, predictor, tau, optimizer):
		super().__init__(env, lr, discount, \
				epsilon_start, epsilon_min, epsilon_decay)

		# Let the discretization buckets be tuples of values in sorted order that
		# span the range (lb_val, ub_val) for each element val in an action 3-tuple. 
		# Think of each element in a disc bucket tuple as the mean of its bucket
		self.main_thrust_buckets = np.array(disc_buckets["main_thrust"])
		self.side_thrust_buckets = np.array(disc_buckets["side_thrust"])
		self.nozzle_buckets = np.array(disc_buckets["nozzle"])
		self.action_network = predictor
		self.target_network = predictor
		self.copy_network_parameters()
		self.tau = tau
		self.optimizer = optimizer(self.action_network.parameters(), lr = self.lr)
		self.actions_mat = None
		self.__init_actions()
		self.n_steps = 0

	def __init_actions(self):
		self.actions_mat = three_cartesian_product(self.main_thrust_buckets, \
			self.side_thrust_buckets, self.nozzle_buckets)

	def compute_q_vals(self, s, mode):
		assert(mode in ["target", "action"])
		input_mat = np.zeros((self.actions_mat.shape[0], self.actions_mat.shape[1] + len(s)))
		input_mat[:,:self.actions_mat.shape[1]] = self.actions_mat
		input_mat[:,self.actions_mat.shape[1]:] = s
		input_mat = torch.from_numpy(input_mat).float()
		if mode == "target":
			return self.target_network.forward(input_mat)
		else:
			return self.action_network.forward(input_mat)

	def next_action(self, s):
		q_vals = self.compute_q_vals(s, "action")
		return self.actions_mat[np.argmax(q_vals.detach().numpy()),:]

	def copy_network_parameters(self):
		self.target_network.load_state_dict(self.action_network.state_dict())

	def update(self, s, a, r, sp):
		# Update counters
		self.n_steps += 1

		# Forward pass to compute next state optimal value
		q_vals = self.compute_q_vals(sp, "target").detach().numpy()

		# Forward pass on current (s, a) pair
		cur_q_in = torch.cat((torch.from_numpy(s).float(), torch.from_numpy(a).float()))
		cur_q_val = self.action_network.forward(cur_q_in)

		# Update training network
		td_target = (r + self.discount_rate*np.max(q_vals)
		loss = F.mse_loss(td_target, cur_q_val)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Update target network
		if (self.n_steps % self.tau == 0):
			self.copy_network_parameters()

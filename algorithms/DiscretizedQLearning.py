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
Class: DiscretizedQAgent

Implements a DQN with fixed Q-targets using neural network function approximators. 
'''
class DiscretizedQAgent(TDAgent):
	def __init__(self, env, lr, discount, \
			epsilon_start, epsilon_min, epsilon_decay, print_debug, \
			predictor, tau, optimizer, minibatch_size, device):
		super().__init__(env, lr, discount, \
				epsilon_start, epsilon_min, epsilon_decay, print_debug)

		self.device = device
		if self.device == torch.device("cuda"):
			torch.set_default_tensor_type(torch.cuda.FloatTensor)

		# Function appoximators
		self.action_network = predictor.to(self.device)
		self.target_network = predictor.to(self.device)
		self.copy_network_parameters()

		# Hyperparameters and optimizer; [tau] is the number of _environment steps_ until
		# parameter copies b/w the target and actor networks (~100 steps/episode in SpaceX rocket env)
		self.tau = tau
		self.minibatch_size = minibatch_size
		self.optimizer = optimizer(self.action_network.parameters(), lr = self.lr)

		# Data and counting variables
		self.n_steps = 0
		self.buffer = ReplayBuffer(1000, minibatch_size, self.device)

		# Logger for TensorboardX
		self.writer = SummaryWriter("runs/test")

	def next_action(self, s):
		with torch.no_grad():
			x = self.action_network(torch.from_numpy(s).float().to(self.device))
		actions, __ = self.action_network.get_actions(x)
		return actions.squeeze().numpy()

	def copy_network_parameters(self):
		self.target_network.load_state_dict(self.action_network.state_dict())

	def update(self, s, a, r, sp):
		# Update counter
		self.n_steps += 1

		# Transform action triple into index in actios_mat and push to buffer
		self.buffer.push(s, a, r, sp)

		# Proceed only if sufficient samples in the buffer
		if self.buffer.size < self.buffer.warmup:
			return

		# Sample from ER buffer
		s, a, r, sp = self.buffer.sample()

		# Forward pass to compute next state optimal Q-values
		with torch.no_grad():
			x = self.target_network(sp)
		__, opt_q_vals = self.target_network.get_actions(x)

		# Forward pass to compute Q-values of current states
		x = self.action_network(s)
		cur_q_vals = self.action_network.get_q_val_for_actions(x, a)

		# Compute TD target and loss
		td_target = (r + self.discount_rate*opt_q_vals)
		loss = F.smooth_l1_loss(cur_q_vals, td_target)

		# Update training network
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Log model loss, avg. Q-value, and magnitude of gradient updates to Tensorboard
		grad_mean = torch.mean(torch.Tensor([torch.mean(torch.abs(x.grad.data)) for x in self.action_network.parameters()]))
		q_mean = torch.mean(opt_q_vals)
		self.writer.add_scalar("actor_loss", loss, self.n_steps)
		self.writer.add_scalar("avg_gradient_update_mag", grad_mean, self.n_steps)
		self.writer.add_scalar("avg_sp_optimal_q_value", q_mean, self.n_steps)

		if (self.n_steps % self.tau == 0):
			# Print debugging information if requested
			if self.print_debug:
				print("Network loss: {n:.{d}}".format(n = loss, d = 2))
				print("Avg. magnitude of gradient updates: {n:.{d}}".format(n = grad_mean, d = 2))
				print("Avg. (sp-optimal) q-value: {n:.{d}}".format(n = q_mean, d = 2))

			# Update target network, anneal epsilon, and prune minibatch of old samples. 
			# Arbitrarily we drop 50% of the buffer -- should tune this, but whatever...
			self.copy_network_parameters()
			self.anneal_epsilon()
			self.buffer.clean(0.5)

	def get_pickleable(self):
		output = {
			"action_network" : self.action_network.state_dict(), 
			"target_network" : self.target_network.state_dict(), 
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

import numpy as np 
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from algorithms.Random import RandomAgent

class TDAgent(object):
	def __init__(self, env, lr, discount, \
			epsilon_start, epsilon_min, epsilon_decay):
		self.env = env
		self.lr = lr
		self.discount_rate = discount
		self.epsilon = epsilon_start
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.sampler = RandomAgent(self.env)

	def zero_epsilon(self):
		self.epsilon = 0

	def epsilon_greedy_action(self, s):
		if np.random.rand() <= self.epsilon:
			return self.sampler.next_action(s)
		return self.next_action(s)

	def anneal_epsilon(self):
		if (self.epsilon > self.epsilon_min):
			self.epsilon *= (1 - self.epsilon_decay)

	def next_action(self, s):
		pass

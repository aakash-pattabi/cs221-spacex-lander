import numpy as np 
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from algorithms.Random import RandomAgent

class TDAgent(object):
	def __init__(self, env, exp, lr, discount, epsilon, epsilon_decay, predictor):
		self.env = env
		self.exp = exp
		self.learning_rate = lr
		self.discount_rate = discount
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.predictor = predictor
		self.sampler = RandomAgent(self.env)

	def epsilon_greedy_action(self, s):
		if np.random.rand() < self.epsilon:
			return self.sampler.next_action(s)
		return self.next_action(s)

	def anneal_epsilon(self):
		self.epsilon *= self.epsilon_decay

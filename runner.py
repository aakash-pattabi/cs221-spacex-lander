from rocket_lander.environments.rocketlander import RocketLander
from rocket_lander.constants import *
from algorithms.Random import RandomAgent
from algorithms.DiscretizedQLearning import DiscretizedQAgent
from predictors import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import pickle

'''
Class: RocketRunner
-------------------
Manages test- and train-time simulations for the SpaceX rocket lander environment. Agents, 
environment settings, and learning settings can be changed in the appropriate config files
'''
class RocketRunner(object):
	def __init__(self, env, agent, agent_config, mode, num_episodes, 
		render, verbose, print_every, checkpoint_every):
		# Basic error-checking
		assert mode in ["train", "test"]
		assert print_every > 0

		self.train_mode = (mode == "train")
		self.num_episodes = num_episodes
		self.render = render
		self.verbose = verbose
		self.print_every = print_every
		self.checkpoint_every = checkpoint_every
		self.env = env
		self.episodic_returns, self.episodic_steps = [], []

		# Use agent_config = None to signal that we're passing a pre-trained agent
		if agent_config is not None:
			agent_config["env"] = self.env
			self.agent = agent(**agent_config)
		else:
			self.agent = agent

		# Turn off epsilon-greedy actions at test time
		if not self.train_mode:
			try:
				self.agent.zero_epsilon()
			except AttributeError:
				print("Agent doesn't learn using e-greedy; nothing to zero!")

	def simulate(self):
		try:
			for episode in range(self.num_episodes):
				total_return, steps = 0, 1
				s = self.env.reset()
				done = False

				while not done:
					a = self.agent.epsilon_greedy_action(s)
					sp, r, done, info = self.env.step(a)
					total_return += r
					steps += 1

					if self.train_mode:
						self.agent.update(s, a, r, sp)

					if self.render:
						self.env.render()
						self.env.refresh(render = False)

					if done:
						if self.verbose and (episode % self.print_every == 0) and (episode > 0):
							running_avg = np.mean(self.episodic_returns[-self.print_every:])
							print("Episode #{}: Running mean over epoch {n:.{d}f}".format(episode, \
									n = running_avg, d = 2))
						
						if (episode % self.checkpoint_every == 0) and (episode > 0):
							try:
								runner.save_model("./output/InitialTestNNAgentEpisode{}.pkl".format(episode))
								runner.save_logs("./output/InitialTestNNEpisode{}.csv".format(episode))
							except:
								pass

						break 

					s = sp
				self.episodic_returns.append(total_return)
				self.episodic_steps.append(steps)
		except KeyboardInterrupt:
			pass

	# Save performance curves to .csv
	def save_logs(self, filename):
		df = np.zeros((len(self.episodic_returns), 3))
		df[:,0] = np.arange(len(self.episodic_returns))
		df[:,1] = np.array(self.episodic_steps)
		df[:,2] = np.array(self.episodic_returns)
		np.savetxt(filename, df, delimiter = ",", header = "Episode,Steps,Return")

	# Dump entire agent (including state information) to .pkl
	def save_model(self, filename):
		with open(filename, "wb") as pkl:
			pickle.dump(self.agent.get_pickleable(), pkl)

if __name__ == "__main__":

	settings = {
		'Side Engines' : True,
	    'Clouds' : True,
	    'Vectorized Nozzle' : True,
	    'Starting Y-Pos Constant' : 1,
	    'Initial Force' : 'random'
	}
	env = RocketLander(settings)
	action_size = len(env.action_space)
	state_size = env.state_size

	main_thrust = np.linspace(0, 1, 5)
	side_thrust = np.linspace(-1, 1, 5)
	nozzle = np.linspace(-NOZZLE_ANGLE_LIMIT,NOZZLE_ANGLE_LIMIT, 10)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	agent_config = {
		"env" : env, 
		"lr" : 1e-5, 
		"discount" : 0.95, 
		"epsilon_start" : 0.9, 
		"epsilon_min" : 0.2, 
		"epsilon_decay" : 1e-3, 
		"predictor" : NNPredictor([300, 300, 300, 300, 300, 300], \
			input_size = state_size, output_size = 5*5*10), 
		"disc_buckets" : {
			"main_thrust" : main_thrust, 
			"side_thrust" : side_thrust, 
			"nozzle" : nozzle
		}, 
		"tau" : 5000, 
		"optimizer" : torch.optim.Adam, 
		"minibatch_size" : 32, 
		"print_debug" : False, 
		"device" : device
	}

	args = {
		"env" : env, 
		"agent" : DiscretizedQAgent, 
		"agent_config" : agent_config, 
		"mode" : "train", 
		"num_episodes" : 100000, 
		"render" : False, 
		"verbose" : True, 
		"print_every" : 100, 
		"checkpoint_every" : 100
	}

	runner = RocketRunner(**args)
	runner.simulate()
	runner.save_logs("./output/InitialTestNN.csv")
	runner.save_model("./output/InitialTestNNAgent.pkl")

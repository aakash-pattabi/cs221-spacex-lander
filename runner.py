from rocket_lander.environments.rocketlander import RocketLander
from rocket_lander.constants import *
from algorithms.Random import RandomAgent
from algorithms.DiscretizedQLearning import DiscretizedQAgent
from predictors import *
import numpy as np

'''
Class: RocketRunner
-------------------
Manages test- and train-time simulations for the SpaceX rocket lander environment. Agents, 
environment settings, and learning settings can be changed in the appropriate config files
'''
class RocketRunner(object):
	def __init__(self, env, agent, agent_config, mode, num_episodes, 
		render, verbose, print_every):

		# Basic error-checking
		assert (mode in ["train", "test"])
		assert(print_every > 0)

		self.train_mode = (mode == "train")
		self.num_episodes = num_episodes
		self.render = render
		self.verbose = verbose
		self.print_every = print_every
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
		cumulative_return = 0
		for episode in range(self.num_episodes):
			total_return, steps = 0, 1
			s = self.env.reset()
			done = False

			while not done:
				a = self.agent.next_action(s)
				sp, r, done, info = self.env.step(a)
				total_return += r
				steps += 1

				if self.train_mode:
					self.agent.update(s, a, r, sp)

				if self.render:
					self.env.render()
					self.env.refresh(render = False)

				if done:
					cumulative_return += total_return
					if self.verbose and (episode % self.print_every == 0):
						print("Episode #{}: Avg. reward {n:.{d}f}, return {n2:.{d2}f}".format(episode, \
								n = total_return/steps, d = 2, 
								n2 = total_return, d2 = 2))
					break 

				s = sp
			self.episodic_returns.append(total_return)
			self.episodic_steps.append(steps)

		def save_logs(self, filename):
			df = np.zeros((self.num_episodes, 3))
			df[:,0] = np.arange(self.num_episodes)
			df[:,1] = np.array(self.episodic_steps)
			df[:,2] = np.array(self.episodic_returns)
			np.savetxt(filename, df, delimiter = ",", header = "Episode,Steps,Return")
				
		print("Average return over all episodes: {n:.{d}f}".format(n = cumulative_return/self.num_episodes, d = 2))

if __name__ == "__main__":

	settings = {
		'Side Engines' : True,
	    'Clouds' : True,
	    'Vectorized Nozzle' : True,
	    'Starting Y-Pos Constant' : 1,
	    'Initial Force' : 'random'
	}
	env = RocketLander(settings)
	state_size = len(env.action_space)
	action_size = env.state_size

	agent_config = {
		"env" : env, 
		"lr" : 1e-4, 
		"discount" : 0.95, 
		"epsilon_start" : 0.9, 
		"epsilon_min" : 0.2, 
		"epsilon_decay" : 0.99, 
		"predictor" : NNPredictor([20, 10, 10, 5], state_size, action_size), 
		"disc_buckets" : {
			"main_thrust" : np.linspace(0, 1, 5), 
			"side_thrust" : np.linspace(-1, 1, 5), 
			"nozzle" : np.linspace(-NOZZLE_ANGLE_LIMIT, NOZZLE_ANGLE_LIMIT, 5)
		}, 
		"tau" : 100, 
		"optimizer" : None
	}

	args = {
		"env" : env, 
		"agent" : DiscretizedQAgent, 
		"agent_config" : agent_config, 
		"mode" : "train", 
		"num_episodes" : 10000, 
		"render" : False, 
		"verbose" : True, 
		"print_every" : 100
	}

	runner = RocketRunner(**args)
	runner.simulate()
	runner.save_logs("./output/InitialTestNN.csv")

from rocket_lander.environments.rocketlander import RocketLander
from rocket_lander.constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT
from algorithms.Random import RandomAgent

'''
Class: RocketRunner
-------------------
Manages test- and train-time simulations for the SpaceX rocket lander environment. Agents, 
environment settings, and learning settings can be changed in the appropriate config files
'''
class RocketRunner(object):
	def __init__(self, agent, agent_config, mode, num_episodes, render, settings):
		self.mode = mode
		self.num_episodes = num_episodes
		self.render = render
		self.env = RocketLander(settings)
		agent_config["env"] = self.env
		self.agent = agent(**agent_config)

	def simulate(self):
		if self.mode == "train":
			self.simulate_train()
		elif self.mode == "test":
			self.simulate_test()
		else:
			raise KeyError("Invalid simulation mode: must be one of [train] or [test]")
		
	def simulate_train(self):
		pass

	def simulate_test(self):
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
				self.env.render()
				self.env.refresh(render = False)
				if done:
					cumulative_return += total_return
					print("Episode #{}: Avg. reward {n:.{d}f}, return {n2:.{d2}f}".format(episode, \
							n = total_return/steps, d = 2, 
							n2 = total_return, d2 = 2))
					break 
		print("Average return over all episodes: {n:.{d}f}".format(n = cumulative_return/self.num_episodes, d = 2))

if __name__ == "__main__":
	settings = {
		'Side Engines' : True,
	    'Clouds' : True,
	    'Vectorized Nozzle' : True,
	    'Starting Y-Pos Constant' : 1,
	    'Initial Force' : 'random'
	}

	agent_config = {}

	args = {
		"agent" : RandomAgent, 
		"mode" : "test", 
		"num_episodes" : 10, 
		"render" : True, 
		"settings" : settings, 
		"agent_config" : agent_config
	}

	runner = RocketRunner(**args)
	runner.simulate()

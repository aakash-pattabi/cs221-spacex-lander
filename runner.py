from rocket_lander.environments.rocketlander import RocketLander
from rocket_lander.constants import LEFT_GROUND_CONTACT, RIGHT_GROUND_CONTACT
from algorithms.Random import RandomAgent

class RocketRunner(object):
	def __init__(self, agent, mode, num_episodes, render, settings):
		self.mode = mode
		self.num_episodes = num_episodes
		self.render = render
		self.env = RocketLander(settings)
		self.agent = agent(self.env)

	def simulate(self):
		if self.mode == "train":
			self.simulate_train()
		elif self.mode == "test":
			self.simulate_test()
		else:
			raise ValueError("Invalid simulation mode: must be one of [train] or [test]")
		
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

	config = {
		"agent" : RandomAgent, 
		"mode" : "test", 
		"num_episodes" : 10, 
		"render" : True, 
		"settings" : settings
	}

	runner = RocketRunner(**config)
	runner.simulate()

import numpy as np 


class TemporalDifferenceLearner(object):
	def __init__(self, exp, lr, discount, epsilon, epsilon_decay, predictor):
		self.exp = exp
		self.learning_rate = lr
		self.discount_rate = discount
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.predictor = predictor

	def greedy_action(self, env):
		n_actions = env.action_space.n
		state = env.env.s
		values = []
		for action in range(n_actions):
			s_a = self.predictor.encode_input(action, state, env)
			q = self.predictor.forward(s_a)
			values.append(q)
		return np.argmax(values)

	def epsilon_greedy_action(self, env):
		if np.random.rand() < self.epsilon:
			return (env.action_space.sample())
		return self.greedy_action(env)

	def anneal_epsilon(self):
		self.epsilon *= self.epsilon_decay

	def softmax_action(self, env):
		n_actions = env.action_space.n
		state = env.env.s
		values = []
		for action in range(n_actions):
			s_a = self.predictor.encode_input(action, state, env)
			q = self.predictor.forward(s_a).detach().numpy()
			values.append(q)
		values = np.array(values)
		values -= np.max(values)
		probs = np.exp(values)/np.sum(np.exp(values))
		cum_probs = np.cumsum(probs)
		return np.min(np.where(cum_probs > np.random.rand()))

class SARSALearner(TemporalDifferenceLearner):
	def __init__(self, exp, lr, discount, epsilon, epsilon_decay, predictor):
		TemporalDifferenceLearner.__init__(self, exp, lr, discount, epsilon, epsilon_decay, 
			predictor)

	def update_parameters(self, s, a, r, sp, ap, env):
		## See pg. 10 in Geist and Pietquin (2010), equation #36 
		self.predictor.zero_grad()
		s_a = self.predictor.encode_input(a, s, env)
		Qs_a = self.predictor.forward(s_a)
		Qs_a.backward()

		sp_ap = self.predictor.encode_input(ap, sp, env)
		Qsp_ap = self.predictor.forward(sp_ap)

		td_error = r + self.discount_rate * (Qsp_ap) - Qs_a

		for param in self.predictor.parameters():
			if param.requires_grad:
				param = param + self.learning_rate * param.grad * td_error

class QLearner(TemporalDifferenceLearner):
	def __init__(self, exp, lr, discount, epsilon, epsilon_decay, predictor):
		TemporalDifferenceLearner.__init__(self, exp, lr, discount, epsilon, epsilon_decay, 
			predictor)

	def update_parameters(self, s, a, r, sp, env):
		## See pg. 10 in Geist and Pietquin (2010), equation #41
		self.predictor.zero_grad()
		s_a = self.predictor.encode_input(a, s, env)
		Qs_a = self.predictor.forward(s_a)
		Qs_a.backward()

		all_Qs_a = []
		for action in range(env.action_space.n):
			s_a = self.predictor.encode_input(action, s, env)
			current_Qs_a = self.predictor.forward(s_a)
			all_Qs_a.append(current_Qs_a)

		td_error = r + self.discount_rate * (np.max(all_Qs_a)) - Qs_a

		for param in self.predictor.parameters():
			if param.requires_grad:
				param = param + self.learning_rate * param.grad * td_error

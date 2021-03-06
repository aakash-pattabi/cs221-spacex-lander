import numpy as np 
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from rocket_lander.constants import * 

class RandomAgent(object):
	def __init__(self):
		pass

	def epsilon_greedy_action(self, s):
		return self.next_action(s)

	def next_action(self, s):
		main_thrust = np.random.rand()
		side_thrust = 2*np.random.rand() - 1
		nozzle = 2*NOZZLE_ANGLE_LIMIT*np.random.rand() - NOZZLE_ANGLE_LIMIT
		return np.array([main_thrust, side_thrust, nozzle])
		

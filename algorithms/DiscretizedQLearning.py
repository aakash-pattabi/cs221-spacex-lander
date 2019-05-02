import numpy as np 
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from rocket_lander.constants import * 
from algorithms.TemporalDifference import TDAgent

class DiscretizedQAgent(TDAgent):
	def __init__(self, config):
		super().__init__(**config)
		self.arg = arg
		

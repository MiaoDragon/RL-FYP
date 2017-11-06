from AgentBase import AgentBase
import numpy as np
import math

# this approach keeps a state history of previous 2 states
class Agent(AgentBase):
    def decide(self):
        return math.floor(np.random.random() * 3)+1
    def observe(self, observation, reward):
        pass

    def reset(self, observation):
        pass
    def term(self):
        pass

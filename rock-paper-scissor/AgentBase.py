# abstract class for Agent and Human
from abc import ABC, abstractmethod
class AgentBase(ABC):
    def __init__(self, alpha):
        pass
    @abstractmethod
    def decide(self):
        # set the next action
        pass
    @abstractmethod
    def observe(self, observaion, reward):
        #observation: (self_action, opponent_action)
        pass
    @abstractmethod
    def reset(self, observation):
        pass
    @abstractmethod
    def term(self):
        pass

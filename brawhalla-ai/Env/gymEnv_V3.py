import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym                      # for testing

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class myGym:
    # a wrapper for gym env for testing
    def __init__(self):
        #self.env = gym.make('CartPole-v0').unwrapped
        self.env = gym.make('CartPole-v0')
        self.action_space = self.env.action_space
    def reset(self):
        obs = self.env.reset()
        self.state = obs

    def step(self, action):
        obs, r, done, _ = self.env.step(action)
        #self.env.render()
        self.state = obs
        if done:
            self.env.reset()
        return self.state, r, done

    def get_state(self):
        return self.state

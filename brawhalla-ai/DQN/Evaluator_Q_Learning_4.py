import time
import torch
import multiprocessing
from torch.autograd import Variable
import torch.optim as optim
from DQN.NNcartpole import DQN
from .Plotter import Plotter
import numpy as np
import copy
import os
import random
from Env.gymEnv_V3 import myGym
import math
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class Evaluator:
    def __init__(self, memory, env):
        # hyperparameters
        self.TRAIN_MAX = 1
        self.TRANSFER = 1
        self.BATCH_SIZE = 32
        self.GAMMA = 0.95
        self.SAMPLE_ALPHA = 0.5
        self.SAMPLE_EPISLON = 0.
        self.SAMPLE_BETA = 0.
        self.plotter = Plotter(folder='DQN/plot/cartpole_simple/singleP')
        #LEARNING_RATE = 0.00025
        LEARNING_RATE = 1e-3
        MOMENTUM = 0.95
        SQUARED_MOMENTUM = 0.95
        MIN_SQUARED_GRAD = 0.01

        self.EPS_START = 1.
        self.EPS_END = 0.05
        self.EPS_DECAY = 200   # DECAY larger: slower
        self.MEMORY_SIZE = 5000
        self.steps_done = 0
        self.net = DQN()  # Deep Net
        self.net.setOptimizer(optim.Adam(self.net.parameters(), lr=LEARNING_RATE))
        #self.net.setOptimizer(optim.RMSprop(self.net.parameters(), lr=LEARNING_RATE,
        #                                        momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
        #                                        eps=MIN_SQUARED_GRAD))
        self.memory = memory
        self.env = env

    def behavior_policy(self, state):
        # We can add epislon here to decay the randomness
        # We store the tensor of size 1x1
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            return self.policy(state)
        else:
            return self.env.action_space.sample()
    def policy(self, state):
        # return tensor of size 1x1
        res = self.net(Variable(state, volatile=True).type(FloatTensor)).data
        return res.max(1)[1][0]
        #return res.max(1)[1].view(1,1)
        #return self.net(Variable(state, volatile=True).type(FloatTensor))\
        #                .data.max(1)[1].view(1,1)

    def minibatch(self, exp_replay, pretrain=False):
        batch = random.sample(list(exp_replay), self.BATCH_SIZE)
        unzipped = list(zip(*batch))
        state_batch = Variable(torch.from_numpy(np.array(unzipped[0])).type(FloatTensor), volatile=True)
        # here previously no type conversion, and result in error
        action_batch = Variable(torch.from_numpy(np.array(unzipped[1])).type(LongTensor), volatile=True)
        reward_batch = Variable(torch.from_numpy(np.array(unzipped[2])).type(FloatTensor), volatile=True)
        target_batch = None
        if pretrain:
            # only use reward
            target_batch = reward_batch
        else:
            term_batch = Variable(torch.from_numpy(np.array(unzipped[4])).type(FloatTensor), volatile=True)
            next_state_batch = Variable(torch.from_numpy(np.array(unzipped[3])).type(FloatTensor), volatile=True)
            # here previously no type conversion, and result in error
            next_state_values = self.net(next_state_batch).max(1)[0].unsqueeze(1)
            next_state_values = term_batch * next_state_values
            next_state_values.volatile = False
            target_batch = reward_batch + (self.GAMMA * next_state_values)

        state_batch.volatile= False
        state_batch.requires_grad = True
        action_batch.volatile = False
        target_batch.volatile = False

        return state_batch, action_batch, target_batch


    def run(self):
        i = 0
        while True:
            i = i + 1
            time_t = 0
            while True:
                time_t += 1
                state = self.env.get_state()
                state_torch = torch.from_numpy(state).type(FloatTensor) # convert to torch and normalize
                state_torch = state_torch.unsqueeze(0).type(FloatTensor)
                action = self.behavior_policy(state_torch)
                next_state, r, done = self.env.step(action)  # 0.03s
                if len(self.memory) == self.MEMORY_SIZE:
                    self.memory.pop(0)
                self.memory.append((state, [action], [r], \
                                            next_state, [1-done])
                                )
                if len(self.memory) < 1000:
                    i = 0
                    if done:
                        break
                    else:
                        continue
                #batch_tuple = self.minibatch(self.memory)
                #loss = self.net.optimize(batch_tuple)

                if done:
                    self.steps_done += 1
                    print('episode: {}, score: {}' . format(i, time_t))
                    self.plotter.plot_train_rewards(time_t)
                    break
            if len(self.memory) >= 1000:
                batch_tuple = self.minibatch(self.memory)
                loss = self.net.optimize(batch_tuple)
        self.plotter.terminate()
env = myGym()
env.reset()
evaluator = Evaluator([], env)
evaluator.run()

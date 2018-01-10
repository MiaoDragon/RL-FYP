import torch
import random
import math
import time
from torch.autograd import Variable
from .Plotter import Plotter
from Env.Environment import Environment
import numpy as np
import os
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Improver(object):
    def __init__(self, net, memory_size, memory, shared):
        self.net = net  # Deep Net
        self.memory = memory
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG', 'weights'}
        #self.env = env
        self.plotter = Plotter(folder='DQN/plot/cartpole_simple/exp4')
        self.steps_done = 0

        # hyperparameters:
        self.EPS_START = 1.
        self.EPS_END = 0.05
        self.EPS_DECAY = 50   # DECAY larger: slower
        self.MEMORY_SIZE = memory_size

    def behavior_policy(self, state):
        # We can add epislon here to decay the randomness
        # We store the tensor of size 1x1
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #print('threshold: {}' . format(eps_threshold))

        if sample > eps_threshold:
            return self.policy(state)
        else:
            #return LongTensor([[self.env.action_space.sample()]])
            return self.env.action_space.sample()
    def policy(self, state):
        # return tensor of size 1x1
        res = self.net(Variable(state, volatile=True).type(FloatTensor)).data
        #print('policy values: {}, {}' . format(res[0][0], res[0][1]))
        #print('policy value: {}' . format(res.max(1)[0][0]))
        return res.max(1)[1][0]
        #return res.max(1)[1].view(1,1)
        #return self.net(Variable(state, volatile=True).type(FloatTensor))\
        #                .data.max(1)[1].view(1,1)

    def save_checkpoint(self, state, filename='save/checkpoint_improver.pth.tar'):
        torch.save(state, filename)

    def run(self):
        POPULATE_FLAG = True
        #MEMORY_SIZE = self.shared['memory'].capacity
        MEMORY_SIZE = self.MEMORY_SIZE
        #POPULATE_MAX = MEMORY_SIZE
        POPULATE_MAX = 1
        populate_num = 0
        #PLOT_FREQ = int(MEMORY_SIZE/32)
        PLOT_FREQ = 20
        R = 0. # for plotting
        step = 0
        Repi = 0
        self.env = Environment()
        self.env.set_all()
        self.env.reset()
        last_time = time.time()
        prev_plot = self.steps_done
        pretrain = True
        #PRETRAIN_MAX = self.MEMORY_SIZE // 2
        PRETRAIN_MAX = 2000


        if os.path.isfile('save/checkpoint_improver.pth.tar'):
            print("=> loading checkpoint '{}'".format('save/checkpoint_improver.pth.tar'))
            checkpoint = torch.load('save/checkpoint_improver.pth.tar')
            print("=> loaded checkpoint '{}'"
                  .format('save/checkpoint_improver.pth.tar'))
        else:
            print("=> no checkpoint found at '{}'".format('save/checkpoint_improver.pth.tar'))

        while True:
            if POPULATE_FLAG:
                if pretrain:
                    if populate_num == PRETRAIN_MAX:
                        pretrain = False
                        self.shared['SENT_FLAG'] = 0
                        POPULATE_FLAG = False
                else:
                    if populate_num == POPULATE_MAX:
                        # reset populate num
                        self.shared['SENT_FLAG'] = 0
                        POPULATE_FLAG = False
            else:
                if self.shared['SENT_FLAG']:
                    # evaluator sent the weights
                    print('copying...')
                    self.net.load_state_dict(self.shared['weights'])
                    POPULATE_FLAG = True
                    populate_num = 0
                    # after evaluating the policy for one round, set the epislon smaller
                    next_step = self.steps_done + 1
                    if next_step != 0:  # loop to back
                        # then set eps_threshold to a small value
                        self.steps_done = next_step
            self.save_checkpoint({
                'state_dict': self.net.state_dict(),
            })
            #print('loop took {0} seconds' . format(time.time()-last_time))

            state = self.env.get_state()  # this is to avoid time delay
            last_time = time.time()
            # 0.003s
            state_torch = torch.from_numpy(state).type(FloatTensor) # convert to torch and normalize
            state_torch = state_torch.unsqueeze(0).type(FloatTensor)
            action = self.behavior_policy(state_torch)

            next_state, r, done = self.env.step(action)  # 0.03s

            if len(self.memory) == MEMORY_SIZE:
                self.memory.pop(0)
            self.memory.append((state, [action], [r], \
                                        next_state, [1-done])
                            )
            #print(len(self.memory))

            # 0.001s
            if POPULATE_FLAG:
                populate_num += 1
            # plot the average reward in PLOT_FREQ episodes
            Repi += r
            if done:
                step += 1
                R += Repi
                Repi = 0. # reset episode reward
            if step and self.steps_done > prev_plot:
                print('average rewards after {} train: {}' . format(self.steps_done-prev_plot, R/step))
                self.plotter.plot_train_rewards(R/step)
                # 2-3s
                R = 0.
                step = 0
                prev_plot = self.steps_done

        self.plotter.terminate()

import torch
from torch.autograd import Variable
import time
import math
import random
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class MainProcess(object):
    def __init__(self, inputs, actor, env):
        self.inputs = inputs
        self.demonet = actor
        self.env = env

        self.steps_done = 0
        self.EPS_START = 1.
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000   # DECAY larger: slower
    def behavior_policy(self, state):
        # We can add epislon here to decay the randomness
        # We store the tensor of size 1x1
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        next_step = self.steps_done + 1
        #print('Improver side: {}' . format(torch.current_device()))
        if next_step != 0:  # loop to back
            # then set eps_threshold to a small value
            self.steps_done = next_step
        if sample > eps_threshold:
            return self.policy(state)
        else:
            return LongTensor([[self.env.action_space.sample()]])
    def policy(self, state):
        # return tensor of size 1x1
        return self.demonet(Variable(state, volatile=True).type(FloatTensor))\
                        .data.max(1)[1].view(1,1)
    def run(self):
        steps = 0
        POPULATE_FLAG = True
        POPULATE_MAX = self.inputs['inputs'].capacity
        populate_num = 0
        PLOT_FREQ = 50
        R = 0.
        step = 0
        Repi = 0
        while True:
           # the computing process is blocked at the first layer conv
            if POPULATE_FLAG:
                if populate_num == POPULATE_MAX:
                   # reset populate num
                    self.inputs['SENT_FLAG'] = 0
                    POPULATE_FLAG = False
            else:
                if self.inputs['SENT_FLAG']:
                    self.net.load_state_dict(self.shared['weights'])
                    POPULATE_FLAG = True
                    populate_num = 0

            state = self.env.get_state()
            state = torch.from_numpy(state).type(FloatTensor) / 255.0 # convert to torch and normalize
            state = state.unsqueeze(0).type(FloatTensor)
            action = self.behavior_policy(state)
            next_state, r, done = self.env.step(action[0,0])
            next_state = torch.from_numpy(next_state).type(FloatTensor) / 255.0 # convert to torch and normalize
            next_state = next_state.unsqueeze(0).type(FloatTensor)
            next_action = self.policy(next_state)
            memory = self.inputs['inputs']
            memory.push((state, action, FloatTensor([[r]]), next_state, next_action, FloatTensor([1-done])))

            self.inputs['inputs'] = memory
            if POPULATE_FLAG:
                populate_num += 1
            else:
                time.sleep(10)
            Repi += r
            if done:
                step += 1
                R += Repi
                Repi = 0.
            if step and step % PLOT_FREQ == 0:
                print('average rewards in {} episode: {}' . format(PLOT_FREQ, R/step))
                R = 0.
                step = 0

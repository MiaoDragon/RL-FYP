import torch
import random
import math
import time
from torch.autograd import Variable
from .Plotter import Plotter
import numpy as np
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Improver(object):
    def __init__(self, net, memory_size, memory, shared, env, semaphore):
        self.net = net  # Deep Net
        self.memory = memory
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG', 'weights'}
        self.env = env
        self.semaphore = semaphore
        self.plotter = Plotter()
        self.steps_done = 0

        # hyperparameters:
        self.EPS_START = 1.
        self.EPS_END = 0.05
        self.EPS_DECAY = 200   # DECAY larger: slower
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
            return LongTensor([[self.env.action_space.sample()]])
    def policy(self, state):
        # return tensor of size 1x1
        return self.net(Variable(state, volatile=True).type(FloatTensor))\
                        .data.max(1)[1].view(1,1)

    def run(self):
        POPULATE_FLAG = True
        #MEMORY_SIZE = self.shared['memory'].capacity
        MEMORY_SIZE = self.MEMORY_SIZE
        #POPULATE_MAX = MEMORY_SIZE
        POPULATE_MAX = 1
        populate_num = 0
        #PLOT_FREQ = int(MEMORY_SIZE/32)
        PLOT_FREQ = 5
        # PSEUDO:
        # keep looping:
        #   0. if Populating Flag:
        #         if have polulated enough memory
        #            wake up Evaluator by setting SENT_FLAG to 0
        #            set Populating Flag to 0
        #         Jump to 2
        #
        #   1. check if SENT_FLAG is set
        #         T: copy weights from buffered weights;
        #            set Polulating Flag to 1
        #         F: ignore
        #
        #   2. (ver1: deal with time delay)
        #      get current obs
        #      get action
        #      step one step, and get next obs, reward
        #      get next action
        #      save (obs, action, reward, next obs, next action)
        #
        #   2. (ver2: directly from last obs)
        #      get next obs from the env, using the next action previously
        #      get the next action from the obs
        #      save (obs, action, reward, next obs, next action)
        R = 0. # for plotting
        step = 0
        Repi = 0
        self.env.reset()
        while True:
            if POPULATE_FLAG:
                if populate_num == POPULATE_MAX:
                    # reset populate num
                    self.shared['SENT_FLAG'] = 0
                    POPULATE_FLAG = False
            else:
                if self.shared['SENT_FLAG']:
                    # evaluator sent the weights
                    self.net.load_state_dict(self.shared['weights'])
                    POPULATE_FLAG = True
                    populate_num = 0
                    # after evaluating the policy for one round, set the epislon smaller
                    next_step = self.steps_done + 1
                    if next_step != 0:  # loop to back
                        # then set eps_threshold to a small value
                        self.steps_done = next_step
            state = self.env.get_state()  # this is to avoid time delay
            # 0.003s
            last_time = time.time()
            state = torch.from_numpy(state).type(FloatTensor) # convert to torch and normalize
            state = state.unsqueeze(0).type(FloatTensor)
            action = self.behavior_policy(state)
            next_state, r, done = self.env.step(action[0,0])
            next_state = torch.from_numpy(next_state).type(FloatTensor) # convert to torch and normalize
            next_state = next_state.unsqueeze(0).type(FloatTensor)
            last_time = time.time()
            next_action = self.policy(next_state) # Since we store the memory for evaluation of policy, not behavior policy
            # 0.001s
            self.semaphore.acquire()
            if len(self.memory) == MEMORY_SIZE:
                self.memory.pop(0)
            self.memory.append((state, action, FloatTensor([[r]]), \
                                        next_state, next_action, FloatTensor([[1-done]]))
                            )
            self.semaphore.release()
            # 0.001s
            if POPULATE_FLAG:
                populate_num += 1
            # plot the average reward in PLOT_FREQ episodes
            Repi += r
            if done:
                step += 1
                R += Repi
                Repi = 0. # reset episode reward
            if step and step % PLOT_FREQ == 0:
                print('average rewards in {} episode: {}' . format(PLOT_FREQ, R/step))
                self.plotter.plot_train_rewards(R/step)
                # 2-3s
                R = 0.
                step = 0

        self.plotter.terminate()

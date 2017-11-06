import torch
import random
import math
import time
from torch.autograd import Variable
from .Plotter import Plotter

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Improver(object):
    def __init__(self, net, shared, env):
        self.net = net  # Deep Net
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG', 'weights'}
        self.env = env
        self.plotter = Plotter()
        self.steps_done = 0

        # hyperparameters:
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
        return self.net(Variable(state, volatile=True).type(FloatTensor))\
                        .data.max(1)[1].view(1,1)

    def run(self):
        POPULATE_FLAG = True
        POPULATE_MAX = self.shared['memory'].capacity
        populate_num = 0
        PLOT_FREQ = int(POPULATE_MAX/32)
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
            state = self.env.get_state()  # this is to avoid time delay
            state = torch.from_numpy(state).type(FloatTensor) / 255.0 # convert to torch and normalize
            state = state.unsqueeze(0).type(FloatTensor)
            action = self.behavior_policy(state)
            next_state, r, done = self.env.step(action[0,0])
            next_state = torch.from_numpy(next_state).type(FloatTensor) / 255.0 # convert to torch and normalize
            next_state = next_state.unsqueeze(0).type(FloatTensor)
            next_action = self.policy(next_state) # Since we store the memory for evaluation of policy, not behavior policy
            memory = self.shared['memory']
            memory.push( (state, action, FloatTensor([[r]]), \
                                        next_state, next_action, FloatTensor([[1-done]])) )
            self.shared['memory'] = memory
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
                R = 0.
                step = 0

        self.plotter.terminate()

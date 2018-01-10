import torch
import random
import math
import time
from torch.autograd import Variable
import torch.optim as optim
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class CartPoleDQN(object):
    def __init__(self, net, shared, env):
        self.net = net  # Deep Net
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG', 'weights'}
        self.env = env
        self.steps_done = 0

        # hyperparameters:
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200   # DECAY larger: slower
        LEARNING_RATE = 0.00025
        MOMENTUM = 0.95
        SQUARED_MOMENTUM = 0.95
        MIN_SQUARED_GRAD = 0.01
        self.net.setOptimizer(optim.RMSprop(self.net.parameters(), lr=LEARNING_RATE,
                                                momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
                                                eps=MIN_SQUARED_GRAD))
        self.env.reset()
    def behavior_policy(self, state):
        # We can add epislon here to decay the randomness
        # We store the tensor of size 1x1
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        print('threshold: {}' . format(eps_threshold))
        next_step = self.steps_done + 1
        if next_step != 0:  # loop to back
            # then set eps_threshold to a small value
            self.steps_done = next_step
        if sample > eps_threshold:
            return self.policy(state)
        else:
            return LongTensor([[random.randrange(2)]])
    def policy(self, state):
        # return tensor of size 1x1
        value = self.net(Variable(state, volatile=True).type(FloatTensor))
        #print(value.data.max(1)[0][0])  # print the max Q value
        return value.data.max(1)[1].view(1,1)
        #return self.net(Variable(state, volatile=True).type(FloatTensor))\
        #                .data.max(1)[1].view(1,1)
    def minibatch(self, exp_replay, pretrain=False):
        batch = exp_replay.sample(self.BATCH_SIZE)
        #print(batch)
        unzipped = list(zip(*batch))
        state_batch = Variable(torch.cat(list(unzipped[0])).clone())
        action_batch = Variable(torch.cat(list(unzipped[1])).clone())
        reward_batch = Variable(torch.cat(list(unzipped[2])).clone())

        if pretrain:
            # only use reward
            return state_batch, action_batch, reward_batch
        else:
            term_batch = Variable(torch.cat(list(unzipped[4])).clone(), volatile=True)
            next_state_values = self.net.evaluate(list(unzipped[3])).max(1)[0].unsqueeze(1)
            #non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, list(unzipped[3]))))
            #non_final_next_states = Variable(torch.cat([s for s in list(unzipped[3]) if s is not None]),
            #                                 volatile=True)
            #next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(Tensor))
            #next_state_values[non_final_mask] = self.net(non_final_next_states).max(1)[0]
            #next_state_values.volatile = False
            #print(next_state_values)
            #next_state_values = self.targetNet.evaluate(list(unzipped[3])).max(1)[0].unsqueeze(1)
            next_state_values = term_batch * next_state_values
            next_state_values.volatile = False
            #next_state_values.requires_grad = False
            target_batch = reward_batch + (self.GAMMA * next_state_values)
            return state_batch, action_batch, target_batch

    def run(self):
        MEMORY_SIZE = self.shared['memory'].capacity
        R = 0. # for plotting
        step = 0
        Repi = 0
        PLOT_FREQ = 5
        pretrain = False

        while True:
            state = self.env.get_state()  # this is to avoid time delay
            state = torch.from_numpy(state).type(FloatTensor)
            state = state.unsqueeze(0).type(FloatTensor)
            #print(state)
            action = self.behavior_policy(state)
            next_state, r, done = self.env.step(action[0,0])
            next_state = torch.from_numpy(next_state).type(FloatTensor) # convert to torch and normalize
            next_state = next_state.unsqueeze(0).type(FloatTensor)
            memory = self.shared['memory']
            memory.push( (state, action, FloatTensor([[r]]), \
                                        next_state, FloatTensor([[1-done]])) )
            self.shared['memory'] = memory
            if len(self.shared['memory']) >= self.BATCH_SIZE:
                batch_tuple = self.minibatch(self.shared['memory'], pretrain)
                #print('got batch tuple')
                #print(batch_tuple[0].type)
                loss = self.net.optimize(batch_tuple)
            # plot the average reward in PLOT_FREQ episodes
            Repi += r
            if done:
                print('reward in episode: {}' . format(Repi))
                Repi = 0. # reset episode reward

        """
        while True:
            self.env.reset()
            last_screen = self.env.get_screen()
            current_screen = self.env.get_screen()
            state = current_screen - last_screen

            while True:
                #state = self.env.get_state()  # this is to avoid time delay
                #state = torch.from_numpy(state).type(FloatTensor) # convert to torch and normalize
                #state = state.unsqueeze(0).type(FloatTensor)
                #print(state)
                action = self.behavior_policy(state)
                _, r, done, _ = self.env.env.step(action[0,0])
                #next_state = torch.from_numpy(next_state).type(FloatTensor) # convert to torch and normalize
                #next_state = next_state.unsqueeze(0).type(FloatTensor)
                last_screen = current_screen
                current_screen = self.env.get_screen()
                next_state = current_screen - last_screen
                memory = self.shared['memory']
                memory.push( (state, action, FloatTensor([[r]]), \
                                            next_state, FloatTensor([[1-done]])) )
                self.shared['memory'] = memory
                state = next_state
                if len(self.shared['memory']) >= self.BATCH_SIZE:
                    batch_tuple = self.minibatch(self.shared['memory'], pretrain)
                    #print('got batch tuple')
                    #print(batch_tuple[0].type)
                    loss = self.net.optimize(batch_tuple)
                # plot the average reward in PLOT_FREQ episodes
                Repi += r
                if done:
                    print('reward in episode: {}' . format(Repi))
                    Repi = 0. # reset episode reward
                    break
        """

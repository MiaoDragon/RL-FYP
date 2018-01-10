import time
import torch
import multiprocessing
from torch.autograd import Variable
import torch.optim as optim
from DQN.DQNcartpole import DQN
import numpy as np
import copy
import os
os.system("taskset -p 0xff %d" % os.getpid())
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class Evaluator(multiprocessing.Process):
    def __init__(self, shared, semaphore):
        multiprocessing.Process.__init__(self)
        # hyperparameters
        self.TRAIN_MAX = 50
        self.TRANSFER = 50
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.SAMPLE_ALPHA = 0.5
        self.SAMPLE_EPISLON = 0.
        self.SAMPLE_BETA = 0.
        
        LEARNING_RATE = 0.00025
        MOMENTUM = 0.95
        SQUARED_MOMENTUM = 0.95
        MIN_SQUARED_GRAD = 0.01

        self.net = DQN()  # Deep Net
        self.targetNet = DQN()
        self.copy_weights()
        self.net.setOptimizer(optim.RMSprop(self.net.parameters(), lr=LEARNING_RATE,
                                                momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
                                                eps=MIN_SQUARED_GRAD))
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG'}
        self.semaphore = semaphore

    def minibatch(self, exp_replay, pretrain=False):

        batch = exp_replay.sample(self.BATCH_SIZE)
        #print(batch)
        unzipped = list(zip(*batch))
        #state_batch = torch.from_numpy(np.concatenate(list(unzipped[0])))
        #state_batch = Variable(state_batch)
        #action_batch = torch.from_numpy(np.concatenate(list(unzipped[1])))
        #action_batch = Variable(action_batch)
        #reward_batch = torch.from_numpy(np.concatenate(list(unzipped[2])))
        #reward_batch = Variable(reward_batch, requires_grad=False)
        state_batch = Variable(torch.cat(list(unzipped[0])).clone())
        action_batch = Variable(torch.cat(list(unzipped[1])).clone())
        reward_batch = Variable(torch.cat(list(unzipped[2])).clone(), requires_grad=False)

        if pretrain:
            # only use reward
            return state_batch, action_batch, reward_batch
        else:
            #term_batch = torch.from_numpy(np.concatenate(list(unzipped[5])))
            #term_batch = Variable(term_batch, volatile=True)
            term_batch = Variable(torch.cat(list(unzipped[5])).clone(), volatile=True)
            #next_action_batch = torch.from_numpy(np.concatenate(list(unzipped[4])))
            #next_action_batch = Variable(next_action_batch, volatile=True)
            next_action_batch = Variable(torch.cat(list(unzipped[4])).clone(), volatile=True)
            #next_state = torch.from_numpy(np.concatenate(list(unzipped[3])))
            #next_state = Variable(next_state, volatile=True)
            #next_state_values = self.targetNet(next_state).gather(1, next_action_batch)
            next_state_values = self.targetNet.evaluate(list(unzipped[3])).gather(1, next_action_batch)
            #non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, list(unzipped[3]))))
            #non_final_next_states = Variable(torch.cat([s for s in list(unzipped[3]) if s is not None]),
            #                                 volatile=True)
            #next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(Tensor))
            #next_state_values[non_final_mask] = self.targetNet(non_final_next_states).gather(1, next_action_batch)
            #next_state_values.volatile = False
            #print(next_state_values)
            #next_state_values = self.targetNet.evaluate(list(unzipped[3])).max(1)[0].unsqueeze(1)
            next_state_values = term_batch * next_state_values
            next_state_values.volatile = False
            #next_state_values.requires_grad = False
            target_batch = reward_batch + (self.GAMMA * next_state_values)
            return state_batch, action_batch, target_batch

    def copy_weights(self):
        self.targetNet.load_state_dict(self.net.state_dict())
    def run(self):
        # keep two nets: Q-net, and target-net
        # keep looping:
        #   0. loop until SENT_FLAG is not set
        #
        #   1. loop for a fixed # of steps:
        #         minibatch, and get the target value for the batch
        #         optimize the net parameters by this batch
        #         for some fixed time, copy weights from Q-net to target-net
        #
        #   2. set copy weights from Q-net to shared weights
        #      set SENT_FLAG to true
        # TODO: pretrain in the first loop
        pretrain = True
        while True:
            while self.shared['SENT_FLAG']:
                # loop until it is set to 0
                print('sleeping... size: {}' . format(len(self.shared['memory'])))
                time.sleep(0.1)
            print('training...')
            for step_i in range(1, self.TRAIN_MAX+1):
                # minibatch, and get the target value
                #print('training... step {}' . format(step_i))
                self.semaphore.acquire()
                memory = copy.deepcopy(self.shared['memory'])
                self.semaphore.release()
                if len(memory) < self.BATCH_SIZE:
                    continue

                batch_tuple = self.minibatch(memory, pretrain)
                #print('got batch tuple')
                #print(batch_tuple[0].type)
                loss = self.net.optimize(batch_tuple)
                #print('loss: {}' . format(loss))
                #print('optimized')
                if step_i % self.TRANSFER == 0:
                    self.copy_weights()
            self.shared['weights'] = self.net.state_dict()
            self.shared['SENT_FLAG'] = True

            pretrain = False

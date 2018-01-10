import time
import torch
import multiprocessing
from torch.autograd import Variable
import torch.optim as optim
from DQN.DQNcartpole import DQN
import numpy as np
import copy
import os
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class Evaluator(multiprocessing.Process):
    def __init__(self, memory, shared, semaphore):
        multiprocessing.Process.__init__(self)
        # hyperparameters
        self.TRAIN_MAX = 10
        self.TRANSFER = 10
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
        self.memory = memory
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG'}
        self.semaphore = semaphore

    def minibatch(self, exp_replay, pretrain=False):
        #batch = exp_replay.sample(self.BATCH_SIZE)
        #print(batch)
        unzipped = list(zip(*exp_replay))
        state_batch = Variable(torch.from_numpy(np.array(unzipped[0])), volatile=True)
        action_batch = Variable(torch.from_numpy(np.array(unzipped[1])).type(LongTensor), volatile=True)
        reward_batch = Variable(torch.from_numpy(np.array(unzipped[2])).type(FloatTensor), volatile=True)
        target_batch = None
        #state_batch = Variable(torch.cat(list(unzipped[0])).clone(), volatile=True)
        #action_batch = Variable(torch.cat(list(unzipped[1])).clone(), volatile=True)
        #reward_batch = Variable(torch.cat(list(unzipped[2])).clone(), volatile=True)
        #target_batch = None
        if pretrain:
            # only use reward
            target_batch = reward_batch
        else:
            term_batch = Variable(torch.from_numpy(np.array(unzipped[4])).type(FloatTensor), volatile=True)
            #term_batch = Variable(torch.cat(list(unzipped[4])).clone(), volatile=True)
            #next_action_batch = Variable(torch.cat(list(unzipped[4])).clone(), volatile=True)
            next_state_batch = Variable(torch.from_numpy(np.array(unzipped[3])), volatile=True)
            dist_norm = self.targetNet.getdistance(state_batch, next_state_batch)
            #print('average distance: {}' . format(dist_norm))
            #next_state_values = self.targetNet.evaluate(list(unzipped[3])).max(1)[0].unsqueeze(1)
            next_state_values = self.targetNet(next_state_batch).max(1)[0].unsqueeze(1)
            #prediction_state_values = self.targetNet(state_batch).gather(1, action_batch)
            #not_action_batch = Variable(torch.from_numpy(1-np.array(unzipped[1])).type(LongTensor), volatile=True)
            #prediction_state_nonterm_values = self.targetNet(state_batch).gather(1, not_action_batch)
            #print('term average value: {}' . format(torch.sum((1-term_batch) * prediction_state_values).data[0]/torch.sum(1-term_batch).data[0]))
            #print('nonterm average value: {}' . format(torch.sum((1-term_batch) * prediction_state_nonterm_values).data[0]/torch.sum(1-term_batch).data[0]))
            next_state_values = term_batch * next_state_values

            next_state_values.volatile = False
            target_batch = reward_batch + (self.GAMMA * next_state_values)
        # calculate the probability for each transition
        state_values = self.net(state_batch).gather(1, action_batch)
        probability_batch = torch.pow(torch.abs(target_batch-state_values), self.SAMPLE_ALPHA).squeeze(1)
        print(probability_batch)
        sample_is = torch.multinomial(probability_batch,self.BATCH_SIZE)
        state_batch = state_batch.index_select(0, sample_is)
        action_batch = action_batch.index_select(0, sample_is)
        target_batch = target_batch.index_select(0, sample_is)
        state_batch.volatile= False
        state_batch.requires_grad = True
        action_batch.volatile = False
        target_batch.volatile = False
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
        os.system("taskset -p 0xff %d" % os.getpid())
        pretrain = True
        i = 0
        while True:
            while self.shared['SENT_FLAG']:
                # loop until it is set to 0
                print('sleeping...')
                time.sleep(0.1)
            for step_i in range(1, self.TRAIN_MAX+1):
                # minibatch, and get the target value
                #print('training... step {}' . format(step_i))
                #self.semaphore.acquire()
                #memory = copy.deepcopy(self.memory)
                memory = self.memory
                #self.semaphore.release()
                if len(memory) < self.BATCH_SIZE:
                    continue
                i += 1
                print('training... {}' . format(i))
                batch_tuple = self.minibatch(memory, pretrain)
                loss = self.net.optimize(batch_tuple)
                #print('loss: {}' . format(loss))
                #print('optimized')
                if step_i % self.TRANSFER == 0:
                    #self.semaphore.acquire()
                    self.copy_weights()
                    #self.semaphore.release()
            self.shared['weights'] = self.net.state_dict()
            self.shared['SENT_FLAG'] = True

            pretrain = False

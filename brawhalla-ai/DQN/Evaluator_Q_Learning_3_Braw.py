import time
import torch
import multiprocessing
from torch.autograd import Variable
import torch.optim as optim
from DQN.DQN import DQN
#from DQN.NNcartpole import DQN
import numpy as np
import copy
import os
import random
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
class Evaluator(multiprocessing.Process):
    def __init__(self, memory, shared):
        multiprocessing.Process.__init__(self)
        # hyperparameters
        self.TRAIN_MAX = 10
        self.TRANSFER = 10
        self.BATCH_SIZE = 32
        self.GAMMA = 0.95
        self.SAMPLE_ALPHA = 0.5
        self.SAMPLE_EPISLON = 0.
        self.SAMPLE_BETA = 0.

        #LEARNING_RATE = 0.00025
        LEARNING_RATE = 1e-4
        MOMENTUM = 0.95
        SQUARED_MOMENTUM = 0.95
        MIN_SQUARED_GRAD = 0.01

        self.net = DQN()  # Deep Net
        self.targetNet = DQN()
        self.copy_weights()
        self.net.setOptimizer(optim.Adam(self.net.parameters(), lr=LEARNING_RATE))
        #self.net.setOptimizer(optim.RMSprop(self.net.parameters(), lr=LEARNING_RATE,
        #                                        momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
        #                                        eps=MIN_SQUARED_GRAD))
        self.memory = memory
        self.shared = shared  # shared resources, {'memory', 'SENT_FLAG'}

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
            next_state_values = self.targetNet(next_state_batch).max(1)[0].unsqueeze(1)
            next_state_values = term_batch * next_state_values
            next_state_values.volatile = False
            target_batch = reward_batch + (self.GAMMA * next_state_values)

        state_batch.volatile= False
        state_batch.requires_grad = True
        action_batch.volatile = False
        target_batch.volatile = False

        return state_batch, action_batch, target_batch


    def copy_weights(self):
        self.targetNet.load_state_dict(self.net.state_dict())

    def save_checkpoint(self, state, filename='save/checkpoint_explorer.pth.tar'):
        torch.save(state, filename)
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
        PRETRAIN_MAX = 0
        os.system("taskset -p 0xff %d" % os.getpid())
        pretrain = True
        i = 0
        if os.path.isfile('save/checkpoint_explorer.pth.tar'):
            print("=> loading checkpoint '{}'".format('save/checkpoint_explorer.pth.tar'))
            checkpoint = torch.load('save/checkpoint_explorer.pth.tar')
            self.net.load_state_dict(checkpoint['state_dict'])
            self.targetNet.load_state_dict(checkpoint['state_dict'])
            self.net.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'"
                  .format('save/checkpoint_explorer.pth.tar'))
        else:
            print("=> no checkpoint found at '{}'".format('save/checkpoint_explorer.pth.tar'))
        while True:
            while self.shared['SENT_FLAG']:
                # loop until it is set to 0
                print('sleeping...')
                time.sleep(1.)
            for step_i in range(1, self.TRAIN_MAX+1):
                memory = self.memory
                #print(len(memory))
                #if len(memory) < 1000:
                if len(memory) < self.BATCH_SIZE:
                    continue
                i += 1
                print('training... {}' . format(i))
                batch_tuple = self.minibatch(memory, pretrain)
                loss = self.net.optimize(batch_tuple)
                #print('loss: {}' . format(loss))
                #print('optimized')
                if step_i % self.TRANSFER == 0:
                    self.copy_weights()
                    self.save_checkpoint({
                        'state_dict': self.net.state_dict(),
                        'optimizer' : self.net.optimizer.state_dict(),
                    })
            self.shared['weights'] = self.net.state_dict()
            self.shared['SENT_FLAG'] = True
            if i >= PRETRAIN_MAX:
                pretrain = False

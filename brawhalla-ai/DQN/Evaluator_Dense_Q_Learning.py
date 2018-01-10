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
        self.BATCH_SIZE = 128
        #self.BATCH_SIZE = 5
        self.GAMMA = 0.99
        #self.SAMPLE_ALPHA = 0.5
        #self.SAMPLE_EPISLON = 0.
        #self.SAMPLE_BETA = 0.
        #self.SAMPLE_S = 44.8
        self.SAMPLE_S = 5.0
        self.SAMPLE_Q = 1.0

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
        if pretrain:
            # only use reward
            target_batch = reward_batch
        else:
            term_batch = Variable(torch.from_numpy(np.array(unzipped[4])).type(FloatTensor), volatile=True)
            next_state_batch = Variable(torch.from_numpy(np.array(unzipped[3])), volatile=True)
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
        # calculate distance matrix
        state_feature_batch = self.targetNet.getstate(state_batch)
        inner_product = state_feature_batch.matmul(state_feature_batch.transpose(1,0))
        state_feature_batch_l2 = (state_feature_batch ** 2).sum(dim=1,keepdim=True).expand_as(inner_product)
        distance_matrix = state_feature_batch_l2 + state_feature_batch_l2.transpose(1,0) - 2 * inner_product
        #print('distance state')
        #print(distance_matrix.data)
        # calculate Q value ditance matrix
        # Here use target value to calculate
        Q_dist_matrix = target_batch.expand_as(distance_matrix)
        Q_dist_matrix = Q_dist_matrix - Q_dist_matrix.transpose(1,0) # not absolute value
        Q_dist_matrix = Q_dist_matrix.abs()
        #print('distance q')
        #print(Q_dist_matrix.data)
        # Number[i,j] = Number[i,j] + (D_f[i,j] <= sample_S^2 AND D_Q[i,j] <= sample_Q AND action[i]=action[j])
        # only consider same actions
        Action_Mask = (action_batch.expand_as(distance_matrix)) == (action_batch.transpose(1,0).expand_as(distance_matrix))
        Mask = (distance_matrix.data <= (self.SAMPLE_S)) & (Q_dist_matrix.data <= self.SAMPLE_Q) & Action_Mask.data
        Cluster = []
        #print('mask')
        counter = 0
        while True:
            # clustering by VERTEX-COVER-ALL-VERTEX, always find largest degree
            #print('counter = {}' . format(counter))
            counter += 1

            Number = Mask.sum(dim=1)
            value, indx = Number.max(dim=0)
            #print('indx= {}' . format(indx))
            if value[0] == 0:
                # already empty
                break
            v = Mask[indx]
            #print(v)
            #print(Mask)
            Cluster.append(v)
            # delete vertices
            Delete = v.expand_as(Mask) | v.transpose(1,0).expand_as(Mask)
            Delete = Delete ^ 1
            #Delete = v.transpose(1,0).matmul(v) ^ 1
            #print(Delete)
            Mask = Mask & Delete
        k = len(Cluster)
        Cluster = torch.cat(Cluster)
        #print('cluster')
        #print(Cluster)
        Number = Cluster.sum(dim=1).type(LongTensor)
        probability_batch = torch.ones(k) / float(k)
        cluster_is = torch.multinomial(probability_batch,self.BATCH_SIZE,replacement=True)
        # convert the cluster indices to number of items in each cluster
        Sample_num = torch.eye(k).index_select(0,cluster_is).sum(dim=0).type(LongTensor)
        #N = Cluster[0].size()[0] # number of vertices
        state_sample = []
        action_sample = []
        target_sample = []
        for i in range(k):
            n = Sample_num[i]
            N = Number[i]
            if n == 0:
                continue
            cluster = Cluster[i]
            # get nonzero indices
            v_indices = cluster.nonzero().squeeze(1)
            if n == N:
                # pick up all
                state_sample.append(state_batch.index_select(0, v_indices))
                action_sample.append(action_batch.index_select(0, v_indices))
                target_sample.append(target_batch.index_select(0, v_indices))
                continue
            prob = torch.ones(v_indices.size()) / n
            if n < N:
                # uniformly pick
                v_indices_is = torch.multinomial(prob, n)
                v_indices = v_indices.index_select(0, v_indices_is)
                state_sample.append(state_batch.index_select(0, v_indices))
                action_sample.append(action_batch.index_select(0, v_indices))
                target_sample.append(target_batch.index_select(0, v_indices))
                continue
            # uniformly pick with replacement
            v_indices_is = torch.multinomial(prob, n, replacement=True)
            v_indices = v_indices.index_select(0, v_indices_is)
            state_sample.append(state_batch.index_select(0, v_indices))
            action_sample.append(action_batch.index_select(0, v_indices))
            target_sample.append(target_batch.index_select(0, v_indices))
        state_batch = torch.cat(state_sample)
        action_batch = torch.cat(action_sample)
        target_batch = torch.cat(target_sample)

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
            if i == 50:
                pretrain = False

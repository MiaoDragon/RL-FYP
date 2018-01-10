import torch
from torch import optim
import multiprocessing as mp
from torch.autograd import Variable
import time
from DQN.DQNcartpole import DQN

class MyProcess(mp.Process):
    def __init__(self, inputs):
        mp.Process.__init__(self)
        self.BATCH_SIZE = 32
        self.TRAIN_MAX = 500
        self.TRANSFER = 100
        self.GAMMA = 1.0
        LEARNING_RATE = 0.00025
        MOMENTUM = 0.95
        SQUARED_MOMENTUM = 0.95
        MIN_SQUARED_GRAD = 0.01
        self.demonet = DQN()
        self.targetnet = DQN()
        self.copy_weights()
        self.demonet.setOptimizer(optim.RMSprop(self.demonet.parameters(), lr=LEARNING_RATE,
                                                momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
                                                eps=MIN_SQUARED_GRAD))
        self.inputs = inputs
        #self.demonet.setOptimizer(optim.Adam(params=self.demonet.parameters()))
    def copy_weights(self):
        self.targetnet.load_state_dict(self.demonet.state_dict())
    def minibatch(self, exp_replay, pretrain=False):
        batch = exp_replay.sample(self.BATCH_SIZE)
        unzipped = list(zip(*batch))
        #state_batch = np.concatenate(list(unzipped[0]))
        #state_batch = Variable(torch.from_numpy(state_batch))
        #action_batch = np.concatenate(list(unzipped[1]))
        #action_batch = Variable(torch.from_numpy(action_batch))
        #reward_batch = np.concatenate(list(unzipped[2]))
        #reward_batch = Variable(torch.from_numpy(reward_batch), requires_grad=False)
        state_batch = Variable(torch.cat(list(unzipped[0])).clone())
        action_batch = Variable(torch.cat(list(unzipped[1])).clone())
        reward_batch = Variable(torch.cat(list(unzipped[2])).clone(), requires_grad=False)

        if pretrain:
            # only use reward
            return state_batch, action_batch, reward_batch
        else:
            #term_batch = np.concatenate(list(unzipped[5]))
            #term_batch = Variable(torch.from_numpy(term_batch), volatile=True)
            #next_action_batch = np.concatenate(list(unzipped[4]))
            #next_action_batch = Variable(torch.from_numpy(next_action_batch), volatile=True)
            #next_state_batch = np.concatenate(list(unzipped[3]))
            #next_state_batch = Variable(torch.from_numpy(next_state_batch), volatile=True)
            #next_state_values = self.targetNet(next_state_batch).gather(1,next_action_batch)
            term_batch = Variable(torch.cat(list(unzipped[5]).clone()), volatile=True)
            next_action_batch = Variable(torch.cat(list(unzipped[4]).clone()), volatile=True)
            next_state_values = self.targetNet.evaluate(list(unzipped[3]).clone()).gather(1, next_action_batch)
            next_state_values = term_batch * next_state_values
            print(next_state_values)
            next_state_values.volatile = False
            next_state_values.requires_grad = False
            target_batch = reward_batch + (self.GAMMA * next_state_values)
            return state_batch, action_batch, target_batch

    def run(self):
        pretrain = True
        while True:
            while self.inputs['SENT_FLAG']:
                print('sleeping... size: {}' . format(len(self.inputs['inputs'])))
                time.sleep(1)
            for step_i in range(self.TRAIN_MAX):
                sample = self.minibatch(self.inputs['inputs'], pretrain)
                self.demonet(sample[0])
                print('hello world')
                loss = self.demonet.optimize(sample)
                #time.sleep(1)
                if step_i % self.TRANSFER == 0:
                    self.copy_weights()
            self.shared['weights'] = self.net.state_dict()
            self.shared['SENT_FLAG'] = True

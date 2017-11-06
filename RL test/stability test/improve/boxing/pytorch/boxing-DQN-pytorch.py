# ref: http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import gym
import math
import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import cv2

from DQN import DQN
from ReplayMemory import ReplayMemory
from Plotter import Plotter
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# preprocessing & state representation
def preprocess(image):
    # return shape: 1 x h x w
    image = np.array(image)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, _, _ = cv2.split(img_yuv)
    image = cv2.resize(y, (84,84), interpolation = cv2.INTER_CUBIC)
    image = torch.from_numpy(image).type(FloatTensor) / 255.0 # convert to torch
    return image.unsqueeze(0).type(FloatTensor)  # add one dim, for further use


# q learning
def copy_weights(qNet, targetNet):
    #ref: https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/5
    #mp = list(qNet.parameters())
    #mcp = list(targetNet.parameters())
    #for i in range(len(mp)):
    #    mcp[i].data[:] = mp[i].data[:]
    targetNet.load_state_dict(qNet.state_dict())


# environment
#env = gym.make('Boxing-v0')
#env = gym.make('SpaceInvaders-v0')
# parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.05
EPS_DECAY = 10000000  # larger, then more random
LEARNING_RATE = 0.00025
MOMENTUM = 0.95
SQUARED_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
# define network
qNet = DQN()
targetNet = DQN()
copy_weights(qNet, targetNet)
qNet.setOptimizer(optim.RMSprop(qNet.parameters(), lr=LEARNING_RATE,
                                momentum=MOMENTUM, alpha=SQUARED_MOMENTUM,
                                eps=MIN_SQUARED_GRAD))
if use_cuda:
    qNet.cuda()
    targetNet.cuda()

def policy(state):
    # return a tensor of size 1 x 1
    return qNet(Variable(state, volatile=True).type(FloatTensor))\
                .data.max(1)[1].view(1,1)

steps_done = 0
def behavior_policy(state):
    # return a tensor of size 1 x 1
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    #print('new threshold:')
    #print(eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        return policy(state)
    else:
        return LongTensor([[env.action_space.sample()]])

def getValue(state_batch):
    # get the value of the next state
    # _ x 1
    return targetNet.evaluate(state_batch).max(1)[0].unsqueeze(1)

def minibatch(exp_replay, pretrain=False):
    # return: (state_var, action_var, target_var)
    batch = exp_replay.sample(BATCH_SIZE)
    unzipped = list(zip(*batch))
    state_batch = Variable(torch.cat(list(unzipped[0])))
    action_batch = Variable(torch.cat(list(unzipped[1])))
    reward_batch = Variable(torch.cat(list(unzipped[2])), requires_grad=False)

    if pretrain:
        # only use reward
        return state_batch, action_batch, reward_batch
    else:
        term_batch = Variable(torch.cat(list(unzipped[4])), volatile=True)
        next_state_values = term_batch * getValue(list(unzipped[3]))
        next_state_values = next_state_values.squeeze(-1)
        next_state_values.volatile = False
        next_state_values.requires_grad = False
        target_batch = reward_batch + (GAMMA * next_state_values)
        return state_batch, action_batch, target_batch

# training loop
RDM_START = 30
LOCAL_MEMORY = 4 # size of state_stack
#FREQ_COPY = 100
EXP_SIZE = 10000
FREQ_COPY = 10000
num_trainepi = 1
epilen = 50000000
num_pretrainepi = 1
REPEAT_TIME = 4
exp_replay = ReplayMemory(EXP_SIZE)


def get_state(obs, state_stack):
    # push obs into state_stack, then return the state
    # return: Tensor of shape: 1x4x84x84
    preprocessed = preprocess(obs)
    if len(state_stack) == 0:
        # initialize then push
        state_stack = [preprocessed for i in range(LOCAL_MEMORY)]
    else:
        state_stack.pop(0)
        state_stack.append(preprocessed)
    return torch.cat(state_stack).unsqueeze(0).type(Tensor)


# populate experience

print('@populating experience...')
while len(exp_replay) < exp_replay.capacity:
    state_stack = []
    obs = env.reset()
    state = get_state(obs, state_stack)
    while len(exp_replay) < exp_replay.capacity:
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        next_state = get_state(obs, state_stack)
        exp_replay.push(state, LongTensor([[action]]), Tensor([r]), next_state, FloatTensor([[1-done]]))
        state = next_state
        if done:
            break

plotter = Plotter()

# pretrain using zero next_state_value
print('@@pretraining...')
for i_episode in range(num_pretrainepi):
    state_stack = []
    obs = env.reset()
    state = get_state(obs, state_stack)
    repeat_i = 0
    for i_frame in range(FREQ_COPY):
        if repeat_i % REPEAT_TIME == 0:
            action = behavior_policy(state)
            repeat_i = 0
        #action = env.action_space.sample()
        obs, r, done, _ = env.step(action[0,0])
        next_state = get_state(obs, state_stack)
        exp_replay.push(state, action, Tensor([r]), next_state, FloatTensor([[1-done]]))
        state = next_state
        if repeat_i % REPEAT_TIME == 0:
            batch_tuple = minibatch(exp_replay, pretrain=True)
            loss = qNet.optimize(batch_tuple)
        repeat_i += 1
        #print('loss: {}' . format(loss))
        if done:
            break
# copy network, remember the trained model
qNet.load()
copy_weights(qNet, targetNet)

# train
print('@@@training...')
FREQ_PLT = REPEAT_TIME * (EXP_SIZE/BATCH_SIZE) # plot per 100 frames
steps = 0
#for i_episode in range(num_trainepi):
while steps < epilen:
    print('episode: {}' . format(i_episode))
    i_episode += 1
    state_stack = []
    obs = env.reset()
    state = get_state(obs, state_stack)
    R = 0. # accumulative reward
    loss_cum = 0.
    repeat_i = 0
    #TODO: random start (why useful?)
    # train
    while steps < epilen:
#    for i_frame in range(epilen):
        if repeat_i % REPEAT_TIME == 0:
            action = behavior_policy(state)
            repeat_i = 0
        obs, r, done, _ = env.step(action[0,0])
        next_state = get_state(obs, state_stack)
        exp_replay.push(state, action, Tensor([r]), next_state, FloatTensor([[1-done]]))
        state = next_state
        if repeat_i % REPEAT_TIME == 0:
            batch_tuple = minibatch(exp_replay)
            loss = qNet.optimize(batch_tuple)
            print(torch.sum(qNet.head.weight.data))
            loss_cum = loss
        steps += 1
        if steps % FREQ_COPY == 0:
            copy_weights(qNet, targetNet)
        #print('loss: {}' . format(loss))
        repeat_i += 1
        R += r
        if steps % FREQ_PLT == 0:
            # print the cummulative reward during FREQ_PLT
            plotter.plot_train_rewards(R)
            R = 0. # reset reward
            # print the average loss during FREQ_PLT
            plotter.plot_errors(loss_cum/FREQ_PLT)
            loss_cum = 0.
        if done:
            break
qNet.save_checkpoint({
    'epoch': epoch + 1,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'best_prec1': best_prec1,
    'optimizer' : qNet.optimizer.state_dict(),
}, is_best=False)


# evaluation
num_evalepi = 30
eval_epilen = 50000000
print('@@@@evaluating...')
for i_episode in range(num_evalepi):
    state_stack = []
    obs = env.reset()
    state = get_state(obs, state_stack)
    R = 0. # accumulative reward
    #TODO: random start (why useful?)
    for i_frame in range(eval_epilen):
        action = policy(state)
        obs, r, done, _ = env.step(action[0,0])
        next_state = get_state(obs, state_stack)
        state = next_state
        R += r
        if done:
            break
    print('reward: {}' . format(R))
    plotter.plot_eval_rewards(R)

print('Complete')
env.render(close=True)
env.close()
plotter.terminate()

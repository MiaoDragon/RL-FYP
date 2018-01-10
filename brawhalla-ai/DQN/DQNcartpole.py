import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(16)
        def itself(val):
            return val
        self.bn1 = itself
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        self.bn2 = itself
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.bn3 = itself
        self.head = nn.Linear(448, 2)

    def setOptimizer(self, opt):
        self.optimizer = opt

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def getstate(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(x.size(0),-1)

    def getdistance(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        size = x.size(0)
        diff = x.view(x.size(0), -1) - y.view(y.size(0), -1)
        diff = diff.norm(dim=1).mean().data[0]
        return diff

    def optimize(self, batch):
        # batch[0]: state variable tensor of size _x4x84x84
        # batch[1]: action variable tensor of size _x1
        # batch[1]: target variable tensor of size _x1
        assert self.optimizer != None
        state_batch = batch[0]
        action_batch = batch[1]
        target_batch = batch[2]
        predictions = self.forward(state_batch)
        predictions = predictions.gather(1, action_batch)
        # huber loss
        #loss = F.smooth_l1_loss(predictions, target_batch)
        loss_ = torch.nn.MSELoss()
        loss = loss_(predictions, target_batch)
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1,1) # gradient clipping
        self.optimizer.step()
        return loss.data[0]

    def evaluate(self, batch):
        # batch: state : _ x 18
        return self(Variable(torch.cat(batch), volatile=True))

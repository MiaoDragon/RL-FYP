import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        """
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.hidden = nn.Linear(7*7*64, 512)
        #self.head = nn.Linear(512, 18)
        self.head = nn.Linear(512, 35)
        self.optimizer = None
        """
        self.conv1 = nn.Conv2d(1, 32, kernel_size=10, stride=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=5)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.hidden = nn.Linear(7*7*64, 512)
        #self.head = nn.Linear(512, 18)
        self.head = nn.Linear(512, 35)
        self.optimizer = None

    def setOptimizer(self, opt):
        self.optimizer = opt

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.hidden(x.view(x.size(0),-1)))
        return self.head(x.view(x.size(0), -1))

    def optimize(self, batch):
        # batch[0]: state variable tensor of size _x4x84x84
        # batch[1]: action variable tensor of size _x1
        # batch[1]: target variable tensor of size _x1
        assert self.optimizer != None
        state_batch = batch[0]
        action_batch = batch[1]
        target_batch = batch[2]
        predictions = self(state_batch).gather(1, action_batch)
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

    def load(self, filename='model/checkpoint.pth.tar'):
        if os.path.isfile(filename):
            print('=> loading checkpoint {}' . format(filename))
            checkpoint = torch.load(filename)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {} (epoch {})' . format(filename, checkpoint['epoch']))
        else:
            print('=> no checkpoint found at {}' . format(filename))

    def save_checkpoint(self, state, is_best=False, filename='model/checkpoint.pth.tar'):
        # ref:https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model/model_best.pth.tar')

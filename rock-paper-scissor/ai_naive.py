from AgentBase import AgentBase
import numpy as np
import math

# this approach keeps a state history of previous 2 states
class Agent(AgentBase):
    def __init__(self, alpha=0.2):
        self.pred_table = [
                            [0.0 for i in range(3)],
                            [[0.0 for i in range(3)] for j in range(9)],
                            [[[0.0 for i in range(3)] for j1 in range(9)] for j2 in range(9)],
                            #[[[[0.0 for i in range(3)] for j1 in range(9)]for j2 in range(9)] for j3 in range(9)],
                            #[[[[[0.0 for i in range(3)] for j1 in range(9)]for j2 in range(9)] for j3 in range(9)] for j4 in range(9)]
                          ]
        self.pred_times = [
                            [[0 for i in range(3)] for j in range(1)],
                            [[0 for i in range(3)] for j in range(9)],
                            #[[0 for i in range(3)] for j in range(81)],
                            #[[0 for i in range(3)] for j in range(729)],
                            #[[0 for i in range(3)] for j in range(6561)]
                          ]
        self.state_prob = [0.0 for j in range(9)]
        self.obs = [] # a window with length 4
                      # the neareast experience has the smallest index
        self.alpha = alpha # as the distribution is not stationary
        self.f = open('agent-alpha{0}.txt' . format(alpha), 'w')
    def decide(self):
        pred = self.pred_table[len(self.obs)]
        for i in range(len(self.obs)):
            #print(self.obs[i])
            pred = pred[self.obs[i][0]*3+self.obs[i][1]]
        #print(pred)
        prob = [0.,0.,0.]
        prob[0] = pred[0]
        prob[1] = pred[1]
        prob[2] = pred[2]
        if prob[0] == prob[1] and prob[1] == prob[2]:
            # choose randomly
            return math.floor(np.random.random() * 3)+1
        # this is the prediction
        # find the max
        maxi = 0
        if prob[1] > prob[maxi]:
            maxi = 1
        if prob[2] > prob[maxi]:
            maxi = 2
        if maxi != 2:
            return maxi+1+1
        else:
            return 0+1

    def observe(self, observation, reward):
        # put observation input the obs list
        # calculate w.r.t. the new experience
        observation = (observation[0]-1, observation[1]-1)

        (_, a) = observation # opponent
        for i in range(len(self.obs)+1):
            pred = self.pred_table[i]
            #print(pred)
            #num = self.pred_times[i]
            for j in range(i):
                pred = pred[self.obs[j][0]*3+self.obs[j][1]]
            #    num = num[self.obs[j]]
            # in python, we can directly edit the sublinks
            for j in range(3):
                if j == a:
                    r = 1
                else:
                    r = 0
                pred[j] = pred[j] + self.alpha * (r - pred[j]) # alpha since non-stationary
            #print(self.pred_table[i])
        # update experience buffer
        if len(self.obs) < 2:
            self.obs = [observation] + self.obs
        else:
            self.obs.pop()
            self.obs = [observation] + self.obs

    def reset(self, observation):
        self.obs = []
        self.f.write('{0}\n' . format(self.pred_table[0]))
        self.f.write('{0}\n' . format(self.pred_table[1]))
        self.f.write('{0}\n' . format(self.pred_table[2]))
        self.f.write('\n')
    def term(self):
        self.f.close()

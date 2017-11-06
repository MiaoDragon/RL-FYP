# update by E[r+V(s')]
# using bootstrapping
# model-based approach (already know the model)
from env import Env
from plotting import learning_curve
from scipy.stats import bernoulli
import numpy as np
def DynamicProgramming(env, pi, T=100):
    print('policy to evaluate: ')
    print(pi)
    Vs = [[0.] for i in range(env.num_state)]
    ts = [0]
    for t in range(1,T):
        ts.append(t)
        for s in range(env.num_state):
            # update
            if s in env.t_states:
                Vs[s].append(0.)
                continue
            a = pi[s]
            V = 0.
            for ns in range(env.num_state):
                V += (env.reward[s][a][ns]+Vs[ns][t-1])*env.prob[s][a][ns]
            Vs[s].append(V)
    #Vsnp = np.array(Vs)
    tsnp = np.array(ts)
    learning_curve(ts, Vs,'learning_curve_DP.png')




Graph = [[[{'state':1,'reward':-1,'prob':0.5},{'state':2,'reward':-1,'prob':0.5}],
            [{'state':2,'reward':-1,'prob':0.5},{'state':0,'reward':-1,'prob':0.5}]],

            [[{'state':3,'reward':-1,'prob':0.5},{'state':2,'reward':-1,'prob':0.5}],
            [{'state':3,'reward':-1,'prob':0.5},{'state':2,'reward':-1,'prob':0.5}]],

            [[{'state':4,'reward':-1,'prob':0.5},{'state':5,'reward':-10,'prob':0.5}],
            [{'state':4,'reward':-1,'prob':0.5},{'state':5,'reward':-10,'prob':0.5}]],

            [[{'state':3,'reward':-1,'prob':0.5},{'state':6,'reward':10,'prob':0.5}],
            [{'state':3,'reward':-1,'prob':0.5},{'state':6,'reward':10,'prob':0.5}]],

            [[{'state':6,'reward':10,'prob':0.5},{'state':2,'reward':-1,'prob':0.5}],
            [{'state':6,'reward':10,'prob':0.5},{'state':2,'reward':-1,'prob':0.5}]],

            [[{'state':5,'reward':-10,'prob':1.0}],
            [{'state':5,'reward':-10,'prob':1.0}]],

            [[{'state':6,'reward':10,'prob':1.0}],
            [{'state':6,'reward':10,'prob':1.0}]],
        ]
t_states = [5,6]
env = Env(Graph, t_states, 7, 2)
#pi = []
#for i in range(env.num_state):
#    pi.append(bernoulli(0.5).rvs())
pi = [0,1,0,0,1,1,1]
DynamicProgramming(env, pi, 20)

# this program uses policy iteration method to solve cartpole-v0 problem
# continuous to discrete states: [x>0?, v>0?, theta>0?, omega>0?]
# update: the update value should be the expectation of the new sample,
import gym
import math
import numpy
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
gamma = 1
plt_n = 0

def update_line(hl, pt):
    hl.set_xdata(np.append(hl.get_xdata(), pt[0]))
    hl.set_ydata(np.append(hl.get_ydata(), pt[1]))
    plt.gca().relim()
    plt.gca().autoscale(True)
    plt.draw()
def boolToN(l):
    # convert the boolean vector to a number
    return reduce(lambda x,y: x*2+y, l)
def policyEval(delta, Pi, env):
    # this evaluates the given policy, and return the values got
    # delta: parameter for stop
    plt.clf()
    hl, = plt.plot([],[])
    plt.xlabel('evaluation iterations')
    plt.ylabel('max difference value')
    print('***policy Evaluation***')
    T = 50  # random sampling step
    alpha = 1.0 # TD parameter. 1 since the probability information should be maintained
    n = env.observation_space.shape[0]
    m = env.action_space.n
    V = [0.0] * (1<<n)
    times = [0 for i in range(1<<n)]  # appearance
    # since bounded time, value is bounded, no need to set upper bound
    diff = [2 * delta] * (1 << n) # update: should be the difference w.r.t. the Q value
    diff_abs = [2*delta] * (1 << n)
    flag = False
    i = 0
    while not flag:
        # repeat until the max of diff < delta && all diff are calculated
        # since the random init of the env, assume it reaches all states
        i += 1
        print('this is the {0}-th loop in evaluation'.format(i))
        print(max(diff_abs))
        update_line(hl, (i, max(diff_abs)))
        obs = env.reset()
        # in order to explore all obss, simulate the algorithm in the textbook by
        # first sampling with a upper bound time limit
        randT = int(numpy.random.random() * T)
        done = False
        for t in range(randT):
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            if done:
                break
        state = boolToN(list(map(lambda x: x > 0, obs)))

        # update
        if done:
            times[state] += 1
            # since
            Q = 0
            diff[state] = 1.0/times[state] * (Q - V[state])
            diff_abs[state] = abs(diff[state])
            # now diff = new V - old V = 1/times * (sample - old V)
            # we are actually adding this difference to V
            V[state] = V[state] + diff[state]
            if max(diff_abs) < delta:
                flag = True
                break
            else:
                continue
        while True:
            #env.render()
            action = Pi[state] # Pi is a function, but simplified to be an array here
            obs, reward, done, info = env.step(action)
            next_state = boolToN(list(map(lambda x: x > 0, obs)))
            if done:
                Q = reward + 0
                times[state] += 1
                diff[state] = 1.0/times[state] * (Q - V[state])
                diff_abs[state] = abs(diff[state])
                V[state] = V[state] + diff[state]
                if max(diff_abs) < delta:
                    flag = True
                break
            else:
                Q = reward + gamma * V[next_state]
                times[state] += 1
                diff[state] = 1.0/times[state] * (Q - V[state])
                diff_abs[state] = abs(diff[state])
                V[state] = V[state] + diff[state]
                if max(diff_abs) < delta:
                    flag = True
                state = next_state
    global plt_n
    plt.savefig('curve' + str(plt_n) + '.png')
    plt_n += 1
    return V

def policyImp(V, env):
    # this given previous values, conduct a new policy
    # we do this by estimating the expected value of Q_Pi
    print('***policy Improve***')
    epi = 10000
    n = env.observation_space.shape[0]
    m = env.action_space.n
    Q = [[0.0 for i in range(m)] for j in range(1<<n)]
    Q = [[0.0 for i in range(m)] for j in range(1<<n)] # (s,a)
    times = [[0 for i in range(m)] for j in range(1<<n)]  # appearance
    obs = env.reset()
    # sample by random actions
    # here need not use running average
    for _ in range(epi):
        obs = env.reset()
        while True:
            #print('Q value: {}' . format(Q))
            #env.render()
            state = boolToN(list(map(lambda x: x > 0, obs)))
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                Qnew = reward + 0
                times[state][action] += 1
                Q[state][action] = Q[state][action] + 1./times[state][action] * (Qnew-Q[state][action])
                break
            else:
                next_state = boolToN(list(map(lambda x: x > 0, obs)))
                Qnew = reward + gamma * V[next_state]
                times[state][action] += 1
                Q[state][action] = Q[state][action] + 1./times[state][action] * (Qnew-Q[state][action])
    # after sampling, choose the maximal action
    print('Q value: {}' . format(Q))
    Pi = [0 for i in range(1<<n)] # policy
    for i in range(len(V)):
        maxj = 0
        for j in range(m):
            if Q[i][j] > Q[i][maxj]:
                maxj = j
        Pi[i] = maxj
    return Pi

def policyIter(T):
    # T: parameter of iterations
    env = gym.make('CartPole-v0')
    env.reset()
    n = env.observation_space.shape[0]
    # init Pi
    Pi = [round(numpy.random.random()) for i in range(1<<n)]  # random policy
    for i in range(T):
        V = policyEval(0.01, Pi, env)
        print('Values after {0}-th iteration:'.format(V))
        Pi = policyImp(V, env)
    return Pi

Pi = policyIter(T=20)
print('Pi: {0}'.format(Pi))
env = gym.make('CartPole-v0')
env.reset()
for i_episode in range(10):
    obs = env.reset()
    for t in range(100):
        env.render()
        #print(obs)
        #action = env.action_space.sample()
        state = boolToN(list(map(lambda x: x > 0, obs)))
        action = Pi[state]
        obs, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {0} timesteps'.format(t+1))
            break

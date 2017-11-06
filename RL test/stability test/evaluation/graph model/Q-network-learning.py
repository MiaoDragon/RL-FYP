# # update by E[r+V(s')]
# using bootstrapping
# model-based approach (already know the model)
# simple linear regression model, to see the convergence rate
import tensorflow as tf
from env import Env
import numpy as np
from plotting import learning_curve
def Qnetwork(env,pi,insize,outsize, T=1000):
    def convertToBit(a, outsize):
        i = 0
        ret = []
        while i < outsize:
            ret.append(a & 1)
            a = a >> 1
            i += 1
        return ret

    learning_rate = 0.01
    reply_maximum = 100
    # here as the training data is not large, just use normal GD
    experience_reply = []

    x = tf.placeholder(shape=[None,insize+outsize+1], dtype=tf.float32)
    # one for bais
    w = tf.random_uniform(shape=[insize+outsize+1,1], minval=0.000, maxval=0.000, dtype=tf.float32)
    w = tf.Variable(w)
    Q = tf.matmul(x,w)
    nextQ = tf.placeholder(shape=[None,1],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ-Q))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    updateModel = trainer.minimize(loss)
    init = tf.initialize_all_variables()

    # parameter
    e = 0.1 # exploration rate
    ts = [0]
    Vs = [[0.] for i in range(env.num_state)]
    def findMaxA(env,s,outsize):
        #maxA = 0
        #maxQ = -10000
        #for i in range(env.num_action):
        #    a = convertToBit(i,outsize)
        #    Qv = sess.run([Q],feed_dict={x:np.array(sbit+a+[1])})
        #    if Qv > maxQ:
        #        Q = Qv
        #        maxA = i
        a = pi[s]
        sbit = convertToBit(s,insize)
        abit = convertToBit(a,outsize)
        Qv = sess.run(Q,feed_dict={x:np.reshape(np.array(sbit+abit+[1]), (1,insize+outsize+1) )})
        return (float(Qv),a)

    with tf.Session() as sess:
        sess.run(init)
        # populate experience_reply


        while len(experience_reply) < reply_maximum:
            s = env.reset()
            while True:
                # choose action
                _,a = findMaxA(env,s,outsize)
                if np.random.rand(1) < e:
                    elements = [i for i in range(env.num_action)]
                    probs = [1./env.num_action for i in range(env.num_action)]
                    a = np.random.choice(elements, p=probs)
                nexts,r,term = env.step(a)
                if len(experience_reply) == reply_maximum:
                    break
                experience_reply.append( (s,a,r,nexts) )
                s = nexts
                if term:
                    break

        for t in range(T):
            s = env.reset()
            print('{0}th iteration' . format(t))
            while True:
                sbit = convertToBit(s,insize)
                rAll = 0
                # choose action
                _,a = findMaxA(env,s,outsize)
                if np.random.rand(1) < e:
                    # random
                    elements = [i for i in range(env.num_action)]
                    probs = [1./env.num_action for i in range(env.num_action)]
                    a = np.random.choice(elements, p=probs)

                nexts,r,term = env.step(a)
                #print(nexts)
                #print(term)
                # store (s,a,r,nexts) into experience replay
                if len(experience_reply) == reply_maximum:
                    experience_reply.pop(0)
                experience_reply.append( (s,a,r,nexts) )
                # train
                state_actions = []
                nextQs = []
                for exp in experience_reply:
                    s = convertToBit(exp[0],insize)
                    a = convertToBit(exp[1], outsize)
                    nextS = convertToBit(exp[3], insize)
                    state_actions.append(np.array(s+a+[1]))
                    nq,_ = findMaxA(env,exp[3],outsize)
                    if exp[3] in env.t_states:
                        # end of state, Q value is 0
                        # if leave this as free variable, then will not converge
                        nq = 0.
                    nextQs.append(exp[2]+nq)
                    #print(state_actions)
                #print('experience reply:')
                #print(experience_reply)
                #print('nextQ:')
                #print(nextQs)
                _,W,J  = sess.run([updateModel,w,loss],feed_dict={x:np.array(state_actions), nextQ:np.reshape(np.array(nextQs), (len(nextQs),1) )})
                #print(J)
                #print(Qv)

                s = nexts
                if term:
                    break
            ts.append(t)
            # get the V value for each state
            for s in range(env.num_state):
                sbit = convertToBit(s,insize)
                maxQ,_ = findMaxA(env,s,outsize)
                Vs[s].append(maxQ)
        #print(ts)
        learning_curve(ts, Vs,'learning_curve_Qnetwork.png')





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
Qnetwork(env,pi,insize=3,outsize=1, T=10000)
